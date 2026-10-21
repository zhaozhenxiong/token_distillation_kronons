#!/usr/bin/env python3
"""Explicit fresh active-cohort generation, four-arm training, evaluation and analysis."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from token_distillation.artifacts import sha256_file, write_json
from token_distillation.config import load_config

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--population', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--report-dir', required=True)
    p.add_argument('--freeze', required=True)
    p.add_argument('--count', type=int, default=800)
    a = p.parse_args()
    c = load_config(a.config)
    out, dataset, report = map(c.output_path, (a.output_dir, a.dataset, a.report_dir))
    population = json.loads(c.output_path(a.population).read_text())
    symbols = [r['symbol'] for r in population['main']]
    from select_active_population import board, BOARDS
    if len(symbols) != a.count or len(set(symbols)) != a.count or any(board(s) not in BOARDS for s in symbols):
        raise ValueError('Exact unique non-STAR/non-Beijing population required')
    if population['selection_uses_post2023_data'] or population['cutoff'] != '2023-12-29':
        raise ValueError('Pretraining selection required')
    if any(path.exists() for path in (out, dataset, c.output_path(str(a.dataset)+'_parts'), report)):
        raise FileExistsError('Fresh study outputs required; no implicit old-task resumption')
    if shutil.disk_usage(c.root).free < 60 * 1024**3:
        raise ValueError('At least 60 GiB free disk required')
    freeze = json.loads(c.output_path(a.freeze).read_text())
    def check_frozen():
        for name, expected in freeze['file_hashes'].items():
            if sha256_file(Path(name)) != expected:
                raise ValueError('Frozen input changed: ' + name)
    check_frozen()
    import torch
    if not torch.backends.mps.is_available():
        raise RuntimeError('Explicit MPS requested but unavailable')
    out.mkdir(parents=True)
    (out/'logs').mkdir()
    start = time.perf_counter()
    status = {'schema':'active_stock_study_v1', 'status':'running', 'created_at':datetime.now(timezone.utc).isoformat(),
              'count':a.count, 'population_sha256':sha256_file(c.output_path(a.population)),
              'freeze_sha256':sha256_file(c.output_path(a.freeze)), 'phases':[],
              'arms':['none','czsc','ict','dual'], 'seeds':[42,123,2026], 'epochs_requested':2,
              'excluded_boards':['star','beijing'], 'model_promotion':False}
    def save():
        write_json(c,out/'study_status.json',status,replace=(out/'study_status.json').exists())
        items=''.join('<li>'+html.escape(row['phase']+': '+row['status'])+'</li>' for row in status['phases'])
        (out/'progress.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><meta http-equiv="refresh" content="60"><title>800股训练进度</title><style>body{font:18px/1.7 system-ui;max-width:900px;margin:40px auto;padding:20px}a{color:#176b86}</style><h1>800股量价活跃样本消融</h1><p>沪深主板与创业板；排除科创板、北交所。四组×三种子，训练两遍。</p><p>状态：'+html.escape(status['status'])+'</p><ul>'+items+'</ul><p><a href="study_status.json">流程状态 JSON</a> · <a href="ablation/experiment_manifest.json">逐组训练评估状态</a> · <a href="../../reports/'+html.escape(report.name)+'/index.html">完成后的评估报告</a></p><p>页面每分钟刷新。阶段开始不表示训练完成；数值结论以完成后的评估报告为准。</p></html>')
    def child(label, arguments):
        check_frozen()
        row={'phase':label,'status':'running','started_at':datetime.now(timezone.utc).isoformat(),
             'command':[sys.executable,*map(str,arguments)]}
        status['phases'].append(row)
        save()
        print('study starting '+label, file=sys.stderr, flush=True)
        try:
            with (out/'logs'/(label+'.json')).open('x') as log:
                subprocess.run(row['command'],cwd=c.root,stdout=log,check=True)
        except BaseException:
            row['status']='failed'
            raise
        row.update(status='completed',completed_at=datetime.now(timezone.utc).isoformat())
        save()
        print('study completed '+label,file=sys.stderr,flush=True)
    try:
        save()
        child('dataset_generation',[c.root/'scripts/build_stratified_dataset.py','--config',c.config_path,
              '--population',c.output_path(a.population),'--output',dataset,'--workers','4'])
        manifest=json.loads((dataset/'manifest.json').read_text())
        if sorted(manifest['symbols']) != sorted(symbols) or any(v['row_count'] < 1 for v in manifest['splits'].values()):
            raise ValueError('Dataset population or chronological stages do not match')
        steps=math.ceil(2*manifest['splits']['train']['row_count']/4)
        interval=math.ceil(steps/4)
        status['formal_budget']={'max_steps':steps,'evaluation_interval':interval,'batch_size':4,'evaluation_batch_size':32,
                                'train_rows':manifest['splits']['train']['row_count'],
                                'split_counts':{k:v['row_count'] for k,v in manifest['splits'].items()}}
        save()
        child('token_cache',[c.root/'scripts/prepare_token_cache.py','--config',c.config_path,'--dataset',dataset,
              '--output-dir',c.output_path(c.values['training']['token_cache']),'--device','mps'])
        experiment=out/'ablation'
        child('four_arm_three_seed_training_and_evaluation',[c.root/'run.py','--config',c.config_path,'run-ablation',
              '--dataset',dataset,'--output-dir',experiment,'--device','mps','--seeds','42','123','2026',
              '--max-steps',steps,'--batch-size','4','--evaluation-interval',interval,'--evaluation-batch-size','32'])
        child('analysis',[c.root/'run.py','--config',c.config_path,'analyze-ablation','--dataset',dataset,
              '--experiment-dir',experiment,'--output-dir',report])
        child('adapter_diagnostics',[c.root/'scripts/diagnose_ablation_adapters.py','--config',c.config_path,
              '--experiment-dir',experiment,'--output',report/'adapter_diagnostics.json'])
        check_frozen()
        status.update(status='completed',completed_at=datetime.now(timezone.utc).isoformat(),wall_seconds=time.perf_counter()-start,
                      frozen_inputs_postcheck_passed=True,report=str(report/'index.html'))
        save()
    except BaseException as exc:
        status.update(status='failed',error=str(exc),completed_at=datetime.now(timezone.utc).isoformat(),wall_seconds=time.perf_counter()-start)
        save()
        raise
    finally:
        write_json(c,c.output_path('audit')/(out.name+'_terminal.local.json'),status)
    print(json.dumps(status,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
