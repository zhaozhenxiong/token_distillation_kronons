#!/usr/bin/env python3
"""Explicit sequential 80-stock controls then 240-stock 4x3 research study."""
import argparse
from datetime import datetime,timezone
import json,math,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from token_distillation.config import load_config
from token_distillation.artifacts import write_json,sha256_file
from token_distillation.training.ablation import run_ablation

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/ablation240.toml');p.add_argument('--dataset',default='data/ablation240_20261021');p.add_argument('--output-dir',default='artifacts/runs/study240_20261021');a=p.parse_args();c=load_config(a.config)
    output=c.output_path(a.output_dir)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);(output/'logs').mkdir();start=time.perf_counter()
    status={'schema':'explicit_240_study_v1','created_at':datetime.now(timezone.utc).isoformat(),'status':'running','config_sha256':sha256_file(c.config_path),'pipeline_sha256':sha256_file(Path(__file__)),'phases':[],'full_universe_training':False,'promoted':False}
    def save():write_json(c,output/'study_status.json',status,replace=(output/'study_status.json').exists())
    def child(label,arguments):
        entry={'phase':label,'status':'running','command':[sys.executable,*map(str,arguments)],'started_at':datetime.now(timezone.utc).isoformat()};status['phases'].append(entry);save();print('study starting '+label,file=sys.stderr,flush=True)
        with (output/'logs'/(label+'.json')).open('x') as out:
            # Inherit stderr so accelerator / optimizer progress remains live.
            result=subprocess.run(entry['command'],cwd=c.root,stdout=out)
        if result.returncode:raise RuntimeError(f'{label} failed, exit={result.returncode}')
        entry.update(status='completed',completed_at=datetime.now(timezone.utc).isoformat());save();print('study completed '+label,file=sys.stderr,flush=True)
    try:
        dataset=c.output_path(a.dataset);main_manifest=json.loads((dataset/'manifest.json').read_text());mechanism=c.output_path(str(a.dataset)+'_mechanism80');small_manifest=json.loads((mechanism/'manifest.json').read_text())
        if len(main_manifest['symbols'])!=240 or len(small_manifest['symbols'])!=80:raise ValueError('Frozen 240/80 population required')
        child('token_cache',[c.root/'scripts/prepare_token_cache.py','--config',c.config_path,'--dataset',dataset,'--output-dir',c.output_path(c.values['training']['token_cache']),'--device','mps'])
        n=small_manifest['splits']['train']['row_count'];steps=math.ceil(2*n/4);interval=math.ceil(steps/4)
        technical=[]
        variants=[(arm,'none',arm) for arm in ('none','czsc','ict','dual')]+[('dual','shuffle','dual_shuffled'),('dual','frozen_backbone','heads_only'),('dual','none','dual_repeat')]
        for arm,control,label in variants:
            run=output/('mechanism80_'+label)
            child('mechanism80_'+label,[c.root/'run.py','--config',c.config_path,'train','--dataset',mechanism,'--output-dir',run,'--teacher-mode',arm,'--mechanism-control',control,'--device','mps','--seed','42','--max-steps',steps,'--batch-size','4','--evaluation-interval',interval,'--evaluation-batch-size','32'])
            summary=json.loads((run/'training_summary.json').read_text())
            if summary['optimizer_steps']!=steps or summary['peak_rss_bytes']>24*1024**3:raise ValueError('80-stock budget/memory gate failed')
            if not summary['tokenizer_prefix_check']['prefix_token_ids_exact']:raise ValueError('Tokenizer prefix gate failed')
            if control=='frozen_backbone' and summary['parameter_movement']['final_optimizer']['lora']['changed_tensors']!=0:raise ValueError('Frozen backbone changed')
            technical.append({'label':label,'control':control,'steps':steps,'selected_step':summary['selected_optimizer_step'],'tune_logloss':summary['selected_tune_direction_log_loss'],'initial_direction_logloss':summary['tune_trace'][0]['metrics']['direction']['log_loss'],'initial_hash':summary['initial_trainable_state_sha256'],'sample_order_sha256':summary['sample_order_sha256'],'wall_seconds':summary['wall_seconds'],'peak_rss_bytes':summary['peak_rss_bytes'],'tune_trace':[{k:v for k,v in row.items() if k=='optimizer_step'}|{'direction_logloss':row['metrics']['direction']['log_loss'],'teachers':row['metrics']['teachers']} for row in summary['tune_trace']],'gradient_diagnostics':summary['gradient_diagnostics'],'parameter_movement':summary['parameter_movement'],'training_summary_sha256':sha256_file(run/'training_summary.json')})
        reference=technical[0]
        for row in technical:
            if row['sample_order_sha256']!=reference['sample_order_sha256']:raise ValueError('80-stock sample order mismatch')
            if row['control']!='frozen_backbone' and row['initial_hash']!=reference['initial_hash']:raise ValueError('80-stock initialization mismatch')
        mechanism_report={'schema':'mechanism80_train_tune_controls_v1','status':'completed','dataset_manifest_sha256':sha256_file(mechanism/'manifest.json'),'rows':technical,'all_fixed_budgets_completed':True,'no_holdout_used_for_decisions':True,'technical_gates_passed':True,'main_hyperparameters_unchanged':True,'interpretation':'Controls use train and tune only; shuffled train diagnostics refer to permuted targets, tune remains real; efficacy may be negative and is not a technical failure.'}
        write_json(c,output/'mechanism80_report.json',mechanism_report)
        main_steps=math.ceil(2*main_manifest['splits']['train']['row_count']/4);main_interval=math.ceil(main_steps/4)
        status['formal_budget']={'max_steps':main_steps,'evaluation_interval':main_interval,'batch_size':4,'evaluation_batch_size':32,'seeds':[42,123,2026],'epochs_requested':2};save()
        print(f'study starting formal 240-stock ablation steps={main_steps}',file=sys.stderr,flush=True)
        experiment=output/'ablation240'
        run_ablation(c,dataset=dataset,output_dir=experiment,device='mps',seeds=(42,123,2026),max_steps=main_steps,batch_size=4,evaluation_interval=main_interval,evaluation_batch_size=32)
        child('analyze240',[c.root/'run.py','--config',c.config_path,'analyze-ablation','--dataset',dataset,'--experiment-dir',experiment,'--output-dir','artifacts/reports/ablation240_20261021'])
        child('adapter_diagnostics',[c.root/'scripts/diagnose_ablation_adapters.py','--config',c.config_path,'--experiment-dir',experiment,'--output','artifacts/reports/ablation240_20261021/adapter_diagnostics.json'])
        status.update(status='completed',completed_at=datetime.now(timezone.utc).isoformat(),wall_seconds=time.perf_counter()-start);save()
    except BaseException as exc:
        status.update(status='failed',error=str(exc),completed_at=datetime.now(timezone.utc).isoformat());save();raise
    print(json.dumps(status,indent=2))
if __name__=='__main__':main()
