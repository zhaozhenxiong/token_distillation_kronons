#!/usr/bin/env python3
"""Completed-study-only board/stratum and mechanism review; no model/source IO."""
import argparse,html,json,math,statistics,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from token_distillation.artifacts import sha256_file,write_json
from token_distillation.config import load_config
from token_distillation.data.dataset import load_dataset
from token_distillation.evaluation.ablation import teacher_constant_baselines,_teacher_comparison
ARMS=('none','czsc','ict','dual')
def effect(values,weights,dates,block=2):
    values=np.asarray(values);weights=np.asarray(weights);dates=np.asarray(dates);unique=sorted(set(dates));seed_effect=np.average(values,axis=1,weights=weights)
    sums=np.array([np.sum(values[:,dates==d].mean(0)*weights[dates==d]) for d in unique]);denom=np.array([weights[dates==d].sum() for d in unique]);ci=None
    if len(unique)>block:
        rng=np.random.default_rng(1729);starts=rng.integers(0,len(unique)-block+1,size=(2000,math.ceil(len(unique)/block)));indices=(starts[...,None]+np.arange(block)).reshape(2000,-1)[:,:len(unique)]
        estimates=sums[indices].sum(1)/denom[indices].sum(1);ci=np.quantile(estimates,[.025,.975]).tolist()
    return {'mean':float(seed_effect.mean()),'seed_SD':float(seed_effect.std(ddof=1)),'seed_values':seed_effect.tolist(),'ci95':ci,'date_clusters':len(unique),'block_length_origins':block,'nominal_descriptive_only':True,'fixed_seed_set_conditional':True}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/ablation240.toml');ap.add_argument('--study',default='artifacts/runs/study240_20261021');ap.add_argument('--report',default='artifacts/reports/ablation240_20261021');ap.add_argument('--population',default='artifacts/reports/population240_20261021/population.json');args=ap.parse_args();c=load_config(args.config);study=c.output_path(args.study);out=c.output_path(args.report)
    status=json.loads((study/'study_status.json').read_text());experiment=study/'ablation240';manifest=json.loads((experiment/'experiment_manifest.json').read_text());comparison=json.loads((out/'ablation_comparison.json').read_text())
    if manifest['status']!='completed' or status['status']!='completed':raise ValueError('Only complete240 study can be analyzed')
    population=json.loads(c.output_path(args.population).read_text());profiles={r['symbol']:r for r in population['main']};seeds=manifest['protocol']['seeds'];arrays={};reference=None
    for arm in ARMS:
        losses=[]
        for seed in seeds:
            rows=[r for r in json.loads((experiment/f'{arm}_seed{seed}_evaluation/predictions.json').read_text())['rows'] if r['split']=='holdout'];rows.sort(key=lambda r:(r['symbol'],r['as_of']));keys=[(r['symbol'],r['as_of'],r['direction_class']) for r in rows]
            if reference is None:reference=rows;identity=keys
            if keys!=identity:raise ValueError('Paired raw prediction populations differ')
            losses.append([-math.log(max(r['direction_probabilities'][r['direction_class']],1e-12)) for r in rows])
        arrays[arm]=np.asarray(losses)
    dates=np.asarray([r['as_of'][:10] for r in reference]);weights=np.asarray([1/population['strata'][profiles[r['symbol']]['stratum']]['inclusion_probability'] for r in reference])
    def group(mask):
        group_weights=weights[mask];group_dates=dates[mask]
        return {'rows':int(mask.sum()),'symbols':len({reference[i]['symbol'] for i in np.flatnonzero(mask)}),'dates':len(set(group_dates)),'arms':{a:{'equal_row_logloss':float(arrays[a][:,mask].mean()),'inverse_probability_weighted_logloss':float(np.average(arrays[a][:,mask],axis=1,weights=group_weights).mean()),'equal_delta_vs_none':effect(arrays[a][:,mask]-arrays['none'][:,mask],np.ones(int(mask.sum())),group_dates),'weighted_delta_vs_none':effect(arrays[a][:,mask]-arrays['none'][:,mask],group_weights,group_dates),'weighted_block5_sensitivity':effect(arrays[a][:,mask]-arrays['none'][:,mask],group_weights,group_dates,block=5)} for a in ARMS}}
    by_board={b:group(np.asarray([profiles[r['symbol']]['board']==b for r in reference])) for b in population['board_quotas']}
    by_stratum={s:group(np.asarray([profiles[r['symbol']]['stratum']==s for r in reference])) for s in population['strata'] if any(profiles[r['symbol']]['stratum']==s for r in reference)}
    mechanism=json.loads((study/'mechanism80_report.json').read_text());_,small=load_dataset(c,'data/ablation240_20261021_mechanism80');train=[r for r in small if r['split']=='train'];tune=[r for r in small if r['split']=='tune'];controls=[]
    for row in mechanism['rows']:
        summary=json.loads((study/('mechanism80_'+row['label'])/'training_summary.json').read_text());selected=next(t for t in summary['tune_trace'] if t['optimizer_step']==summary['selected_optimizer_step']);constants=teacher_constant_baselines(train,tune,summary['teacher_statistics'])
        controls.append({'label':row['label'],'control':row['control'],'steps':row['steps'],'selected_step':row['selected_step'],'tune_logloss':row['tune_logloss'],'tune_teacher_acquisition':_teacher_comparison(selected['metrics']['teachers'],constants),'gradient_diagnostics':row['gradient_diagnostics'],'parameter_movement':row['parameter_movement']})
    repeat=abs(next(r['tune_logloss'] for r in controls if r['label']=='dual_repeat')-next(r['tune_logloss'] for r in controls if r['label']=='dual'))
    result={'schema':'stratified240_completed_review_v1','comparison_sha256':sha256_file(out/'ablation_comparison.json'),'population_sha256':sha256_file(c.output_path(args.population)),'scope':'fixed retrospective 240 core sample; inverse inclusion weighting describes eligible2023 core conditional on observed admitted windows, not all local stocks','stress_excluded_from_effect_metrics':True,'holdout':group(np.ones(len(reference),dtype=bool)),'by_board':by_board,'by_stratum':by_stratum,'mechanism80_train_tune':controls,'same_objective_repeat_abs_tune_difference':repeat,'repeat_limit':'one repeat pair only; not a statistical numerical-noise distribution','multiple_comparison_adjusted':False,'forecast_or_trading_returns_asserted':False}
    if (out/'stratified_comparison.json').exists():raise FileExistsError('Stratified review already exists')
    write_json(c,out/'stratified_comparison.json',result)
    parts=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>240股分层与机制对照</title><style>body{font:15px system-ui;max-width:1150px;margin:32px auto;padding:20px;color:#172536}table{border-collapse:collapse;width:100%;margin:20px 0}th,td{border:1px solid #d8e0e6;padding:8px;text-align:right}th:first-child,td:first-child{text-align:left}</style><h1>240股分层与机制对照</h1><p>分层来自2023截面；压力样本不混入效果评价。分层差异为回溯描述，未作多重比较校正；逆概率权重仅对应满足主池准入且具有可用窗口的人口。</p>']
    def table(title,groups):
        parts.append('<h2>'+title+'</h2><table><tr><th>分组</th><th>窗口/股票/日期</th><th>无教师 LogLoss</th><th>CZSC Δ</th><th>ICT Δ</th><th>双教师 Δ</th></tr>')
        for name,g in groups.items():
            parts.append('<tr><td>'+html.escape(name)+f'</td><td>{g["rows"]}/{g["symbols"]}/{g["dates"]}</td><td>{g["arms"]["none"]["equal_row_logloss"]:.6f}</td>'+''.join(f'<td>{g["arms"][a]["equal_delta_vs_none"]["mean"]:+.6f}</td>' for a in ARMS[1:])+'</tr>')
        parts.append('</table>')
    table('留出分板块',by_board);table('留出36层',by_stratum)
    parts.append('<h2>80股机制对照（仅train/tune）</h2><table><tr><th>组</th><th>步数/选中</th><th>tune LogLoss</th></tr>')
    for row in controls:parts.append(f'<tr><td>{html.escape(row["label"])}</td><td>{row["steps"]}/{row["selected_step"]}</td><td>{row["tune_logloss"]:.6f}</td></tr>')
    parts.append(f'</table><p>相同目标重复一次的tune差值：{repeat:.8f}。单次重复不足以估计完整数值噪声分布。打乱控制只置换训练段已知教师值并保留mask；tune目标保持真实。冻结主干控制用于观察内部头的重建能力，不单独证明主干语义吸收。</p><p><a href="index.html">四组完整消融报告</a> · <a href="stratified_comparison.json">完整加权指标、区间与机制JSON</a></p></html>')
    (out/'stratified.html').write_text('\n'.join(parts))
    print(json.dumps({'report':str(out/'stratified.html'),'repeat_tune_difference':repeat,'strata':len(by_stratum)},indent=2))
if __name__=='__main__':main()
