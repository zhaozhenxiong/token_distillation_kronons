#!/usr/bin/env python3
"""Explicit read-only 2023 census and deterministic stratified research selection."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from token_distillation.config import load_config
from token_distillation.adapters.khquant import KhquantSource
from token_distillation.teachers.ict_teacher import VERIFIED_SOURCES
POLICY='pre2024_strata240_v1'
BOARDS=('sh_main','sz_main','chinext','star')
def board(symbol):
    for prefix,suffix,name in [('60','.SH','sh_main'),('68','.SH','star'),('00','.SZ','sz_main'),('30','.SZ','chinext')]:
        if len(symbol)==9 and symbol.startswith(prefix) and symbol.endswith(suffix) and symbol[:6].isdigit():return name
    return None

def rank(symbol):return hashlib.sha256(f'{POLICY}|2023-12-29|42|{symbol}'.encode()).hexdigest()

def summarize(symbol,rows,calendar):
    required=set(calendar[-128:]); observed=set(); amounts=[]; verified_dates=[]; prices={}; gaps=[]; invalid=0; zero=0
    for r in rows:
        day=r['date']; values=[r[k] for k in ('open','high','low','close','volume','amount')]
        try:
            o,h,l,c,v,a=map(float,values)
            legal=np.isfinite([o,h,l,c,v,a]).all() and min(o,h,l,c)>0 and v>=0 and a>=0 and h>=max(o,c,l) and l<=min(o,c,h)
        except (ValueError,TypeError):legal=False
        good=legal and r['source'] in VERIFIED_SOURCES and r['has_trade_price']==1 and day in calendar
        if not good:invalid+=1;continue
        if v<=0 or a<=0:zero+=1;continue
        observed.add(day);amounts.append(a);verified_dates.append(day);prices[day]=(o,c)
    returns=[];raw_returns=[]
    for left,right in zip(calendar,calendar[1:]):
        if left not in prices or right not in prices:continue
        value=float(np.log(prices[right][1]/prices[left][1]));raw_returns.append(value)
        if abs(prices[right][0]/prices[left][1]-1)>(.25 if board(symbol) in ('chinext','star') else .15):gaps.append(right)
        else:returns.append(value)
    median=float(np.median(amounts)) if amounts else None
    reasons=[]
    if len(observed)<200:reasons.append('fewer_than_200_verified_2023_bars')
    if required-observed:reasons.append('incomplete_last_128_sessions')
    if median is None or median<10_000_000:reasons.append('median_amount_below_10m')
    if len(returns)<100:reasons.append('insufficient_adjacent_return_pairs')
    return {'symbol':symbol,'board':board(symbol),'verified_2023_rows':len(observed),'first_verified_observed_2023_date':min(verified_dates) if verified_dates else None,'last128_missing':len(required-observed),'invalid_or_unverified_rows':invalid,'zero_turnover_rows':zero,'median_amount_cny':median,'volatility':float(np.std(returns,ddof=1)) if len(returns)>1 else None,'raw_volatility':float(np.std(raw_returns,ddof=1)) if len(raw_returns)>1 else None,'adjacent_return_pairs':len(returns),'suspected_gap_dates':gaps,'core_eligible':not reasons,'reasons':reasons,'selection_hash':rank(symbol)}

def allocate_board(counts,total):
    if sum(counts.values())<total:raise ValueError('Insufficient qualified core population')
    q={b:min(20,counts.get(b,0)) for b in BOARDS}
    while sum(q.values())<total:
        available=[b for b in BOARDS if q[b]<counts.get(b,0)]
        b=max(available,key=lambda b:(counts[b]/(q[b]+1),-BOARDS.index(b)))
        q[b]+=1
    return q

def balanced(groups,n):
    keys=sorted(groups); cursors={k:0 for k in keys}; chosen=[]
    while len(chosen)<n:
        available=[k for k in keys if cursors[k]<len(groups[k])]
        if not available:raise ValueError('Sample quota exceeds population')
        k=min(available,key=lambda k:(cursors[k],k));chosen.append(groups[k][cursors[k]]);cursors[k]+=1
    return chosen

def select(profiles,total=240):
    core=[dict(r) for r in profiles if r['core_eligible']]
    quotas=allocate_board(Counter(r['board'] for r in core),total); layers={}; chosen=[]; cuts={}
    for b in BOARDS:
        pool=[r for r in core if r['board']==b]
        if not pool:continue
        lc=np.quantile([r['median_amount_cny'] for r in pool],[1/3,2/3]);vc=np.quantile([r['volatility'] for r in pool],[1/3,2/3]);cuts[b]={'amount':lc.tolist(),'volatility':vc.tolist()}
        groups=defaultdict(list)
        for r in pool:
            r['stratum']=f'{b}|L{np.searchsorted(lc,r["median_amount_cny"],side="right")}|V{np.searchsorted(vc,r["volatility"],side="right")}'
            groups[r['stratum']].append(r)
        for group in groups.values():group.sort(key=lambda r:r['selection_hash'])
        picked=balanced(groups,quotas[b]);chosen.extend(picked)
        for key,group in sorted(groups.items()):
            n=sum(r['stratum']==key for r in picked);layers[key]={'eligible':len(group),'selected':n,'inclusion_probability':n/len(group)}
    groups=defaultdict(list)
    for r in chosen:groups[r['stratum']].append(r)
    for group in groups.values():group.sort(key=lambda r:r['selection_hash'])
    subset=balanced(groups,min(80,total)); used={r['symbol'] for r in chosen};stress=[]
    predicates=[('low_liquidity',lambda r:r['median_amount_cny'] is not None and 0<r['median_amount_cny']<10_000_000),('gaps_or_zero',lambda r:r['last128_missing']>0 or r['zero_turnover_rows']>0),('short_history',lambda r:r['verified_2023_rows']<200),('suspected_gap',lambda r:bool(r['suspected_gap_dates']))]
    for category,predicate in predicates:
        pool=sorted((r for r in profiles if r['symbol'] not in used and predicate(r)),key=lambda r:r['selection_hash'])
        for r in pool[:6]:stress.append({**r,'stress_category':category});used.add(r['symbol'])
    return {'board_quotas':quotas,'within_board_tertile_cutoffs':cuts,'strata':layers,'main':sorted(chosen,key=lambda r:r['symbol']),'mechanism80':sorted(r['symbol'] for r in subset),'stress':stress}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--config',default='configs/ablation.toml');ap.add_argument('--output-dir',required=True);args=ap.parse_args()
    c=load_config(args.config);out=c.output_path(args.output_dir)
    if out.exists():raise FileExistsError(out)
    start=time.perf_counter();source=KhquantSource(c.source_app,c.raw_db,c.state_root,calendar_path=c.calendar_path);calendar=source.calendar_dates('2023-01-01','2023-12-29')
    before=c.raw_db.stat(); profiles=[];digest=hashlib.sha256()
    with source.connection() as db:
        db.execute('PRAGMA temp_store=MEMORY')
        cur=db.execute('SELECT stock,date,open,high,low,close,volume,amount,source,has_trade_price FROM stock_daily_normalized WHERE date>=? AND date<=? ORDER BY stock,date',('2023-01-01','2023-12-29'))
        for symbol,group in itertools.groupby(cur,key=lambda r:r['stock']):
            if not board(symbol):continue
            rows=[dict(r) for r in group]
            for r in rows:digest.update((json.dumps(r,sort_keys=True,separators=(',',':'))+'\n').encode())
            profiles.append(summarize(symbol,rows,calendar))
    selection=select(profiles)
    report={'schema':'stratified_pre2024_population_v1','created_at':datetime.now(timezone.utc).isoformat(),'cutoff':'2023-12-29','policy':POLICY,'universe':'sh_sz_a_v1','input_2023_rows_sha256':digest.hexdigest(),'selection_uses_post2023_data':False,'availability_basis':'assumed_session_close; historical ingestion and revisions unverified','source_identity':source.source_identity(),'calendar_identity':source.calendar_identity(),'observed_2023_symbols':len(profiles),'core_qualified':sum(r['core_eligible'] for r in profiles),'profiles':profiles,**selection,'elapsed_seconds':time.perf_counter()-start,'raw_db_size_mtime_unchanged':(before.st_size,before.st_mtime_ns)==(c.raw_db.stat().st_size,c.raw_db.stat().st_mtime_ns),'training_started':False,'limitations':['Stored retrospective universe, not survivor-free.','Within-board tertiles and rare-stratum balancing are not population-proportional; inclusion probabilities retained.','Stress cohort separate from performance comparison.']}
    out.mkdir(parents=True)
    (out/'population.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (out/'main_symbols.txt').write_text('\n'.join(r['symbol'] for r in report['main'])+'\n')
    (out/'mechanism80_symbols.txt').write_text('\n'.join(report['mechanism80'])+'\n')
    print(json.dumps({k:report[k] for k in ('observed_2023_symbols','core_qualified','board_quotas','elapsed_seconds')},ensure_ascii=False))
if __name__=='__main__':main()
