#!/usr/bin/env python3
"""Explicit bounded-memory teacher generation, immutable chunk merge/subsets."""
import argparse
from collections import Counter
from copy import deepcopy
import heapq
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from token_distillation.config import load_config,SPLIT_NAMES
from token_distillation.artifacts import sha256_file,write_json
from token_distillation.data.dataset import build_dataset,_hash
from token_distillation.data.filtering import derive_dataset

def merge(config,inputs,output,symbols=None):
    manifests=[json.loads((d/'manifest.json').read_text()) for d in inputs]
    first=deepcopy(manifests[0]);keep=set(symbols) if symbols is not None else set(s for m in manifests for s in m['symbols'])
    for m in manifests:
        for key in ('source_identity','calendar_identity','config','teacher_schemas'):
            if m[key]!=first[key]:raise ValueError('Chunk source/config identity mismatch: '+key)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);pop=[];coverage={};shards={}
    for split in SPLIT_NAMES:
        def stream(d,m):
            p=d/m['splits'][split]['path']
            if sha256_file(p)!=m['splits'][split]['sha256']:raise ValueError('Chunk changed')
            with p.open() as f:
                for line in f:
                    r=json.loads(line)
                    if r['symbol'] in keep:yield (r['as_of'],r['symbol'],line,r)
        streams=[stream(d,m) for d,m in zip(inputs,manifests)];count=0;start=end=None;cov={}
        with (output/(split+'.jsonl')).open('x') as f:
            for _,_,line,r in heapq.merge(*streams,key=lambda item:item[:2]):
                f.write(line);count+=1;start=start or r['as_of'];end=r['as_of'];pop.append(r['sample_sha256'])
                for name,t in r['teachers'].items():
                    c=cov.setdefault(name,{})
                    for kind in ('continuous','event','structure'):
                        masks=t.get(kind+'_mask',[])
                        for label,value in ((kind+'_known',sum(masks)),(kind+'_total',len(masks))):c[label]=c.get(label,0)+value
        shards[split]={'path':split+'.jsonl','sha256':sha256_file(output/(split+'.jsonl')),'row_count':count,'first_as_of':start,'last_as_of':end};coverage[split]=cov
    overall={}
    for cov in coverage.values():
        for name,stats in cov.items():
            for key,value in stats.items():overall.setdefault(name,{})[key]=overall.setdefault(name,{}).get(key,0)+value
    for stats in overall.values():stats['status']='valid_supervision_present' if any(stats.get(k+'_known',0) for k in ('continuous','event','structure')) else 'no_valid_supervision'
    first['symbols']=sorted(keep);first['splits']=shards;first['total_samples']=len(pop);first['population_sha256']=_hash(pop);first['teacher_coverage']={'by_split':coverage,'overall':overall}
    first['symbol_snapshots']={s:v for m in manifests for s,v in m['symbol_snapshots'].items() if s in keep}
    first['chunk_derivation']={'inputs':[{'manifest':str(d/'manifest.json'),'sha256':sha256_file(d/'manifest.json')} for d in inputs],'bounded_generation_symbols_per_chunk':8,'sample_bytes_preserved':True,'symbol_subset':symbols is not None}
    first['generator_sha256']=sha256_file(Path(__file__))
    first['admission']={'candidate_counts':{k:sum(m['admission']['candidate_counts'][k] for m in manifests) for k in SPLIT_NAMES},'attempted_counts':{k:sum(m['admission']['attempted_counts'][k] for m in manifests) for k in SPLIT_NAMES},'skipped_counts':dict(sum((Counter(m['admission']['skipped_counts']) for m in manifests),Counter())),'post_generation_symbol_subset':symbols is not None}
    if all('derivation' in m for m in manifests):
        first['derivation']={'schema':'chunk_merged_expost_label_gap_derivation_v1','policy':first['derivation']['policy'],'rejected_counts_by_split':{k:sum(m['derivation']['rejected_counts_by_split'][k] for m in manifests) for k in SPLIT_NAMES},'expost_quality_selection_bias':'Conditional on ex-post label-gap admission; unavailable at prediction origin.','admitted_sample_bytes_unchanged':True,'input_manifests_preserved':True}
    first.pop('manifest_sha256',None);first['manifest_sha256']=_hash(first);write_json(config,output/'manifest.json',first)
    return first

def build_chunk(config_path, parts_path, index, symbols):
    c=load_config(config_path);parts=Path(parts_path)
    raw=parts/f'chunk{index:02d}_raw';filtered=parts/f'chunk{index:02d}_filtered'
    if (filtered/'manifest.json').exists():
        m=json.loads((filtered/'manifest.json').read_text())
        if m['symbols']!=sorted(symbols):raise ValueError('Existing chunk population changed')
        for sh in m['splits'].values():
            if sha256_file(filtered/sh['path'])!=sh['sha256']:raise ValueError('Existing chunk bytes changed')
        return str(filtered),m['total_samples'],True
    if (raw/'manifest.json').exists():
        m=json.loads((raw/'manifest.json').read_text())
        if m['symbols']!=sorted(symbols):raise ValueError('Existing raw population changed')
    else:
        if raw.exists():raise ValueError('Incomplete raw chunk requires separate diagnosis')
        m=build_dataset(c,symbols=symbols,start='2024-01-02',end='2026-10-09',label_through='2026-10-16',output_dir=raw,stride=5)
    derive_dataset(c,dataset=raw,output_dir=filtered)
    return str(filtered),m['total_samples'],False

def main():
    import concurrent.futures
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/ablation240.toml');ap.add_argument('--population',default='artifacts/reports/population240_20261021/population.json');ap.add_argument('--output',default='data/ablation240_20261021');ap.add_argument('--workers',type=int,default=1);ap.add_argument('--continue-chunks',action='store_true');args=ap.parse_args();c=load_config(args.config)
    if not 1<=args.workers<=4:raise ValueError('At most four isolated CPU generation workers')
    population=json.loads(c.output_path(args.population).read_text());symbols=[r['symbol'] for r in population['main']];out=c.output_path(args.output);parts=c.output_path(str(args.output)+'_parts')
    if out.exists() or (parts.exists() and not args.continue_chunks):raise FileExistsError('New generation directory required unless explicit continue-chunks')
    parts.mkdir(parents=True,exist_ok=True);chunks=[];started=time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(build_chunk,str(c.config_path),str(parts),i//8,symbols[i:i+8]) for i in range(0,len(symbols),8)]
        for future in concurrent.futures.as_completed(futures):
            path,count,reused=future.result();chunks.append(Path(path))
            print(f'chunks={len(chunks)}/{len(futures)} rows={count} reused_verified_chunk={reused} elapsed={time.perf_counter()-started:.1f}s',file=sys.stderr,flush=True)
    chunks.sort();m=merge(c,chunks,out)
    if population.get('mechanism80'):
        subset=c.output_path(str(args.output)+'_mechanism80');merge(c,[out],subset,population['mechanism80'])
    write_json(c,out/'generation_timing.json',{'elapsed_seconds':time.perf_counter()-started,'rows':m['total_samples'],'chunk_count':len(chunks),'workers':args.workers,'population_sha256':sha256_file(c.output_path(args.population)),'training_started':False})
    print(json.dumps({'dataset':str(out),'rows':m['total_samples'],'splits':{k:v['row_count'] for k,v in m['splits'].items()},'seconds':time.perf_counter()-started}))
if __name__=='__main__':main()
