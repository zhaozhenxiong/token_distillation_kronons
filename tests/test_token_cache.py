from dataclasses import replace
import json
import numpy as np
import pytest
from token_distillation.config import load_config
from token_distillation.artifacts import sha256_file
from token_distillation.training import token_cache as tc

def test_cache_identity_subset_and_tamper(tmp_path,monkeypatch):
    c=load_config();c=replace(c,root=tmp_path,state_root=tmp_path/'artifacts/state');c.values['training']['token_cache']='data/cache';directory=tmp_path/'data/cache';directory.mkdir(parents=True)
    n=c.context_length+c.horizon
    arrays={'s1':np.zeros((2,n),dtype=np.int64),'s2':np.ones((2,n),dtype=np.int64),'sample_hashes':np.asarray(['a'*64,'b'*64])}
    for name,a in arrays.items():np.save(directory/(name+'.npy'),a,allow_pickle=False)
    monkeypatch.setattr(tc,'cache_key',lambda config:{'pinned':'synthetic'})
    m={'schema':'frozen_token_cache_v1','bindings':{'pinned':'synthetic'},'prefix_proof':{'prefix_token_ids_exact':True},'rows':2,'files':{name:{'path':name+'.npy','sha256':sha256_file(directory/(name+'.npy'))} for name in arrays}}
    (directory/'manifest.json').write_text(json.dumps(m))
    rows,proof=tc.cached_tokens(c,[{'sample_sha256':'b'*64}]);assert rows[0]['_s1'].shape==(n,) and proof['checked_samples']==1
    with pytest.raises(ValueError,match='absent'):tc.cached_tokens(c,[{'sample_sha256':'c'*64}])
    with (directory/'s1.npy').open('ab') as f:f.write(b'tamper')
    with pytest.raises(ValueError,match='identity'):tc.cached_tokens(c,[{'sample_sha256':'a'*64}])

from test_dataset import fixture
from token_distillation.data import dataset as ds
from token_distillation.data.compact import compact_sample

def test_validated_cache_reuses_exact_population_and_rejects_changed_source(fixture,monkeypatch):
    f=fixture;ds.build_dataset(f.config,**f.params,output_dir='data/raw',max_per_split=2)
    manifest,raw=ds.load_dataset(f.config,'data/raw');rows=[{k:v for k,v in compact_sample(r).items() if k not in ('normalized_context','normalized_future')} for r in raw]
    c=f.config;c.values['training']['token_cache']='data/cache';directory=c.root/'data/cache';directory.mkdir()
    (directory/'rows.json').write_text(json.dumps(rows));m={'schema':'frozen_token_cache_v1','bindings':{'test':1},'rows':len(rows),'deep_validated_rows':len(rows),'compact_rows_file':{'path':'rows.json','sha256':sha256_file(directory/'rows.json')}}
    (directory/'manifest.json').write_text(json.dumps(m));monkeypatch.setattr(tc,'cache_key',lambda c:{'test':1})
    actual,loaded=tc.load_model_dataset(c,'data/raw');assert loaded==rows
    with (c.root/'data/raw/train.jsonl').open('a') as h:h.write('changed')
    with pytest.raises(ValueError,match='shard changed'):tc.load_model_dataset(c,'data/raw')
