import importlib.util
from pathlib import Path
import pytest
from test_dataset import fixture
from token_distillation.data import dataset
spec=importlib.util.spec_from_file_location('chunk_merge',Path(__file__).parents[1]/'scripts/build_stratified_dataset.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_chunk_merge_and_subset_are_byte_equivalent(fixture):
    f=fixture;parts=[]
    for i,symbol in enumerate(f.params['symbols']):
        out=f.config.root/f'data/part{i}';dataset.build_dataset(f.config,**{**f.params,'symbols':[symbol]},output_dir=out);parts.append(out)
    output=f.config.root/'data/merged';m.merge(f.config,parts,output)
    _,merged=dataset.load_dataset(f.config,output)
    dataset.build_dataset(f.config,**f.params,output_dir='data/direct');_,direct=dataset.load_dataset(f.config,'data/direct')
    assert [r['sample_sha256'] for r in merged]==[r['sample_sha256'] for r in direct]
    subset=f.config.root/'data/subset';m.merge(f.config,[output],subset,['600000.SH']);_,rows=dataset.load_dataset(f.config,subset)
    assert [r['sample_sha256'] for r in rows]==[r['sample_sha256'] for r in direct if r['symbol']=='600000.SH']
    with (parts[0]/'train.jsonl').open('a') as h:h.write('tamper')
    with pytest.raises(ValueError,match='changed'):m.merge(f.config,parts,f.config.root/'data/tamper')
