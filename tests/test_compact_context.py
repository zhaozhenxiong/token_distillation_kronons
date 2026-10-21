from copy import deepcopy
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from token_distillation.data.compact import context_features
from token_distillation.models.student import KronosStudent
from token_distillation.training.runner import shuffle_teacher_targets
from test_runtime import SyntheticBackbone,samples
from test_dataset import fixture
from token_distillation.data import dataset

def test_context_feature_inputs_are_causal_and_restore_missing_coordinate():
    spec=importlib.util.spec_from_file_location('example',Path(__file__).parents[1]/'scripts/check_normalization_identifiability.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    history,future,shifted,_,_,left,right=m.counterexample()
    def row(raw,stats):return {'raw_context':raw.tolist(),'normalization':{'mean':stats.mean.tolist(),'std':stats.std.tolist(),'epsilon':stats.epsilon},'raw_future':future.tolist()}
    a=row(history,left.stats);b=row(shifted,right.stats)
    assert len(context_features(a))==20
    assert not np.array_equal(context_features(a),context_features(b))
    changed=deepcopy(a);changed['raw_future']=(future*100).tolist()
    assert context_features(changed)==context_features(a)

def test_stream_compaction_keeps_verified_training_values(fixture):
    f=fixture;dataset.build_dataset(f.config,**f.params,output_dir='data/one',max_per_split=3)
    _,full=dataset.load_dataset(f.config,'data/one');_,small=dataset.load_dataset(f.config,'data/one',compact=True)
    assert [r['sample_sha256'] for r in full]==[r['sample_sha256'] for r in small]
    assert 'provenance' not in small[0] and 'raw_context' not in small[0]
    for a,b in zip(full,small):
        np.testing.assert_array_equal(np.asarray(a['normalized_context'],dtype=np.float32),b['normalized_context'])
        for name in a['teachers']:
            assert a['teachers'][name]['continuous_targets']==b['teachers'][name]['continuous_targets']

def test_head_features_required_and_do_not_modify_ar():
    student=KronosStudent(SyntheticBackbone(),3,lora_rank=2,context_feature_dim=20)
    x=torch.zeros((2,4),dtype=torch.long)
    with pytest.raises(ValueError,match='context features'):student(x,x)
    a=student(x,x,context_features=torch.zeros(2,20));b=student(x,x,context_features=torch.ones(2,20))
    assert torch.equal(a['s1_logits'],b['s1_logits'])
    assert not torch.equal(a['direction_logits'],b['direction_logits'])

def test_shuffle_preserves_masks_distribution_and_forbids_nontrain():
    rows=[deepcopy(samples()[0]) for _ in range(20)]
    for i,row in enumerate(rows):
        row['split']='train'
        for t in row['teachers'].values():t['continuous_targets'][0]=float(i)
    shuffled=shuffle_teacher_targets(rows,42)
    for name in rows[0]['teachers']:
        assert sorted(r['teachers'][name]['continuous_targets'][0] for r in shuffled)==list(map(float,range(20)))
        assert [r['teachers'][name]['continuous_mask'] for r in rows]==[r['teachers'][name]['continuous_mask'] for r in shuffled]
    assert rows!=shuffled
    rows[0]['split']='holdout'
    with pytest.raises(ValueError,match='train-only'):shuffle_teacher_targets(rows,42)
