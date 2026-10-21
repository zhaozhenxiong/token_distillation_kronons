import importlib.util
from pathlib import Path
import numpy as np
from token_distillation.evaluation.ablation import date_block_interval
s=importlib.util.spec_from_file_location('stratified_effect',Path(__file__).parents[1]/'scripts/analyze_stratified_study.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_paired_weighted_date_blocks_reduce_to_existing_equal_weight_method():
    rng=np.random.default_rng(42);values=rng.normal(-.01,.03,(3,80));dates=np.repeat([f'2026-05-{i:02d}' for i in range(1,21)],4)
    actual=m.effect(values,np.ones(80),dates);expected=date_block_interval(values,dates)
    np.testing.assert_allclose(actual['ci95'],expected['ci95'],rtol=0,atol=1e-15)
    assert abs(actual['mean']-values.mean())<1e-15
    w=np.arange(1,81);weighted=m.effect(values,w,dates)
    assert abs(weighted['mean']-np.average(values,axis=1,weights=w).mean())<1e-15
