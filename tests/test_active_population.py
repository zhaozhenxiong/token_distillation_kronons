import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('active_population', Path(__file__).parents[1]/'scripts/select_active_population.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)

def profiles():
    return [dict(symbol=f'{prefix}{i:03d}{suffix}', board=b, core_eligible=True,
                 median_amount_cny=20e6+i*1e6, volatility=.01+i*.0001,
                 median_amplitude=.02+i*.0001, volume_log_change=.1+i*.001)
            for b,prefix,suffix in [('sh_main','600','.SH'),('sz_main','000','.SZ'),
                                    ('chinext','300','.SZ'),('star','688','.SH'),('beijing','430','.BJ')]
            for i in range(100)]

def test_activity_selection_excludes_boards_and_future_and_is_deterministic():
    rows = profiles()
    a = p.select_active(rows, 120)
    b = p.select_active([dict(r, future_return=99, future_missing=True) for r in reversed(rows)], 120)
    assert [r['symbol'] for r in a['main']] == [r['symbol'] for r in b['main']]
    assert len({r['symbol'] for r in a['main']}) == 120
    assert a['board_quotas'] == dict(sh_main=40, sz_main=40, chinext=40)
    assert all(r['board'] in p.BOARDS and r['volatility_percentile'] >= .4 for r in a['main'])
    assert min(int(r['symbol'][3:6]) for r in a['main']) == 60

def test_activity_selection_refuses_insufficient_candidates():
    with pytest.raises(ValueError, match='activity-qualified'):
        p.select_active(profiles(), 800)
