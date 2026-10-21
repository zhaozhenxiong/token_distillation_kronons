import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('population240',Path(__file__).parents[1]/'scripts/profile_local_universe.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
def test_selection_is_independent_of_future_and_repeatable():
    rows=[]
    for b,prefix,suffix in [('sh_main','600','.SH'),('sz_main','000','.SZ'),('chinext','300','.SZ'),('star','688','.SH')]:
        for i in range(100):
            symbol=f'{prefix}{i:03d}{suffix}'
            rows.append(dict(symbol=symbol,board=b,core_eligible=True,median_amount_cny=1e7+i*1e6,volatility=(i%9+.1)/100,selection_hash=p.rank(symbol),last128_missing=0,zero_turnover_rows=0,verified_2023_rows=240,suspected_gap_dates=[]))
    a=p.select(rows); altered=[dict(r,future_return=100,post2023_missing=True) for r in reversed(rows)];c=p.select(altered)
    assert [r['symbol'] for r in a['main']]==[r['symbol'] for r in c['main']]
    assert len(a['main'])==240 and len(a['mechanism80'])==80
    assert set(a['mechanism80'])<=set(r['symbol'] for r in a['main'])
    assert len(set(r['symbol'] for r in a['main']))==240
    assert sum(r['selected'] for r in a['strata'].values())==240
    assert all(0<r['inclusion_probability']<=1 for r in a['strata'].values())
def test_symbols_only_sh_sz_a():
    assert p.board('600001.SH')=='sh_main'
    assert p.board('430001.BJ') is None
    assert p.board('510001.SH') is None
