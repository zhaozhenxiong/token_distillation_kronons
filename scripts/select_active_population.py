#!/usr/bin/env python3
"""Explicit pre-2024 active-stock selection; no STAR, Beijing or future outcomes."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from profile_local_universe import board, summarize
from token_distillation.adapters.khquant import KhquantSource
from token_distillation.config import load_config
from token_distillation.teachers.ict_teacher import VERIFIED_SOURCES

BOARDS = ('sh_main', 'sz_main', 'chinext')
POLICY = 'pre2024_active_nonstar_nonbj_v1'

def select_active(profiles, total=800):
    if type(total) is not int or total < 1:
        raise ValueError('Positive integer sample size required')
    core = [dict(r) for r in profiles if r['core_eligible'] and r['board'] in BOARDS]
    ranked = []
    metrics = ('median_amount_cny', 'volatility', 'median_amplitude', 'volume_log_change')
    for b in BOARDS:
        pool = [r for r in core if r['board'] == b]
        if not pool:
            continue
        frame = pd.DataFrame(pool)
        for metric in metrics:
            values = frame[metric].to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError('Finite nonnegative activity metrics required')
            frame[metric + '_percentile'] = frame[metric].rank(method='average', pct=True)
        for r in frame.to_dict('records'):
            r['activity_eligible'] = r['median_amount_cny'] >= 20_000_000 and r['volatility_percentile'] >= .4
            r['activity_score'] = sum(r[k + '_percentile'] * w for k, w in zip(metrics, (.35, .35, .15, .15)))
            r['activity_tiebreak'] = hashlib.sha256((POLICY + '|' + r['symbol']).encode()).hexdigest()
            ranked.append(r)
    eligible = [r for r in ranked if r['activity_eligible']]
    counts = Counter(r['board'] for r in eligible)
    if len(eligible) < total:
        raise ValueError(f'Only {len(eligible)} activity-qualified symbols for requested {total}')
    quotas = {b: int(total * counts[b] / len(eligible)) for b in BOARDS}
    while sum(quotas.values()) < total:
        b = max((b for b in BOARDS if quotas[b] < counts[b]),
                key=lambda b: (total * counts[b] / len(eligible) - quotas[b], -BOARDS.index(b)))
        quotas[b] += 1
    chosen = []
    for b in BOARDS:
        pool = sorted((r for r in eligible if r['board'] == b),
                      key=lambda r: (-r['activity_score'], r['activity_tiebreak']))
        chosen.extend(pool[:quotas[b]])
    return {'main': sorted(chosen, key=lambda r: r['symbol']), 'board_quotas': quotas,
            'activity_qualified': len(eligible), 'ranked_profiles': sorted(ranked, key=lambda r: r['symbol']),
            'mechanism80': [], 'stress': [],
            'ranking_policy': {'within_board_percentiles': dict(zip(metrics, (.35, .35, .15, .15))),
                               'minimum_median_amount_cny': 20_000_000, 'minimum_within_board_volatility_percentile': .4,
                               'board_allocation': 'proportional_to_activity_qualified_largest_remainder',
                               'selection': 'highest_activity_score_within_each_board'}}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--count', type=int, default=800)
    a = p.parse_args()
    c = load_config(a.config)
    if c.values['splits']['train'][0] != '2024-01-01':
        raise ValueError('This frozen selector requires training beginning 2024-01-01')
    out = c.output_path(a.output_dir)
    if out.exists():
        raise FileExistsError(out)
    start = time.perf_counter()
    source = KhquantSource(c.source_app, c.raw_db, c.state_root, calendar_path=c.calendar_path)
    calendar = source.calendar_dates('2023-01-01', '2023-12-29')
    before = c.raw_db.stat()
    profiles = []
    digest = hashlib.sha256()
    with source.connection() as db:
        db.execute('PRAGMA temp_store=MEMORY')
        cursor = db.execute('SELECT stock,date,open,high,low,close,volume,amount,source,has_trade_price FROM stock_daily_normalized WHERE date>=? AND date<=? ORDER BY stock,date', ('2023-01-01', '2023-12-29'))
        for symbol, group in itertools.groupby(cursor, key=lambda r: r['stock']):
            if board(symbol) not in BOARDS:
                continue
            rows = [dict(r) for r in group]
            for r in rows:
                digest.update((json.dumps(r, sort_keys=True, separators=(',', ':')) + '\n').encode())
            profile = summarize(symbol, rows, calendar)
            volumes = {}
            amplitudes = []
            for r in rows:
                try:
                    o,h,l,close,v,amount = [float(r[k]) for k in ('open','high','low','close','volume','amount')]
                    good = (r['source'] in VERIFIED_SOURCES and r['has_trade_price'] == 1
                            and r['date'] in calendar and np.isfinite([o,h,l,close,v,amount]).all()
                            and min(o,h,l,close,v,amount) > 0 and h >= max(o,close,l) and l <= min(o,close,h))
                except (TypeError, ValueError):
                    good = False
                if good:
                    volumes[r['date']] = v
                    amplitudes.append((h-l)/close)
            changes = [abs(np.log(volumes[b]/volumes[a])) for a,b in zip(calendar,calendar[1:]) if a in volumes and b in volumes]
            profile.update(median_amplitude=float(np.median(amplitudes)) if amplitudes else 0.,
                           volume_log_change=float(np.median(changes)) if changes else 0.)
            profiles.append(profile)
    selection = select_active(profiles, a.count)
    after = c.raw_db.stat()
    if (before.st_size,before.st_mtime_ns) != (after.st_size,after.st_mtime_ns):
        raise ValueError('Source database changed during selection; freeze a fresh selection explicitly')
    report = {'schema':'active_pre2024_population_v1', 'created_at':datetime.now(timezone.utc).isoformat(),
              'policy':POLICY, 'cutoff':'2023-12-29', 'universe':'sh_sz_a_v1',
              'excluded_boards':['star','beijing'], 'selection_uses_post2023_data':False,
              'input_2023_rows_sha256':digest.hexdigest(), 'observed_2023_symbols':len(profiles),
              'core_qualified':sum(r['core_eligible'] for r in profiles), 'profiles':profiles,
              'source_identity':source.source_identity(), 'calendar_identity':source.calendar_identity(),
              'raw_db_size_mtime_unchanged':True, 'training_started':False,
              'availability_basis':'assumed_session_close; historical ingestion and revisions unverified',
              'limitations':['Retrospective stored universe, not survivor-free.',
                             'Activity-selected cohort; results do not represent the full market.',
                             'Unadjusted prices may contain corporate-action jumps; existing overnight-gap exclusions apply to return volatility.'],
              'elapsed_seconds':time.perf_counter()-start, **selection}
    out.mkdir(parents=True)
    (out/'population.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (out/'main_symbols.txt').write_text('\n'.join(r['symbol'] for r in report['main'])+'\n')
    print(json.dumps({'selected':len(report['main']), 'board_quotas':report['board_quotas'], 'activity_qualified':report['activity_qualified']}))

if __name__ == '__main__':
    main()
