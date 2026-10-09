"""Repeat the UN Comtrade catalog capture every month with the same selection (stage 1C).

The prospective track needs the same nine reporters and the same fixed period list
captured again each calendar month so dataset checksums and row counts can be
compared vintage to vintage. This wrapper keeps the selection in
data/comtrade-vintages/schedule.json, refuses to run twice in one UTC month unless
forced, and delegates the actual capture to scripts/capture_comtrade_vintage.py.

Usage:
  python scripts/monthly_vintage_capture.py            # capture if this month has none
  python scripts/monthly_vintage_capture.py --dry-run  # show the planned selection only
  python scripts/monthly_vintage_capture.py --force    # capture again this month

A premium key, if present in COMTRADE_API_KEY, is only read by the capture script;
it is never written to the schedule or manifests.
"""
import argparse
import datetime as dt
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT));import btl_paths
from adapters.contracts import ROUTE_SUPPLIERS
from scripts import capture_comtrade_vintage as capture

SCHEDULE = ROOT / 'data/comtrade-vintages/schedule.json'
FIRST_BASIS = '2026-09'


def default_schedule():
    return dict(
        schema_version='btl-v3-vintage-schedule-v1',
        cadence='monthly',
        rule='Same reporters and same fixed periods every month; a new UTC snapshot id per run; never backdate',
        reporters=list(ROUTE_SUPPLIERS),  # same order as the first capture's catalog URL
        periods=list(capture.months('202510', '202609')),
        first_basis_month=FIRST_BASIS,
        outcome_end='2026-12',
        maturity_buffer_end='2027-05',
        bulk=True,
        bulk_limit=1,
        runs=[])


def load_schedule():
    if SCHEDULE.exists():
        return json.loads(SCHEDULE.read_text(encoding='utf-8'))
    return default_schedule()


def save_schedule(schedule):
    SCHEDULE.parent.mkdir(parents=True, exist_ok=True)
    SCHEDULE.write_text(json.dumps(schedule, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def month_key(when):
    return when.strftime('%Y-%m')


def run_this_month(schedule, when):
    key = month_key(when)
    return next((r for r in schedule['runs'] if r['month'] == key), None)


def plan(schedule, when):
    return dict(month=month_key(when), reporters=schedule['reporters'], periods=schedule['periods'],
                bulk=schedule['bulk'], bulk_limit=schedule['bulk_limit'])


def main(argv=None, now=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)
    when = now or dt.datetime.now(dt.timezone.utc)
    schedule = load_schedule()
    existing = run_this_month(schedule, when)
    planned = plan(schedule, when)
    if args.dry_run:
        print(json.dumps(dict(action='dry_run', existing_run=existing, plan=planned), ensure_ascii=False))
        return 0
    if existing and not args.force:
        print(json.dumps(dict(action='skipped_already_captured', run=existing), ensure_ascii=False))
        return 0
    capture_args = []
    for reporter in planned['reporters']:
        capture_args += ['--reporter', reporter]
    for period in planned['periods']:
        capture_args += ['--period', period]
    if planned['bulk']:
        capture_args += ['--bulk', '--limit', str(planned['bulk_limit'])]
    before = {p.name for p in (capture.BASE / 'manifests').glob('*')} if (capture.BASE / 'manifests').exists() else set()
    sys.argv = ['capture_comtrade_vintage.py'] + capture_args
    capture.main()
    after = {p.name for p in (capture.BASE / 'manifests').glob('*')}
    run_id = max(after - before) if after - before else None
    summary = {}
    if run_id:
        summary = json.loads((capture.BASE / 'manifests' / run_id / 'capture-summary.json').read_text(encoding='utf-8'))
    record = dict(month=month_key(when), run_id=run_id, captured_at_utc=summary.get('captured_at_utc'),
                  catalog_status=summary.get('catalog_status'), catalog_rows=summary.get('catalog_rows'),
                  catalog_change=summary.get('catalog_change'), bulk_attempted=summary.get('bulk_attempted'),
                  subscription_key_configured=summary.get('subscription_key_configured'))
    schedule['runs'] = [r for r in schedule['runs'] if r['month'] != record['month']] + [record]
    save_schedule(schedule)
    print(json.dumps(dict(action='captured', run=record), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
