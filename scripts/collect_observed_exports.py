"""UN Comtrade collection of candidate -> target export reports, one request per month.

Without a subscription key the public preview endpoint is used (500 rows per call, a small daily request quota). With
COMTRADE_API_KEY in the environment or in the project's .env the data endpoint is used instead (same parameters, a much
larger quota). The key is sent as a request header only: it is never printed and never written to a manifest.

Either endpoint takes one period per call but several reporters and partners, so a single call
returns every candidate's reported exports of the analysed HS6 codes to the four targets for that month
(at most 9 x 4 x 10 rows, under the 500-row preview cap). Raw replies and a manifest per month are kept under
data/observed-exports/; `--build` compiles them into data/observed-exports.json for the API.

Contract: retrieved vintage only. A missing row is not a zero. Rows become usable from the retrieval month,
because the historical publication month of these figures is not known (no backdating).

`--imports` collects the other leg of a route the same way: every candidate's reported imports of the panel items from the
world and from China (9 x 2 x 18 rows at most per month) into data/observed-imports/, compiled into
data/observed-imports.json. The app does not read it; the real-data benchmark (ml/benchmark/real_bench.py) does.

`--wider` collects the same two legs for eight more reporting countries that public reporting names as transit points
(WIDER below) into data/observed-wider-exports/ and data/observed-wider-imports/ (with --imports), compiled into
data/observed-wider.json. They widen the real-data benchmark's candidate lists only: the app's candidate set is unchanged.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT));import btl_paths
from adapters.contracts import COUNTRIES, ISO, ITEMS_BASE, ITEMS_EXTRA, display_hs, month_index  # noqa: E402
from common import CANDIDATES, TARGETS  # noqa: E402

FOLDER = ROOT / 'data/observed-exports'
EXTRA = ROOT / 'data/observed-exports-extra'  # the eight items of the expanded panel set, collected 2026-09 with --extra
OUTPUT = ROOT / 'data/observed-exports.json'
IMPORTS = ROOT / 'data/observed-imports'
IMPORTS_OUTPUT = ROOT / 'data/observed-imports.json'
IMPORT_PARTNERS = {'0': 'WLD', '156': 'CHN'}  # world total and the supplier of route definition v2
WIDER = {'268': 'GEO', '31': 'AZE', '496': 'MNG', '688': 'SRB', '344': 'HKG', '764': 'THA', '458': 'MYS', '762': 'TJK'}  # evaluation-only reporters
WIDER_EXPORTS = ROOT / 'data/observed-wider-exports'
WIDER_IMPORTS = ROOT / 'data/observed-wider-imports'
WIDER_OUTPUT = ROOT / 'data/observed-wider.json'
ALL_CODES = ITEMS_BASE + ['880730'] + ITEMS_EXTRA
ENDPOINT = 'https://comtradeapi.un.org/public/v1/preview/C/M/HS?'
DATA_ENDPOINT = 'https://comtradeapi.un.org/data/v1/get/C/M/HS?'  # subscription key required


def periods(first, last):
    y, m = int(first[:4]), int(first[4:])
    while f'{y}{m:02d}' <= last:
        yield f'{y}{m:02d}'
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def api_key():
    """The subscription key from the environment or the project's .env (COMTRADE_API_KEY), or None."""
    key = os.environ.get('COMTRADE_API_KEY') or os.environ.get('UN_COMTRADE_API_KEY')
    env = ROOT / '.env'
    if not key and env.exists():
        for line in env.read_text(encoding='utf-8').splitlines():
            if line.strip().startswith('COMTRADE_API_KEY='):
                key = line.split('=', 1)[1].strip().strip('"').strip("'")
    return key or None


def collect(first, last, limit, folder=FOLDER, codes=None, flow='X', partners=None, reporters=None):
    codes = codes or ITEMS_BASE + ['880730'];FOLDER = folder
    partners = partners or [str(int(COUNTRIES[t])) for t in TARGETS]
    reporters = reporters or [str(int(COUNTRIES[c])) for c in CANDIDATES]
    FOLDER.mkdir(parents=True, exist_ok=True)
    key = api_key()
    endpoint, cap = (DATA_ENDPOINT, 100000) if key else (ENDPOINT, 500)
    headers = {'User-Agent': 'before-the-list-local-research/1.0', **({'Ocp-Apim-Subscription-Key': key} if key else {})}
    calls = 0
    for period in periods(first, last):
        manifest = FOLDER / f'{period}.manifest.json'
        if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8'))['status'] == 'received':
            continue
        if calls >= limit:
            break
        params = dict(reporterCode=','.join(reporters), partnerCode=','.join(partners),
                      flowCode=flow, period=period, cmdCode=','.join(codes), partner2Code=0, customsCode='C00', motCode=0, maxRecords=cap)
        url = endpoint + urlencode(params)  # the key travels in a header, so the recorded URL holds no secret
        item = dict(url=url, endpoint='data (subscription key)' if key else 'public preview', period=period, retrieved_at=dt.datetime.now(dt.timezone.utc).isoformat(), raw_file=f'{period}.json')
        calls += 1
        time.sleep(1.5)
        try:
            with urlopen(Request(url, headers=headers), timeout=60) as response:
                blob = response.read()
            (FOLDER / item['raw_file']).write_bytes(blob)
            body = json.loads(blob)
            data = body.get('data')
            item.update(sha256=hashlib.sha256(blob).hexdigest(), rows=len(data) if isinstance(data, list) else 0,
                        status='received' if isinstance(data, list) and len(data) < cap and not body.get('error') else 'unknown_or_truncated')
        except HTTPError as exc:
            item.update(status='request_failed', error=str(exc), http_status=exc.code)
        except Exception as exc:  # network errors are recorded, never turned into data
            item.update(status='request_failed', error=str(exc))
        manifest.write_text(json.dumps(item, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
        print(json.dumps({k: item.get(k) for k in ('period', 'status', 'rows', 'error')}), flush=True)
        if item['status'] == 'request_failed':
            if item.get('http_status') == 429:
                time.sleep(20)
            else:
                break
    return calls


def build():
    rows, months = [], 0
    for manifest in sorted(FOLDER.glob('*.manifest.json')) + sorted(EXTRA.glob('*.manifest.json')):
        item = json.loads(manifest.read_text(encoding='utf-8'))
        if item['status'] != 'received':
            continue
        months += manifest.parent == FOLDER
        available = month_index(item['retrieved_at'][:7])
        for r in json.loads((manifest.parent / item['raw_file']).read_text(encoding='utf-8'))['data']:
            reporter, partner = ISO.get(f"{r['reporterCode']:03d}"), ISO.get(f"{r['partnerCode']:03d}")
            hs = display_hs(str(r['cmdCode']), str(r['cmdCode']), r.get('classificationCode'))
            if reporter not in CANDIDATES or partner not in TARGETS or reporter == partner or hs is None or r.get('primaryValue') is None:
                continue
            rows.append(dict(reporter=reporter, partner=partner, hs=hs, month=month_index(str(r['period'])), usd=float(r['primaryValue']),
                             edition=r.get('classificationCode'), available_month=available, period=str(r['period'])))
    rows.sort(key=lambda r: (r['partner'], r['hs'], r['reporter'], r['month']))
    OUTPUT.write_text(json.dumps(dict(source='UN Comtrade public preview API (monthly, exports reported by the candidate)', months_received=months,
                                      contract='retrieved vintage; missing is not zero; usable from the retrieval month', rows=rows),
                                 ensure_ascii=False, separators=(',', ':')) + '\n', encoding='utf-8', newline='\n')  # LF everywhere: manifest.json hashes this file
    print(json.dumps({'months': months, 'rows': len(rows)}))


def build_imports():
    rows, months = [], 0
    for manifest in sorted(IMPORTS.glob('*.manifest.json')):
        item = json.loads(manifest.read_text(encoding='utf-8'))
        if item['status'] != 'received':
            continue
        months += 1
        available = month_index(item['retrieved_at'][:7])
        for r in json.loads((IMPORTS / item['raw_file']).read_text(encoding='utf-8'))['data']:
            reporter, partner = ISO.get(f"{r['reporterCode']:03d}"), IMPORT_PARTNERS.get(str(r['partnerCode']))
            hs = display_hs(str(r['cmdCode']), str(r['cmdCode']), r.get('classificationCode'))
            if reporter not in CANDIDATES or partner is None or reporter == partner or hs is None or r.get('primaryValue') is None or r.get('flowCode') != 'M':
                continue
            rows.append(dict(reporter=reporter, partner=partner, hs=hs, month=month_index(str(r['period'])), usd=float(r['primaryValue']),
                             edition=r.get('classificationCode'), available_month=available, period=str(r['period'])))
    rows.sort(key=lambda r: (r['partner'], r['hs'], r['reporter'], r['month']))
    IMPORTS_OUTPUT.write_text(json.dumps(dict(source='UN Comtrade public preview API (monthly, imports reported by the candidate; partner WLD = world, CHN = China)',
                                              months_received=months, contract='retrieved vintage; missing is not zero; usable from the retrieval month', rows=rows),
                                         ensure_ascii=False, separators=(',', ':')) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'months': months, 'rows': len(rows)}))


def build_wider():
    out = {}
    for name, folder, flow in (('exports', WIDER_EXPORTS, 'X'), ('imports', WIDER_IMPORTS, 'M')):
        rows, months = [], 0
        for manifest in sorted(folder.glob('*.manifest.json')):
            item = json.loads(manifest.read_text(encoding='utf-8'))
            if item['status'] != 'received':
                continue
            months += 1
            available = month_index(item['retrieved_at'][:7])
            for r in json.loads((folder / item['raw_file']).read_text(encoding='utf-8'))['data']:
                reporter = WIDER.get(str(r['reporterCode']))
                partner = ISO.get(f"{r['partnerCode']:03d}") if flow == 'X' else IMPORT_PARTNERS.get(str(r['partnerCode']))
                hs = display_hs(str(r['cmdCode']), str(r['cmdCode']), r.get('classificationCode'))
                if reporter is None or partner is None or (flow == 'X' and partner not in TARGETS) or hs is None or r.get('primaryValue') is None or r.get('flowCode') != flow:
                    continue
                rows.append(dict(reporter=reporter, partner=partner, hs=hs, month=month_index(str(r['period'])), usd=float(r['primaryValue']),
                                 edition=r.get('classificationCode'), available_month=available, period=str(r['period'])))
        rows.sort(key=lambda r: (r['partner'], r['hs'], r['reporter'], r['month']))
        out[name] = dict(months_received=months, rows=rows)
    WIDER_OUTPUT.write_text(json.dumps(dict(source='UN Comtrade public preview API (monthly, reported by the listed countries)', reporters=sorted(WIDER.values()),
                                            contract='retrieved vintage; missing is not zero; evaluation-only reporters, not candidates of the app', **out),
                                       ensure_ascii=False, separators=(',', ':')) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({k: {'months': v['months_received'], 'rows': len(v['rows'])} for k, v in out.items()}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--first', default='202001')
    p.add_argument('--last', default='202606')
    p.add_argument('--limit', type=int, default=0, help='number of new requests this run (0 = only build)')
    p.add_argument('--extra', action='store_true', help='collect the expanded panel items into data/observed-exports-extra')
    p.add_argument('--imports', action='store_true', help='collect candidate-reported imports (world and China) of every panel item into data/observed-imports')
    p.add_argument('--wider', action='store_true', help='collect the evaluation-only reporters (WIDER) instead of the candidates; with --imports their imports')
    a = p.parse_args()
    if a.wider:
        if a.limit:
            if a.imports:
                collect(a.first, a.last, a.limit, WIDER_IMPORTS, ALL_CODES, 'M', list(IMPORT_PARTNERS), list(WIDER))
            else:
                collect(a.first, a.last, a.limit, WIDER_EXPORTS, ALL_CODES, 'X', None, list(WIDER))
        build_wider()
    elif a.imports:
        if a.limit:
            collect(a.first, a.last, a.limit, IMPORTS, ALL_CODES, 'M', list(IMPORT_PARTNERS))
        build_imports()
    else:
        if a.limit:
            collect(a.first, a.last, a.limit, *((EXTRA, ITEMS_EXTRA) if a.extra else ()))
        build()
