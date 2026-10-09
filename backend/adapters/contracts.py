"""Explicit country, classification and observation contracts shared by ETL/API."""
COUNTRIES = {'RUS':'643','IRN':'364','PRK':'408','CHN':'156','KAZ':'398',
             'TUR':'792','ARE':'784','ARM':'051','UZB':'860','KGZ':'417',
             'IND':'356','VNM':'704','DEU':'276','JPN':'392','KOR':'410'}
ISO = {v:k for k,v in COUNTRIES.items()}
ROUTE_SUPPLIERS = {'CHN':'KOR','KAZ':'DEU','TUR':'DEU','ARE':'JPN','ARM':'DEU',
                   'UZB':'KOR','KGZ':'JPN','IND':'DEU','VNM':'KOR'}
# Route definition v1 (above) is kept for the candidate list/order and for targets without observed supply.
# Route definition v2 (decision B, 2026-09-18): for Russia the six candidates with observed China-sourced
# imports take China as supplier; every other route keeps its v1 supplier (demonstration only).
ROUTE_DEFINITION = 'route-v2-chn-supply'
CHN_SUPPLY_ROUTES = {'RUS': ('ARM', 'IND', 'KAZ', 'KGZ', 'TUR', 'UZB')}
def supplier_for(target, candidate):
    return 'CHN' if candidate in CHN_SUPPLY_ROUTES.get(target, ()) else ROUTE_SUPPLIERS[candidate]
def route_suppliers(target):
    return {c: supplier_for(target, c) for c in ROUTE_SUPPLIERS if c != target}
# Panel item sets. `base` is the original nine; `expanded` adds eight Common High Priority List codes whose HS6 text is the same in
# HS 2017 and HS 2022. The active set is a deployment setting (BTL_PANEL_SET, else models/panel-set.json, else base): the model pack
# and the item list must match, and switching back to `base` with the archived pack restores the previous behaviour exactly.
ITEMS_BASE = ['854231','848210','845710','851762','901580','854129','841370','880330','853710']
ITEMS_EXTRA = ['852691','847150','848220','903020','854239','853690','845811','850440']
ITEMS_ALL = ITEMS_BASE + ITEMS_EXTRA
def panel_set():
    import json, os
    from pathlib import Path
    value = os.environ.get('BTL_PANEL_SET')
    if not value:
        path = Path(__file__).resolve().parents[2] / 'models/panel-set.json'
        value = json.loads(path.read_text(encoding='utf-8')).get('set') if path.exists() else 'base'
    return 'expanded' if value == 'expanded' else 'base'
ITEMS = ITEMS_ALL if panel_set() == 'expanded' else list(ITEMS_BASE)
HS22 = {v: ('880730' if v == '880330' else v) for v in ITEMS_ALL if v != '854231'}

def display_hs(canonical, reported=None, edition=None, ambiguous=False):
    # Native HS2017 is usable even when its later concordance is ambiguous.
    if edition == 'H5' and reported in ITEMS_ALL:
        return reported
    if ambiguous or canonical == '854231':
        return None
    return next((old for old,new in HS22.items() if canonical == new), None)

def month_index(period):
    value = str(period).replace('-', '')[:6]
    if len(value) != 6 or not 1 <= int(value[4:]) <= 12:
        raise ValueError('Expected monthly YYYYMM period')
    return (int(value[:4])-1990)*12+int(value[4:])-1

def causal_fill(history, method):
    """Predict one absent month using only preceding values; never replace zero."""
    known = [v for v in history[-3:] if v is not None]
    if method == 'seasonal' and len(history) >= 12 and history[-12] is not None:
        return float(history[-12])
    if method == 'last' and history and history[-1] is not None:
        return float(history[-1])
    if method == 'median3' and len(known) >= 2:
        return float(sorted(known)[len(known)//2]) if len(known)%2 else float(sum(known)/len(known))
    return None
