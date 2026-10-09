from pathlib import Path
from contextlib import contextmanager
import hashlib, json, sqlite3
import numpy as np
ROOT=Path(__file__).resolve().parents[1]  # project root; every data and model path hangs off it
DB=ROOT/'data/test.db'
TARGETS=['RUS','IRN','PRK','CHN']
CANDIDATES=['CHN','KAZ','TUR','ARE','ARM','UZB','KGZ','IND','VNM']
def candidates_for(target):return [c for c in CANDIDATES if c!=target]
from adapters.contracts import ITEMS,ITEMS_ALL,ITEMS_BASE,ITEMS_EXTRA,panel_set  # the active item set is a deployment setting, see contracts.py
NAMES_ALL=['프로세서·컨트롤러','볼 베어링','머시닝센터','데이터 송수신 장비','측량·기상 관측기기','트랜지스터(1W 이상)','원심펌프','항공기 기타 부품','전기 제어반(1kV 이하)',
           '무선항법 장치','전산 처리장치','테이퍼 롤러 베어링','오실로스코프','기타 집적회로','회로 접속기기','NC 수평 선반','정지형 변환기(정류기)']
NAMES=NAMES_ALL[:len(ITEMS)]
FEATURES=['import_growth','export_growth','import_yoy','export_yoy','coupled_growth','import_volatility','export_volatility','coverage','mirror_gap','event_count','months_since_event','horizon']
def mi(date): return (int(date[:4])-1990)*12+int(date[5:7])-1
def mt(m):return f'{1990+m//12:04d}-{m%12+1:02d}'
def seed(text):return int(hashlib.sha256(text.encode()).hexdigest()[:8],16)
@contextmanager
def connect():
 c=sqlite3.connect(DB);c.row_factory=sqlite3.Row
 try:
  with c:yield c
 finally:c.close()
class Panel:
 def __init__(self):
  with connect() as c:
   self.events=[dict(r) for r in c.execute('select * from events')]
   self.end=int(c.execute("select value from metadata where key='panel_end'").fetchone()[0]);self.rows={}
   for r in c.execute('select * from trade order by month,revision'):
    key=(r['target'],r['hs'],r['candidate']);self.rows.setdefault(key,[]).append(dict(r))
 def history(self,key,basis,length=24):
  rows={}
  for r in self.rows[key]:
   if basis-length<r['month']<=basis and r['available_month']<=basis:rows[r['month']]=r
  return [rows.get(m,{'month':m,'inbound':None,'outbound':None,'mirror':None,'available_month':None}) for m in range(max(0,basis-length+1),basis+1)]
 def features(self,key,basis,horizon,hist=None):
  hist=hist if hist is not None else self.history(key,basis);a=np.array([[r['inbound'] if r['inbound'] is not None else np.nan,r['outbound'] if r['outbound'] is not None else np.nan] for r in hist]);latest=hist[-12:]
  known=np.where(np.isfinite(a).any(axis=1))[0]
  if len(known):a=a[:known[-1]+1]
  def avg(x):return float(np.nanmean(x)) if np.isfinite(x).any() else 0.
  cur=np.array([avg(a[-3:,j]) for j in range(2)]);base=np.array([avg(a[-6:-3,j]) for j in range(2)]) if len(a)>3 else cur
  growth=np.clip(np.where(base>0,cur/np.maximum(base,1)-1,0),-1,5);year=np.array([avg(a[-15:-12,j]) for j in range(2)]) if len(a)>=15 else cur
  yoy=np.clip(np.where(year>0,cur/np.maximum(year,1)-1,0),-1,5)
  vol=[float(np.nanstd(a[-12:,j])/max(avg(a[-12:,j]),1)) if np.isfinite(a[-12:,j]).any() else 0 for j in range(2)]
  cov=sum((r[k] is not None) for r in latest for k in ['inbound','outbound'])/(len(latest)*2)
  mirror=[abs(r['outbound']-r['mirror'])/max(r['outbound'],1) for r in latest if r['outbound'] is not None and r['mirror'] is not None]
  events=[e for e in self.events if e['target']==key[0] and mi(e['published'])<=basis and mi(e['effective'])<=basis]
  age=min(120,basis-max([mi(e['effective']) for e in events],default=basis-120))
  x=[*growth,*yoy,min(max(growth[0],0),max(growth[1],0)),*vol,cov,float(np.mean(mirror)) if mirror else 0,len(events),age,horizon]
  return np.array(x,dtype=float),cur,latest,events
