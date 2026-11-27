"""Butun deneyler: her biri 20 deneme x 1 s, uyarim 100 Hz Poisson (Shiu vd. 2024 ayarlari)."""
import sys, json; sys.path.insert(0, '.')
import pandas as pd
import ortak as O
from model import run_exp, default_params
from brian2 import Hz
P = dict(default_params); P['n_run'] = 20; P['r_poi'] = 100 * Hz
O.SONUC.mkdir(exist_ok=True)
LPLC2 = O.tip("cell_type", "LPLC2"); LC4 = O.tip("cell_type", "LC4"); GF = O.tip("cell_type", "DNp01")
def kos(ad, exc, slnc=()):
    run_exp(ad, list(exc), O.SONUC, O.COMP, O.CON, params=P, neu_slnc=list(slnc), n_proc=4)
def oran(ad):
    df = pd.read_parquet(O.SONUC / f"{ad}.parquet"); return df.groupby('flywire_id').size() / P['n_run']
kos("seker", O.SEKER)
kos("aci", O.ACI)
kos("seker_aci", O.SEKER + O.ACI)
kos("lplc2", LPLC2)
kos("lplc2_lc4", LPLC2 + LC4)
# susturma adaylari: seker deneyinde en aktif duyusal olmayan 12 noron
r = oran("seker"); a = O.an()
aday = [int(i) for i in r.sort_values(ascending=False).index if a.reindex([i]).super_class.iloc[0] not in ("sensory",) and i not in O.MN9][:12]
json.dump(aday, open(O.SONUC / "susturma_adaylari.json", "w"))
for i in aday:
    kos(f"seker_sus_{i}", O.SEKER, [i])
print("DENEYLER_BITTI", flush=True)
