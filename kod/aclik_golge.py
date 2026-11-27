"""Beslenirken gölge: şeker nöronları + LPLC2 birlikte uyarılınca dev lif (DNp01) zayıflıyor mu?
Karşılaştırma: yalnız LPLC2 (mevcut lplc2.parquet) ve özgüllük kontrolü acı + LPLC2.
Ayarlar deneyler.py ile aynı: 20 deneme x 1 s, 100 Hz Poisson. -> sonuclar/aclik_golge.json (ozet.json'a dokunmaz)"""
import sys, json, time; sys.path.insert(0, '.')
import ortak as O
from model import run_exp, default_params
from brian2 import Hz
import numpy as np, pandas as pd
N = 20
GF = O.tip("cell_type", "DNp01")
def oz(ad):   # analiz.py ile aynı hesap (içe aktarılmadı: analiz.py ozet.json'ı yeniden yazıyor)
    df = pd.read_parquet(O.SONUC / f"{ad}.parquet"); r = df.groupby('flywire_id').size() / N
    ilk = df.groupby(['trial', 'flywire_id']).t.min().groupby('flywire_id').median()
    return dict(ad=ad, aktif=int((r >= 1).sum()), mn9=[round(float(r.get(m, 0)), 1) for m in O.MN9],
                gf=[round(float(r.get(g, 0)), 1) for g in GF],
                mn9_gecikme_ms=[round(float(ilk.get(m, np.nan)) * 1000, 1) for m in O.MN9],
                gf_gecikme_ms=[round(float(ilk.get(g, np.nan)) * 1000, 1) for g in GF])
P = dict(default_params); P['n_run'] = 20; P['r_poi'] = 100 * Hz
LPLC2 = O.tip("cell_type", "LPLC2")
sonuc = {}
for ad, exc in (("seker_lplc2", O.SEKER + LPLC2), ("aci_lplc2", O.ACI + LPLC2)):
    if not (O.SONUC / f"{ad}.parquet").exists():
        t = time.time(); run_exp(ad, list(exc), O.SONUC, O.COMP, O.CON, params=P, n_proc=4)
        print(ad, "sure dk", round((time.time() - t) / 60, 1), flush=True)
for ad in ("lplc2", "seker", "seker_lplc2", "aci_lplc2"):
    sonuc[ad] = oz(ad); print(sonuc[ad], flush=True)
json.dump(sonuc, open(O.SONUC / "aclik_golge.json", "w"), indent=1, default=str)
print("ACLIK_GOLGE BITTI", flush=True)
