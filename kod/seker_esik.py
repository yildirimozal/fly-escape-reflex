"""Ters yön testi, tavan dışında: şeker tadı dev lifin (DNp01) gölge eşiğini yükseltiyor mu?
aclik_golge.py'de LPLC2 100 Hz'te dev lif doyumdaydı (yaklaşık 148 Hz), etki görünmezdi. Burada LPLC2 uyarım hızı
eşik bölgesinde taranır; her hızda şekersiz ve şekerli (şeker GRN'leri 100 Hz, aclik_golge ile aynı) koşulda ölçülür.
Gölge = 210 LPLC2'nin hepsi, r_poi2 hızında (zayıf ya da uzak gölge = düşük hız). Deneme 200 ms: GF 7-10 ms'de ateşler.
  python seker_esik.py dene                -> süre ölçümü (1 koşul, 2 deneme)
  python seker_esik.py tara 20 30 40 ...   -> sonuclar/seker_esik/{yalniz,seker}_<hz>.parquet, seker_esik.json
"""
import json
import sys
import time

sys.path.insert(0, '.')
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from brian2 import Hz, ms  # noqa: E402

import ortak as O  # noqa: E402
from model import default_params, run_exp  # noqa: E402

GF = O.tip("cell_type", "DNp01")
LPLC2 = O.tip("cell_type", "LPLC2")
CIKTI = O.SONUC / "seker_esik"
N_DENEME, T_MS = 12, 200


def kos(ad, hz, seker, n=N_DENEME, n_proc=6):
    P = dict(default_params); P['n_run'] = n; P['t_run'] = T_MS * ms
    P['r_poi'] = 100 * Hz; P['r_poi2'] = hz * Hz
    t = time.time()
    run_exp(ad, list(O.SEKER) if seker else [], CIKTI, O.COMP, O.CON, params=P, neu_exc2=list(LPLC2), n_proc=n_proc)
    return (time.time() - t) / 60


def oz(ad, n=N_DENEME):
    yol = CIKTI / f"{ad}.parquet"
    df = pd.read_parquet(yol)
    out = {}
    for ad_n, ids in (("gf", GF), ("mn9", O.MN9)):
        s = df[df.flywire_id.isin(ids)]
        hiz = [float((s[s.flywire_id == i].shape[0]) / n / (T_MS / 1000)) for i in ids]
        ilk = s.groupby(['trial', 'flywire_id']).t.min()
        ates = [float(np.mean([(tr, i) in ilk.index for tr in range(n)])) for i in ids]
        gec = [float(ilk.xs(i, level='flywire_id').median() * 1000) if i in ilk.index.get_level_values(1) else None for i in ids]
        out[ad_n] = dict(hiz_hz=[round(h, 1) for h in hiz], ateslenen_deneme=ates, ilk_ms_medyan=gec,
                         deneme_sayim=[int((s.trial == tr).sum()) for tr in range(n)],          # iki nöron toplamı
                         deneme_ilk_ms=[float(s[s.trial == tr].t.min() * 1000) if (s.trial == tr).any() else None
                                        for tr in range(n)])
    return out


if __name__ == "__main__":
    CIKTI.mkdir(exist_ok=True)
    if sys.argv[1] == "dene":
        dk = kos("dene_seker_50", 50, True, n=2, n_proc=2)
        print("dene süre dk", round(dk, 2), oz("dene_seker_50", n=2), flush=True)
    else:
        hizlar = [int(x) for x in sys.argv[2:]]
        sonuc = {}
        for hz in hizlar:
            for ad, seker in ((f"yalniz_{hz}", False), (f"seker_{hz}", True)):
                if not (CIKTI / f"{ad}.parquet").exists():
                    print(ad, "süre dk", round(kos(ad, hz, seker), 1), flush=True)
        for yol in sorted(CIKTI.glob("*.parquet")):                     # biten bütün koşullar
            if not yol.stem.startswith("dene"):
                sonuc[yol.stem] = oz(yol.stem); print(yol.stem, sonuc[yol.stem]["gf"]["hiz_hz"], flush=True)
        json.dump(sonuc, open(O.SONUC / "seker_esik.json", "w"), indent=1)
        print("SEKER_ESIK BITTI", flush=True)
