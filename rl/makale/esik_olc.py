"""Doğuştan refleksin eşik ailesi, bedel taramasıyla aynı sahnelerde (tarama.sabit ile aynı tohumlar, 300 karşılaşma).
Eşik = gölgenin gözde kapladığı açı; aşılınca 7 ms sonra kalkış. 20° = tarama/refleks_b*.csv (bağlantı şeması değeri).
Refleksin davranışı B'den bağımsızdır (B yalnız ödülü değiştirir), bu yüzden B=10 ortamı kullanılır.
  python esik_olc.py 25 30 35 40 50 60  -> tarama/esik_<derece>.csv
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tarama import CIKTI, Ortam, yaz  # noqa: E402


def olc(derece, n=300):
    ortam = Ortam(10, seed=4242)
    ortam.env.refleks = True
    ortam.env.esik = np.radians(derece)
    satir = []
    while len(satir) < n:
        torch.manual_seed(len(satir))
        o, _ = ortam.reset(seed=4242 + len(satir)); R = 0.0
        while True:
            o, r, b, kes, info = ortam.step(0); R += r
            if b or kes:
                break
        satir.append([len(satir), 0, int(bool(info["gercek"])), int(info["kalkti"] is not None), int(bool(info["yakalandi"])), round(R, 2)])
    yaz(os.path.join(CIKTI, f"esik_{derece:g}.csv"), satir)
    g = np.array([s[2] for s in satir]); k = np.array([s[3] for s in satir]); y = np.array([s[4] for s in satir])
    print(f"eşik {derece}°: gerçek {g.sum()} kaçış {1 - y[g == 1].mean():.3f}, sahtede kalkış {k[g == 0].mean():.3f}", flush=True)


if __name__ == "__main__":
    for d in sys.argv[1:]:
        olc(float(d))
    print("ESIK BITTI", flush=True)
