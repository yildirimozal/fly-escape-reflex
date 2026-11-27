"""Öğrenmiş devreleri deterministik politikayla değerlendirir: refleksle aynı 300 karşılaşma (tarama.sabit ile aynı
tohumlar: ortam 4242, bölüm i -> reset(4242 + i), torch.manual_seed(i)). Eğitim kaydındaki ölçüler stokastik politikadandı.
  python deterministik.py <model_adi> [n=300]   -> tarama/det_<model_adi>.csv (tarama.py ile aynı sütunlar)
  model_adi: ogren_b<B>_s<s> | olum_k<K>_b<B>_s<s>
"""
import os
import re
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from tarama import CIKTI, Ortam, yaz  # noqa: E402


def degerlendir(ad, n=300):
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    B = int(re.search(r"_b(\d+)_", ad).group(1))
    model = PPO.load(os.path.join(CIKTI, ad + ".zip"), device="cpu")
    ortam = Ortam(B, seed=4242)
    satir = []
    while len(satir) < n:
        torch.manual_seed(len(satir))
        o, _ = ortam.reset(seed=4242 + len(satir)); R = 0.0
        while True:
            k = int(model.predict(o, deterministic=True)[0])
            o, r, b, kes, info = ortam.step(k); R += r
            if b or kes:
                break
        satir.append([len(satir), 0, int(bool(info["gercek"])), int(info["kalkti"] is not None), int(bool(info["yakalandi"])), round(R, 2)])
    yaz(os.path.join(CIKTI, f"det_{ad}.csv"), satir)
    print("det", ad, "bolum", n, "yakalanma", sum(r[4] for r in satir), "kalkis", sum(r[3] for r in satir), flush=True)


if __name__ == "__main__":
    degerlendir(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 300)
