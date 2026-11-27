"""Öğrenmiş devrelerin adım başına kalkış olasılığı (stokastik politikanın kendisi).
Refleksin ilk 20 karşılaşması (tohum 4242 + i) hiç kalkmadan yürütülür; her 10 ms adımda politikanın P(kalk) değeri okunur.
  arka plan: gölge görünmezken adım başına ortalama P(kalk)
  tepe     : gölge görünürken bölümdeki en yüksek P(kalk) (20 bölümün medyanı)
  python olasilik.py  -> olasilik_ozet.json  (tarama/ içindeki bütün ogren_* ve olum_* modelleri)
"""
import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from tarama import CIKTI, Ortam  # noqa: E402


def olc(ad, n=20):
    from stable_baselines3 import PPO
    m = PPO.load(os.path.join(CIKTI, ad + ".zip"), device="cpu")
    ort = Ortam(10, seed=4242)
    arka, tepe = [], []
    for i in range(n):
        torch.manual_seed(i); o, _ = ort.reset(seed=4242 + i); ps = []
        while True:
            with torch.no_grad():
                p = m.policy.get_distribution(m.policy.obs_to_tensor(o)[0]).distribution.probs[0, 1].item()
            ps.append((bool(o[0] > 0), p))
            o, r, b, kes, info = ort.step(0)
            if b or kes:
                break
        arka += [p for v, p in ps if not v]
        g = [p for v, p in ps if v]
        if g:
            tepe.append(max(g))
    return dict(arka_plan=float(np.mean(arka)), tepe_medyan=float(np.median(tepe)), tepe_05_ustu=float(np.mean(np.array(tepe) > 0.5)))


if __name__ == "__main__":
    torch.set_num_threads(1)
    out = {}
    for yol in sorted(glob.glob(os.path.join(CIKTI, "*.zip"))):
        ad = os.path.basename(yol)[:-4]
        out[ad] = olc(ad)
        print(ad, {k: round(v, 3) for k, v in out[ad].items()}, flush=True)
    json.dump(out, open(os.path.join(HERE, "olasilik_ozet.json"), "w"), indent=1)
