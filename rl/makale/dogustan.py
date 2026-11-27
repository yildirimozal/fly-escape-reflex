"""Makale 1: "Ölümcül kararlar neden doğuştandır?" - ilk gün devam/dur testi.

Test: zararsız (sahte) gölgelerin oranı p_sahte = 0,8 olan dünyada öğrenen kaçış devresi (PPO, kacis_devresi ile aynı)
boşuna kalkışı bağlantı şeması refleksinin altına indirip gerçek gölgelerde >= 9/10 kaçışı koruyabiliyor mu?
Karşılaştırma: refleksin bütün eşik ayarları (10-35°, + 7 ms dev lif gecikmesi) aynı sınav sahnelerinde.

  python dogustan.py esik                -> esik_taramasi.json
  python dogustan.py egit 0.8 200000     -> ogrenen_p080.csv, modeller/ogrenen_p080_<adim>.zip
"""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
RL = os.path.dirname(HERE)
sys.path.insert(0, RL)
from kacis_devresi import KacisOrtam, bolum, sinav  # noqa: E402

MOD = os.path.join(HERE, "modeller")


class Cevre(KacisOrtam):
    """Eğitim sahnesi: gölge 0,2-1,8 s'de; (1 - p_sahte) olasılıkla gerçek."""

    def __init__(self, p_sahte=0.5, seed=None):
        super().__init__(seed=seed)
        self.p_sahte = p_sahte

    def reset(self, seed=None, options=None):
        if self.egitim:
            self.env.sahne_zorla = dict(gorev="kacis", golgeler=[dict(t0=self.rng.uniform(0.2, 1.8),
                                                                       gercek=self.rng.random() >= self.p_sahte)])
        self.o, info = self.env.reset(seed=seed)
        return self._o(), info


def esik_taramasi():
    ortam = KacisOrtam(); ortam.egitim = False
    out = {}
    for derece in (6.5, 10, 15, 20, 25, 30, 35, 45):
        ortam.env.esik = np.radians(derece)
        g = f = 0
        for i in range(20):
            gercek = i < 10
            sahne = dict(gorev="kacis", golgeler=[dict(t0=0.4 + 0.15 * (i % 10), gercek=gercek)])
            info, _ = bolum(ortam, None, sahne, 1000 + i, refleks=True)
            if gercek:
                g += not info["yakalandi"]
            else:
                f += info["kalkti"] is not None
        out[derece] = dict(kacis=g, bosuna_kalkis=f)
        print(f"eşik {derece}°: kaçış {g}/10, boşuna kalkış {f}/10", flush=True)
    json.dump(out, open(os.path.join(HERE, "esik_taramasi.json"), "w"), indent=1)


def egit(p_sahte, toplam):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
    torch.set_num_threads(2)
    os.makedirs(MOD, exist_ok=True)
    ad = f"ogrenen_p{int(round(p_sahte * 100)):03d}"
    venv = VecMonitor(SubprocVecEnv([(lambda i=i: Cevre(p_sahte, seed=700 + i)) for i in range(8)]))
    model = PPO("MlpPolicy", venv, learning_rate=3e-4, n_steps=512, batch_size=512, n_epochs=10, gamma=0.99,
                gae_lambda=0.95, ent_coef=0.05, policy_kwargs=dict(net_arch=[64, 64]), seed=1, device="cpu", verbose=0)
    f = open(os.path.join(HERE, ad + ".csv"), "w"); f.write("adim,kacis,bosuna_kalkis,odul,gecti,sure_dk\n")
    t0 = time.time()
    while model.num_timesteps < toplam:
        model.learn(total_timesteps=25_000, reset_num_timesteps=False)
        n = model.num_timesteps
        model.save(os.path.join(MOD, f"{ad}_{n}"))
        s = sinav(model)
        f.write(f"{n},{s['kacis']},{s['bosuna_kalkis']},{s['odul']:.2f},{int(s['gecti'])},{(time.time() - t0) / 60:.1f}\n"); f.flush()
        print(f"[{ad}] adım {n}: kaçış {s['kacis']}/10, boşuna kalkış {s['bosuna_kalkis']}/10, geçti={s['gecti']}", flush=True)
    venv.close()
    print(f"{ad} BITTI", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "esik":
        esik_taramasi()
    else:
        egit(float(sys.argv[2]), int(sys.argv[3]))
