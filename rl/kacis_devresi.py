"""5. aşama: ayrı öğrenen kaçış devresi (dev lif yolunun öğrenen karşılığı).

Hareket beyni (SAC, 1-4. aşamalar, beyin_G4_tat) dondurulur ve yiyecek aramayı sürdürür (gürültülü politika).
Kaçış devresi küçük bir PPO ağı (ilk 4 deneme DQN'di): her 10 ms'de "bekle / kalk". Girdisi yalnız gözdeki gölge (açı, büyüme hızı,
yön sin/cos) ve ayaktaki tat (2). Ödül dünyadan: tehditsiz kalkış -10, gerçek gölgede kalkış -2,
yakalanma -10 ve kalan sürenin kaybı. Devre yalnız gözde gölge varken karar verir (1. deneme: rastgele
kalkışlar gölgeden önce olup -10 aldığı için devre hiç kalkmamayı öğrendi).
(Neden ayrı: tek SAC ağında aç/kapa kalkış öğrenilirken duruş ve koku becerileri üç denemede çöktü.)

  python kacis_devresi.py egit 300000   -> modeller/kacis_<adim>.zip, kacis_sinav.csv
  python kacis_devresi.py sinav <kacis.zip>
"""
import csv
import json
import os
import sys
import time

import gymnasium as gym
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dunya import Sinek  # noqa: E402
from mufredat import TERIM  # noqa: E402

HAREKET = os.path.join(HERE, "modeller", "beyin_G4_tat.zip")
MOD = os.path.join(HERE, "modeller")
GOZ = slice(108, 112)
TAT = slice(106, 108)
OLCUT = "10 gerçek gölgenin en az 9'undan kaçış ve 10 sahte gölgede en çok 3 boşuna kalkış"


class KacisOrtam(gym.Env):
    def __init__(self, seed=None):
        from stable_baselines3 import SAC
        torch.set_num_threads(1)
        self.env = Sinek(gorevler=["kacis"], terimler=TERIM["kacis"], seed=seed)
        self.hareket = SAC.load(HAREKET, device="cpu")
        self.observation_space = gym.spaces.Box(-10, 10, (6,), np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self.egitim = True
        self.kapi = True      # False: doğuştan "gölge varken karar" kapısı yok (makale: öğrenmenin bedeli)
        self.rng = np.random.default_rng(seed)

    def _o(self):
        return np.concatenate([self.o[GOZ], self.o[TAT]]).astype(np.float32)

    def reset(self, seed=None, options=None):
        if self.egitim:   # eğitim: gölge 0,2-1,8 s'de (sınavın 0,4-1,75 aralığını kapsar), yarısı gerçek
            self.env.sahne_zorla = dict(gorev="kacis", golgeler=[dict(t0=self.rng.uniform(0.2, 1.8), gercek=self.rng.random() < 0.5)])
        self.o, info = self.env.reset(seed=seed)
        return self._o(), info

    def step(self, k):
        oh = self.o.copy(); oh[GOZ] = 0.0     # hareket beyni gölge görmeden eğitildi; gölgeyi yalnız devre görür
        a, _ = self.hareket.predict(oh, deterministic=False)
        a = a.copy()
        # devre yalnız gözde gölge varken karar verir (dev lif yolu görsel uyarıyla tetiklenir); yoksa "bekle"
        a[48] = 1.0 if (k == 1 and (self.o[GOZ][0] > 0 or not self.kapi)) else -1.0
        self.o, r, b, kes, info = self.env.step(a)
        if self.env.kalkti is not None and not (b or kes):
            # kalkış tek eylem: uçuş programı sonuna kadar oynatılır, sonucun ödülü bu karara yazılır (4. deneme sonrası)
            while not (b or kes):
                a2 = a.copy(); a2[48] = -1.0
                self.o, r2, b, kes, info = self.env.step(a2)
                r += r2
            b, kes = True, False
        if self.egitim and not b and self.env.golge is None and self.env.golgeler and \
                all(self.env.t > gl["t0"] for gl in self.env.golgeler):
            kes = True    # gölge geçti: eğitim bölümü burada kesilir (2. deneme: tek karar/4 s örnek azlığı)
        return self._o(), r, b, kes, info


def bolum(ortam, devre, sahne, seed, refleks=False):
    ortam.env.sahne_zorla = sahne
    ortam.env.refleks = refleks
    torch.manual_seed(seed)
    o, _ = ortam.reset(seed=seed)
    R = 0.0
    while True:
        k = int(devre.predict(o, deterministic=True)[0]) if devre is not None else 0
        o, r, b, kes, info = ortam.step(k)
        R += r
        if b or kes:
            return info, R


def sinav(devre, refleks=False):
    ortam = KacisOrtam()
    ortam.egitim = False
    sonuc = []
    for i in range(20):
        gercek = i < 10
        sahne = dict(gorev="kacis", golgeler=[dict(t0=0.4 + 0.15 * (i % 10), gercek=gercek)])
        info, R = bolum(ortam, devre, sahne, 1000 + i, refleks)
        sonuc.append(dict(gercek=gercek, kalkti=info["kalkti"] is not None, yakalandi=info["yakalandi"], R=R,
                          esik_t=ortam.env.esik_t, kalkis_t=info["kalkti"]))
    g = [s for s in sonuc if s["gercek"]]; f = [s for s in sonuc if not s["gercek"]]
    kac = sum(not s["yakalandi"] for s in g); bos = sum(s["kalkti"] for s in f)
    # kalkış anında gölgenin gözdeki açısı yerine eşikten farkı (gerçek gölgelerde)
    fark = [round((s["kalkis_t"] - s["esik_t"]) * 1000, 1) for s in g if s["kalkis_t"] is not None and s["esik_t"] is not None]
    return dict(gecti=kac >= 9 and bos <= 3, kacis=kac, bosuna_kalkis=bos, odul=float(np.mean([s["R"] for s in sonuc])),
                kalkis_esige_gore_ms=fark, ayrinti=sonuc)


def egit(toplam):
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
    torch.set_num_threads(2)
    venv = VecMonitor(SubprocVecEnv([(lambda i=i: KacisOrtam(seed=500 + i)) for i in range(8)]))
    model = PPO("MlpPolicy", venv, learning_rate=3e-4, n_steps=512, batch_size=512, n_epochs=10, gamma=0.99,
                gae_lambda=0.95, ent_coef=0.05, policy_kwargs=dict(net_arch=[64, 64]), seed=1, device="cpu", verbose=0)
    model.save(os.path.join(MOD, "kacis_0"))
    yol_csv = os.path.join(HERE, "kacis_sinav.csv")
    f = open(yol_csv, "w")
    f.write("adim,kacis,bosuna_kalkis,odul,gecti\n")
    s0 = sinav(model)
    f.write(f"0,{s0['kacis']},{s0['bosuna_kalkis']},{s0['odul']:.2f},{int(s0['gecti'])}\n"); f.flush()
    print("[kacis] adım 0:", {k: v for k, v in s0.items() if k != "ayrinti"}, flush=True)
    durum = dict(olcut=OLCUT, hareket_beyni=HAREKET, sinavlar=[], t0=time.time())

    class Kayit(BaseCallback):
        def __init__(self):
            super().__init__(); self.e = open(os.path.join(HERE, "kacis_egri.csv"), "w"); self.e.write("adim,bolum_odulu,uzunluk\n")

        def _on_step(self):
            for info in self.locals.get("infos", []):
                if "episode" in info:
                    self.e.write(f"{self.num_timesteps},{info['episode']['r']:.2f},{info['episode']['l']}\n"); self.e.flush()
            return True
    cb = Kayit()
    while model.num_timesteps < toplam:
        model.learn(total_timesteps=25_000, reset_num_timesteps=False, callback=cb)
        n = model.num_timesteps
        model.save(os.path.join(MOD, f"kacis_{n}"))
        s = sinav(model)
        f.write(f"{n},{s['kacis']},{s['bosuna_kalkis']},{s['odul']:.2f},{int(s['gecti'])}\n"); f.flush()
        durum["sinavlar"].append(dict(adim=n, **{k: v for k, v in s.items() if k != "ayrinti"}))
        print(f"[kacis] adım {n}:", {k: v for k, v in s.items() if k != "ayrinti"}, flush=True)
        if s["gecti"]:
            model.save(os.path.join(MOD, "beyin_G5_kacis_devresi"))
            durum.update(gecti=True, bitis=n, sure_dk=round((time.time() - durum["t0"]) / 60, 1))
            print(f"*** kacis GEÇTİ ({n} adım) ***", flush=True)
            break
    else:
        durum.update(gecti=False, bitis=model.num_timesteps)
        print("!!! kacis bütçe içinde GEÇEMEDİ", flush=True)
    durum["refleks"] = {k: v for k, v in sinav(None, refleks=True).items() if k != "ayrinti"}
    durum["devresiz_hareket_beyni"] = {k: v for k, v in s0.items() if k != "ayrinti"}
    json.dump(durum, open(os.path.join(HERE, "kacis_durum.json"), "w"), indent=1, ensure_ascii=False, default=str)
    venv.close()
    print("KACIS BITTI", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "egit":
        egit(int(sys.argv[2]))
    else:
        from stable_baselines3 import PPO
        s = sinav(PPO.load(sys.argv[2], device="cpu"))
        print({k: v for k, v in s.items() if k != "ayrinti"})
