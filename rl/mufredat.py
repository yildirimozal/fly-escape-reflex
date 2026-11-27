"""Yeni doğmuş sineğin müfredatı: tek beyin (SAC politikası), görevler sırayla, geçiş testiyle.

Her aşamada yeni görev %50, önceki görevler kalan %50 (ilk aşamalar %60/%40 ile eğitildi) (unutmasın diye tekrar). Ödül terimleri birikir.
Aşama her 25 bin adımda 10 sabit sahnede (deterministik) sınanır; ölçütü geçince sonraki aşamaya geçilir.
Bütçe biterse müfredat DURUR ve bu raporlanır (geçmiş gibi gösterilmez).

  python mufredat.py           -> modeller/, egri.csv, sinav.csv, asamalar.json (kaldığı yerden sürer)
"""
import csv
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dunya import GOREVLER, Sinek  # noqa: E402

from stable_baselines3 import SAC  # noqa: E402
from stable_baselines3.common.buffers import ReplayBuffer  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor  # noqa: E402

MOD = os.path.join(HERE, "modeller")
N_ENV = 8
PARCA = 25_000
TERIM = {"ayakta": ("durus",), "yuru": ("durus", "yuru"), "koku": ("durus", "yuru", "koku"),
         "tat": ("durus", "yuru", "koku"), "kacis": ("durus", "yuru", "koku", "kacis")}
BUTCE = {"ayakta": 400_000, "yuru": 2_500_000, "koku": 1_500_000, "tat": 1_000_000, "kacis": 1_000_000}
# Geçiş ölçütleri (eğitimden önce sabitlendi)
OLCUT = {
    "ayakta": "10 doğumun en az 9'unda 2. saniyede dik (>0,9) ve gövde yüksekliği > 0,95 mm",
    "yuru": ("10 denemenin en az 8'inde ortalama ileri hız >= 5 mm/s; adımların >= %80'inde en az 3 ayak yerde, "
             ">= %90'ında gövde dik (>0,8) ve 1,5 mm'den alçakta (2. denemede sertleştirildi: ilk ölçüt zıplayan takla "
             "hilesini geçiriyordu)"),
    "koku": "10 denemenin en az 7'sinde 5 s içinde tatlı damlaya ulaşma",
    "tat": "10 denemenin en az 8'inde ilk değilen damla tatlı",
    "kacis": "10 gerçek gölgenin en az 9'undan kaçış ve 10 sahte gölgede en çok 3 boşuna kalkış",
}


def ortam_yap(gorev_sira, k, seed):
    yeni = gorev_sira[k]
    gorevler = gorev_sira[:k + 1]
    agirlik = [0.5 / k] * k + [0.5] if k > 0 else [1.0]     # tat 1. denemesinden sonra tekrar payı %40 -> %50

    def f():
        return Sinek(gorevler=gorevler, agirlik=agirlik, terimler=TERIM[yeni], seed=seed)
    return f


DETERMINISTIK = False   # 4. aşamadan itibaren sınav, eğitilen (gürültülü) politikayla; tohum sabit. 1-3 deterministik geçti.


def bolum(env, model, sahne, seed, refleks=False, deterministik=None):
    det = DETERMINISTIK if deterministik is None else deterministik
    env.sahne_zorla = sahne
    env.refleks = refleks
    torch.manual_seed(seed)
    o, _ = env.reset(seed=seed)
    hizlar, info, R = [], {}, 0.0
    duzgun = []
    while True:
        a, _ = model.predict(o, deterministic=det) if model is not None else (np.zeros(49), None)
        o, r, b, k, info = env.step(a)
        R += r
        hizlar.append(info["ileri_hiz"])
        duzgun.append((info["temas"] >= 3, info["dik"] > 0.8, info["z"] < 1.5))
        if b or k:
            break
    info["duzgun"] = np.mean(duzgun, axis=0)       # oranlar: >=3 ayak yerde, dik, havada değil
    return info, float(np.mean(hizlar)), R


def sinav(model, gorev, terimler, refleks=False):
    env = Sinek(gorevler=[gorev], terimler=terimler)
    sonuc = []
    if gorev == "kacis":
        for i in range(20):
            gercek = i < 10
            sahne = dict(gorev="kacis", golgeler=[dict(t0=0.4 + 0.15 * (i % 10), gercek=gercek)])
            info, _, R = bolum(env, model, sahne, 1000 + i, refleks)
            sonuc.append(dict(gercek=gercek, kalkti=info["kalkti"] is not None, yakalandi=info["yakalandi"], R=R))
        g = [s for s in sonuc if s["gercek"]]; f = [s for s in sonuc if not s["gercek"]]
        kac = sum(not s["yakalandi"] for s in g); bos = sum(s["kalkti"] for s in f)
        return dict(gecti=kac >= 9 and bos <= 3, kacis=kac, bosuna_kalkis=bos,
                    odul=np.mean([s["R"] for s in sonuc]))
    for i in range(10):
        info, v, R = bolum(env, model, dict(gorev=gorev), 1000 + i)
        if gorev == "ayakta":
            ok = info["dik"] > 0.9 and info["z"] > 0.95
        elif gorev == "yuru":
            # 2. denemede sertleştirildi: zıplayan/takla atan "yürüyüş" geçemez
            ok = (v >= 5.0 and info["t"] >= 1.99 and info["duzgun"][0] >= 0.8 and info["duzgun"][1] >= 0.9
                  and info["duzgun"][2] >= 0.9)
        elif gorev == "koku":
            ok = info["tatli"]
        elif gorev == "tat":
            ok = info["ilk_tat"] == "seker"
        sonuc.append(dict(ok=ok, v=v, tat=info["ilk_tat"], R=R, z=info["z"], dik=info["dik"]))
    n = sum(s["ok"] for s in sonuc)
    esik = {"ayakta": 9, "yuru": 8, "koku": 7, "tat": 8}[gorev]
    out = dict(gecti=n >= esik, basari=n, odul=float(np.mean([s["R"] for s in sonuc])),
               hiz=float(np.mean([s["v"] for s in sonuc])))
    if gorev == "tat":
        out["ilk_tat"] = [s["tat"] for s in sonuc]
    return out


class Kayit(BaseCallback):
    def __init__(self, asama, yol):
        super().__init__()
        self.asama, self.t0 = asama, time.time()
        yeni = not os.path.exists(yol)
        self.f = open(yol, "a", newline="")
        self.w = csv.writer(self.f)
        if yeni:
            self.w.writerow(["adim", "asama", "gorev", "bolum_odulu", "bolum_uzunlugu", "sure_sn"])

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.w.writerow([self.num_timesteps, self.asama, info.get("gorev"), round(info["episode"]["r"], 2),
                                 info["episode"]["l"], round(time.time() - self.t0)])
                self.f.flush()
        return True


def main():
    os.makedirs(MOD, exist_ok=True)
    torch.set_num_threads(2)
    dyol = os.path.join(HERE, "asamalar.json")
    durum = json.load(open(dyol)) if os.path.exists(dyol) else dict(asamalar=[], k=0, model=None, olcut=OLCUT)
    k = durum["k"]
    model = None
    while k < len(GOREVLER):
        g = GOREVLER[k]
        venv = VecMonitor(SubprocVecEnv([ortam_yap(GOREVLER, k, 100 * k + i) for i in range(N_ENV)]),
                          info_keywords=("gorev",))
        if model is None and durum["model"]:
            model = SAC.load(durum["model"], env=venv, device="cpu")
        if model is None:
            model = SAC("MlpPolicy", venv, learning_rate=3e-4, buffer_size=500_000, batch_size=256, gamma=0.99,
                        tau=0.005, train_freq=1, gradient_steps=4, learning_starts=10_000, ent_coef="auto_0.1",
                        policy_kwargs=dict(net_arch=[256, 256]), seed=1, device="cpu", verbose=0)
            model.save(os.path.join(MOD, "beyin_0"))     # yeni doğmuş: eğitilmemiş beyin
        else:
            model.set_env(venv)
            model.replay_buffer = ReplayBuffer(model.buffer_size, venv.observation_space, venv.action_space,
                                               device="cpu", n_envs=N_ENV)
            model.learning_starts = model.num_timesteps + 2_000
        bas = model.num_timesteps
        bilgi = next((a for a in durum["asamalar"] if a["gorev"] == g), None)
        if bilgi is None:
            bilgi = dict(gorev=g, baslangic=bas, olcut=OLCUT[g], sinavlar=[], t0=time.time())
            durum["asamalar"].append(bilgi)
        cb = Kayit(k + 1, os.path.join(HERE, "egri.csv"))
        while True:
            model.learn(total_timesteps=PARCA, reset_num_timesteps=False, callback=cb, progress_bar=False)
            n = model.num_timesteps
            yol = os.path.join(MOD, f"beyin_{n}")
            model.save(yol)
            s = sinav(model, g, TERIM[g])
            onceki = {h: sinav(model, h, TERIM[g])["basari"] for h in GOREVLER[:k] if h != "kacis"}
            s.update(adim=n, asama_adimi=n - bilgi["baslangic"], sure_dk=round((time.time() - bilgi["t0"]) / 60, 1),
                     onceki=onceki)
            bilgi["sinavlar"].append({a: (b.item() if isinstance(b, np.generic) else b) for a, b in s.items()})
            durum["model"] = yol
            print(f"[{g}] adım {n} (aşamada {n - bilgi['baslangic']}): {s}", flush=True)
            with open(os.path.join(HERE, "sinav.csv"), "a") as f:
                f.write(f"{n},{k + 1},{g},{int(s['gecti'])},{s.get('basari', s.get('kacis'))},{s['odul']:.2f},"
                        f"{s.get('hiz', 0):.2f},{s.get('bosuna_kalkis', '')}\n")
            json.dump(durum, open(dyol, "w"), indent=1, ensure_ascii=False, default=str)
            if s["gecti"]:
                bilgi.update(gecti=True, bitis=n, sure_dk=s["sure_dk"])
                model.save(os.path.join(MOD, f"beyin_G{k + 1}_{g}"))
                if g == "kacis":
                    bilgi["refleks"] = sinav(None, "kacis", TERIM[g], refleks=True)
                k += 1
                durum["k"] = k
                json.dump(durum, open(dyol, "w"), indent=1, ensure_ascii=False, default=str)
                print(f"*** {g} GEÇTİ ({n - bilgi['baslangic']} adım) ***", flush=True)
                break
            if n - bilgi["baslangic"] >= BUTCE[g]:
                bilgi.update(gecti=False, bitis=n)
                json.dump(durum, open(dyol, "w"), indent=1, ensure_ascii=False, default=str)
                print(f"!!! {g} bütçe içinde GEÇEMEDİ, müfredat durdu", flush=True)
                venv.close()
                return
        venv.close()
    print("MÜFREDAT TAMAM", flush=True)


if __name__ == "__main__":
    main()
