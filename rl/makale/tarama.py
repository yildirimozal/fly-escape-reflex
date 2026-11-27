"""Makale 1, ana deney: boşuna kalkışın bedeli (B) taranır. Her B için kapısız öğrenen devre (PPO, 3 tohum),
bağlantı şeması refleksi ve hiç kalkmayan sinek aynı karşılaşma dağılımında (yarısı gerçek gölge) ölçülür.
Bölüm = bir karşılaşma; kayıt: gerçek, kalktı, yakalandı, bölüm ödülü (uygunluk).
  python tarama.py ogren <tohum> <adim> <B>  -> tarama/ogren_b<B>_s<tohum>.csv
  python tarama.py refleks|hic <n> <B>       -> tarama/<ad>_b<B>.csv
"""
import csv
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from dogustan import Cevre  # noqa: E402

CIKTI = os.path.join(HERE, "tarama")


class Ortam(Cevre):
    def __init__(self, bedel, seed=None, kapi=False):
        super().__init__(0.5, seed=seed)
        self.env.kalkis_bedeli = float(bedel)
        self.kapi = kapi

    def step(self, k):
        o, r, b, kes, info = super().step(k)
        info["gercek"] = bool(self.env.golgeler[0]["gercek"]) if self.env.golgeler else None
        return o, r, b, kes, info


def yaz(yol, satir):
    with open(yol, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["bolum", "adim", "gercek", "kalkti", "yakalandi", "odul"]); w.writerows(satir)


def ogren(tohum, toplam, bedel):
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
    torch.set_num_threads(1)
    venv = VecMonitor(SubprocVecEnv([(lambda i=i: Ortam(bedel, seed=1000 * tohum + i)) for i in range(4)]))
    model = PPO("MlpPolicy", venv, learning_rate=3e-4, n_steps=512, batch_size=512, n_epochs=10, gamma=0.99,
                gae_lambda=0.95, ent_coef=0.05, policy_kwargs=dict(net_arch=[64, 64]), seed=tohum, device="cpu", verbose=0)
    satir = []

    class K(BaseCallback):
        def _on_step(self):
            for info, bitti in zip(self.locals["infos"], self.locals["dones"]):
                if bitti:
                    satir.append([len(satir), self.num_timesteps, int(bool(info.get("gercek"))), int(info.get("kalkti") is not None),
                                  int(bool(info.get("yakalandi"))), round(float(info["episode"]["r"]), 2)])
            return True
    model.learn(total_timesteps=toplam, callback=K())
    model.save(os.path.join(CIKTI, f"ogren_b{bedel}_s{tohum}"))
    yaz(os.path.join(CIKTI, f"ogren_b{bedel}_s{tohum}.csv"), satir)
    venv.close()
    print("ogren", bedel, tohum, "bolum", len(satir), "yakalanma", sum(r[4] for r in satir), flush=True)


def sabit(ad, n, bedel):
    ortam = Ortam(bedel, seed=4242)
    ortam.env.refleks = ad == "refleks"
    satir = []
    while len(satir) < n:
        torch.manual_seed(len(satir))
        o, _ = ortam.reset(seed=4242 + len(satir)); R = 0.0
        while True:
            o, r, b, kes, info = ortam.step(0); R += r
            if b or kes:
                break
        satir.append([len(satir), 0, int(bool(info["gercek"])), int(info["kalkti"] is not None), int(bool(info["yakalandi"])), round(R, 2)])
    yaz(os.path.join(CIKTI, f"{ad}_b{bedel}.csv"), satir)
    print(ad, bedel, "bolum", n, "yakalanma", sum(r[4] for r in satir), flush=True)


if __name__ == "__main__":
    os.makedirs(CIKTI, exist_ok=True)
    if sys.argv[1] == "ogren":
        ogren(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    else:
        sabit(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
