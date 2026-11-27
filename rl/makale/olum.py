"""Ölüm bedeli düzeltmesi: öğrenen devrenin ödülünde yakalanma, gelecek ömrün kaybı kadar pahalı.

Eski ödülde (tarama.py) yakalanma -10 ve bölümün kalanının kaybıydı (toplam en çok yaklaşık 100 birim). Bu, bir boşuna
kalkışın bedeliyle (B = 10-100) aynı düzeydeydi; öğrenen devre bir ölümü bir öğünle takas ediyordu. Gerçekte ölüm,
kalan ömürdeki bütün öğünleri götürür.
Burada: gerçek gölgede yakalanma = -K * V   (V = 168: bir karşılaşmalık öğünün değeri, omur.py ile aynı birim;
K = kalan ömür, karşılaşma sayısı; ana koşu K = 100, omur.py ve aclik.py'deki T = 100 ile aynı).
Sahte gölgeye temas (simülasyon yan etkisi, omur.py) eski -10'da kalır. Bütün ödüller V'ye bölünür (birim: öğün);
bu, en iyi politikayı değiştirmez, PPO'da değer kaybının politika gradyanını kırpmada bastırmasını önler.
  python olum.py ogren <tohum> <adim> <B> <K>  -> tarama/olum_k<K>_b<B>_s<tohum>.csv (tarama.py ile aynı sütunlar)
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from tarama import CIKTI, Ortam, yaz  # noqa: E402

V = 168.2


class OlumOrtam(Ortam):
    def __init__(self, bedel, K, seed=None):
        super().__init__(bedel, seed=seed)
        self.K = float(K)

    def step(self, k):
        o, r, b, kes, info = super().step(k)
        if b and info.get("yakalandi") and info.get("gercek"):
            r += 10.0 - self.K * V          # dunya.py'nin -10'u yerine gelecek ömrün kaybı
        return o, r / V, b, kes, info


def ogren(tohum, toplam, bedel, K):
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
    torch.set_num_threads(1)
    venv = VecMonitor(SubprocVecEnv([(lambda i=i: OlumOrtam(bedel, K, seed=1000 * tohum + i)) for i in range(4)]))
    model = PPO("MlpPolicy", venv, learning_rate=3e-4, n_steps=512, batch_size=512, n_epochs=10, gamma=0.99,
                gae_lambda=0.95, ent_coef=0.05, policy_kwargs=dict(net_arch=[64, 64]), seed=tohum, device="cpu", verbose=0)
    satir = []

    class Kb(BaseCallback):
        def _on_step(self):
            for info, bitti in zip(self.locals["infos"], self.locals["dones"]):
                if bitti:
                    satir.append([len(satir), self.num_timesteps, int(bool(info.get("gercek"))), int(info.get("kalkti") is not None),
                                  int(bool(info.get("yakalandi"))), round(float(info["episode"]["r"]), 4)])
            return True
    model.learn(total_timesteps=toplam, callback=Kb())
    ad = f"olum_k{K}_b{bedel}_s{tohum}"
    model.save(os.path.join(CIKTI, ad))
    yaz(os.path.join(CIKTI, ad + ".csv"), satir)
    venv.close()
    print("olum", K, bedel, tohum, "bolum", len(satir), "yakalanma", sum(r[4] for r in satir), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "ogren":
        ogren(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
