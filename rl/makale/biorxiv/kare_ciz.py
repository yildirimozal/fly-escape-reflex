"""Şekil 1a için yazısız simülasyon kareleri: gerçek gölge yaklaşır, refleks (20° + 7 ms) kalkışı başlatır.
Kareler: gölge 10°, eşik anı (20°), kalkıştan 60 ms sonra. Açık zemin, sabit kamera.
  python kare_ciz.py   (rl_env)  -> kareler/kare_{1,2,3}.png, kareler/kare.json
"""
import json
import os
import sys

import mujoco as mj
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
RL = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, RL)
from kacis_devresi import KacisOrtam  # noqa: E402

GEN, YUK = 1200, 900
CIK = os.path.join(HERE, "kareler")
os.makedirs(CIK, exist_ok=True)


def acik_sahne(m, k=1.9):
    """Baskı için özgün koyu sahnenin açılmış hâli: gökyüzü ve dama zemin k kat aydınlatılır, renkler korunur."""
    for ad in ("skybox", "checker"):
        i = mj.mj_name2id(m, mj.mjtObj.mjOBJ_TEXTURE, ad)
        n = m.tex_width[i] * m.tex_height[i] * m.tex_nchannel[i]
        a = m.tex_adr[i]
        m.tex_data[a:a + n] = np.clip(m.tex_data[a:a + n].astype(np.float32) * k, 0, 255).astype(np.uint8)


torch.manual_seed(7)
ort = KacisOrtam(seed=7)
ort.egitim = False
ort.env.refleks = True
env = ort.env
env.sahne_zorla = dict(gorev="kacis", golgeler=[dict(t0=0.6, gercek=True, az=np.radians(150), el=np.radians(35), hiz=60.0)])
env.m.vis.global_.offwidth, env.m.vis.global_.offheight = GEN, YUK
acik_sahne(env.m)
rend = mj.Renderer(env.m, YUK, GEN)
o, _ = ort.reset(seed=7)
kam = mj.MjvCamera(); kam.type = mj.mjtCamera.mjCAMERA_FREE
kam.distance, kam.elevation, kam.azimuth = 27.0, -14.0, 75.0

hedefler = {"kare_1": None, "kare_2": None, "kare_3": None}
kayit = {}
t_kalk = None
while env.t < env.sure - 1e-9:
    oh = ort.o.copy(); oh[108:112] = 0.0
    a, _ = ort.hareket.predict(oh, deterministic=False)
    a = a.copy(); a[48] = -1.0
    ort.o, r, b, kes, info = env.step(a)
    aci = np.degrees(env._golge_gor()[0])
    sinek = env.d.xpos[env.tb].copy()
    ad = None
    if hedefler["kare_1"] is None and env.golge is not None and aci >= 10:
        ad = "kare_1"
    elif hedefler["kare_2"] is None and env.esik_t is not None:
        ad = "kare_2"
    if env.kalkti is not None and t_kalk is None:
        t_kalk = env.t
    if hedefler["kare_3"] is None and t_kalk is not None and env.t - t_kalk >= 0.06:
        ad = "kare_3"
    if ad:
        kam.lookat[:] = [sinek[0] - 2.5, sinek[1] + 1.5, 5.0]
        rend.update_scene(env.d, camera=kam)
        Image.fromarray(rend.render()).save(os.path.join(CIK, ad + ".png"))
        hedefler[ad] = True
        kayit[ad] = dict(t=round(env.t, 3), aci_derece=round(float(aci), 1), z_mm=round(float(sinek[2]), 2))
        print(ad, kayit[ad], flush=True)
    if b or kes or all(hedefler.values()):
        break
kayit.update(kalkti=env.kalkti, esik_t=env.esik_t, yakalandi=bool(env.yakalandi))
json.dump(kayit, open(os.path.join(CIK, "kare.json"), "w"), indent=1)
print(kayit)
