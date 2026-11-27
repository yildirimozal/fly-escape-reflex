"""Yeni doğmuş sanal sinek: tüm görevlerin paylaştığı tek dünya (NeuroMechFly v2 gövdesi, MuJoCo).

Tek beyin (öğrenen politika), tek duyu seti, tek eylem seti. Görevler yalnızca sahnenin nasıl kurulduğuyla
ve ödülde hangi terimlerin açık olduğuyla ayrılır:

  ayakta : yan yatmış / bacakları karışık doğar, doğrulup ayakta durmalı
  yuru   : ayakta başlar, gövdesinin baktığı yöne yürümeli (koku yokken)
  koku   : 6–12 mm uzakta tek tatlı damla, kokusunu iki anteniyle izleyip ulaşmalı
  tat    : üç damla (tatlı, acı, tatlı+acı). Ödülü BEYİN MODELİ verir: değdiği damlanın tadında
           Shiu vd. 2024 modelinin MN9 (hortum motor nöronu) hızı. Tatlı 102,8 Hz -> 1, karışık 1,1 Hz, acı 0.
  kacis  : yiyecek ararken gölge gelir (yarısı gerçekten üstüne, yarısı yanından geçer). "Kalk" eylemi
           doğuştan kalkış programını başlatır (orta bacak itişi, kanat, basit itki; aerodinamik yok).
  gosteri: hepsi bir arada.

Duyular (112): 42 bacak eklem açısı + 42 hız, gövdeye göre yerçekimi, gövde hızı, açısal hız, yükseklik,
6 ayak teması, VNC ritmi (sin/cos, 10 Hz, doğuştan verildi), 2 koku kanalı × (ortalama, sol-sağ farkı),
2 tat (tatlı, acı; ayakla), gölge (gözdeki açı, büyüme hızı, yön sin/cos).
Eylem (49): 42 bacak eklemine nötr duruştan sapma, 6 ayak yapışması, 1 kalkış komutu.
"""
import json
import os

import gymnasium as gym
import mujoco as mj
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SN = os.path.dirname(HERE)
OZ = json.load(open(os.path.join(SN, "sonuclar", "ozet.json")))
MN9 = {k: float(sum(OZ[k]["mn9"])) for k in ("seker", "aci", "seker_aci")}   # beyin modelinin ölçümü (Hz)
MN9_NORM = {k: v / MN9["seker"] for k, v in MN9.items()}
TATLAR = {"seker": (1.0, 0.0), "aci": (0.0, 1.0), "seker_aci": (1.0, 1.0)}     # ayaktaki tat reseptörleri
KOKU = {"seker": (1.0, 0.0), "aci": (0.0, 1.0), "seker_aci": (0.5, 0.5)}       # kanal A meyve, kanal B öteki
RENK = {"seker": (1.0, 0.68, 0.2, 0.8), "aci": (0.85, 0.2, 0.25, 0.8), "seker_aci": (0.75, 0.45, 0.85, 0.8)}

DT_FIZ = 2e-4
DT_KON = 0.01
N_ALT = int(round(DT_KON / DT_FIZ))
DAMLA_R = 0.6
GOLGE_R = 1.6
LAMBDA = 8.0            # koku azalma uzunluğu, mm
ESIK = np.radians(20.0)  # bağlantı şeması refleksi: gözdeki açı eşiği
GF_GECIKME = 0.007       # LPLC2 -> dev lif ilk ateşleme (beyin modeli)
GOREVLER = ["ayakta", "yuru", "koku", "tat", "kacis"]


def model_kur():
    sp = mj.MjSpec.from_file(os.path.join(HERE, "model", "sinek.xml"))
    for i in range(3):
        b = sp.worldbody.add_body(name=f"damla{i}", mocap=True, pos=[0, 0, -50])
        b.add_geom(type=mj.mjtGeom.mjGEOM_SPHERE, size=[DAMLA_R, 0, 0], rgba=[1, 1, 1, 0.8], contype=0, conaffinity=0)
    b = sp.worldbody.add_body(name="golge", mocap=True, pos=[0, 0, -80])
    b.add_geom(type=mj.mjtGeom.mjGEOM_SPHERE, size=[GOLGE_R, 0, 0], rgba=[0.32, 0.32, 0.38, 1], contype=0, conaffinity=0)   # yalnız görüntü (koyu gökyüzünde görünsün)
    m = sp.compile()
    m.opt.timestep = DT_FIZ
    return m


class Sinek(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, gorevler=("ayakta",), agirlik=None, terimler=("durus",), seed=None):
        self.m = m = model_kur()
        self.d = mj.MjData(m)
        self.gorevler = list(gorevler)
        self.agirlik = None if agirlik is None else np.array(agirlik, float) / np.sum(agirlik)
        self.terimler = set(terimler)
        self.rng = np.random.default_rng(seed)
        ad = [m.actuator(i).name for i in range(m.nu)]
        self.bacak = np.array(sorted(i for i, a in enumerate(ad) if "position" in a and (
            a.split("/")[1][:2] in ("lf", "lm", "lh", "rf", "rm", "rh") and "tarsus" not in a.split("-")[0]
            or ("c_thorax-" in a and "coxa" in a))))
        self.yapis = np.array([i for i, a in enumerate(ad) if "adhesion" in a])
        self.notr = m.key_ctrl[0].copy()
        self.olcek = np.array([0.9 if "pitch" in ad[i] else 0.4 for i in self.bacak])
        jid = m.actuator_trnid[self.bacak, 0]
        self.qa, self.va = m.jnt_qposadr[jid], m.jnt_dofadr[jid]
        self.q0 = m.key_qpos[0][self.qa].copy()
        bid = lambda n: mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, n)
        self.tb, self.hd = bid("nmf/c_thorax"), bid("nmf/c_head")
        self.ant = [bid("nmf/l_arista"), bid("nmf/r_arista")]
        self.ayak = [bid(f"nmf/{l}_tarsus5") for l in ("lf", "lm", "lh", "rf", "rm", "rh")]
        self.zemin = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, "ground_plane")
        self.ayak_geom = {}
        for gi in range(m.ngeom):
            bn = m.body(m.geom_bodyid[gi]).name
            for j, l in enumerate(("lf", "lm", "lh", "rf", "rm", "rh")):
                if bn.startswith(f"nmf/{l}_tarsus"):
                    self.ayak_geom[gi] = j
        self.damla_mid = [m.body_mocapid[bid(f"damla{i}")] for i in range(3)]
        self.damla_gid = [m.body_geomadr[bid(f"damla{i}")] for i in range(3)]
        self.golge_mid = m.body_mocapid[bid("golge")]
        self.M = sum(m.body_mass[i] for i in range(m.nbody) if m.body(i).name.startswith("nmf/"))
        self.G = -m.opt.gravity[2]
        self.i_ir = ad.index("nmf/c_head-c_rostrum-pitch-position")
        self.i_ih = ad.index("nmf/c_rostrum-c_haustellum-pitch-position")
        self.kanat = {s: [ad.index(f"nmf/c_thorax-{s}_wing-{a}-position") for a in ("yaw", "pitch", "roll")] for s in ("l", "r")}
        self.itme = {}
        for leg in ("lm", "rm"):
            self.itme[ad.index(f"nmf/{leg}_coxa-{leg}_trochanterfemur-pitch-position")] = -1.0
            self.itme[ad.index(f"nmf/{leg}_trochanterfemur-{leg}_tibia-pitch-position")] = 0.6
        # gösteri için uçuş duruşu: itişten sonra bacaklar toplanır (yalnız gorev == "gosteri"; deneyleri etkilemez)
        self.katla = []
        for leg in ("lf", "lm", "lh", "rf", "rm", "rh"):
            self.katla.append((ad.index(f"nmf/{leg}_coxa-{leg}_trochanterfemur-pitch-position"), -1.0))
            self.katla.append((ad.index(f"nmf/{leg}_trochanterfemur-{leg}_tibia-pitch-position"), 1.8))
        self.observation_space = gym.spaces.Box(-10, 10, (112,), np.float32)
        self.action_space = gym.spaces.Box(-1, 1, (49,), np.float32)
        self.kalkis_bedeli = 10.0   # tehditsiz kalkışın toplam bedeli (makale: bedel taraması)
        self.esik = ESIK            # refleks eşiği (makale: eşik taraması için ayarlanabilir)
        self.refleks = False        # True: kalkışı politika değil 20° + 7 ms kuralı verir (karşılaştırma)
        self.sahne_zorla = None     # gösteri / değerlendirme için sabit sahne

    # ---------- sahne ----------
    def _ayak_tut(self, rot=None, z=None):
        d = self.d
        mj.mj_resetDataKeyframe(self.m, d, 0)
        yaw = self.rng.uniform(-np.pi, np.pi)
        q = np.zeros(4); mj.mju_axisAngle2Quat(q, np.array([0, 0, 1.0]), yaw)
        if rot is not None:
            q2 = np.zeros(4); mj.mju_axisAngle2Quat(q2, np.array([1.0, 0, 0]), rot)
            q3 = np.zeros(4); mj.mju_mulQuat(q3, q, q2); q = q3
        d.qpos[3:7] = q
        if z is not None:
            d.qpos[2] = z
        return yaw

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        r = self.rng
        s = self.sahne_zorla or {}
        g = s.get("gorev") or (self.gorevler[r.choice(len(self.gorevler), p=self.agirlik)])
        self.gorev = g
        m, d = self.m, self.d
        if g == "ayakta" or s.get("dogum"):
            rot = r.choice([-1, 1]) * r.uniform(0.3, 1.6)
            self._ayak_tut(rot=rot, z=1.8)
            d.qpos[self.qa] = self.q0 + r.uniform(-0.6, 0.6, len(self.qa)) * self.olcek
        else:
            self._ayak_tut()
        mj.mj_forward(m, d)
        self.sure = {"ayakta": 2.0, "yuru": 2.0, "koku": 5.0, "tat": 5.0, "kacis": 4.0, "gosteri": 9.0}[g]
        if "sure" in s:
            self.sure = s["sure"]
        # damlalar
        self.damlalar = []
        for i in range(3):
            d.mocap_pos[self.damla_mid[i]] = (0, 0, -50)
        govde = d.xpos[self.tb][:2].copy()
        if g in ("koku", "kacis"):
            tipler = ["seker"]
        elif g in ("tat", "gosteri"):
            tipler = list(r.permutation(["seker", "aci", "seker_aci"]))
        else:
            tipler = []
        if "damlalar" in s:
            konumlar = [np.array(p, float) for _, p in s["damlalar"]]; tipler = [t for t, _ in s["damlalar"]]
        else:
            konumlar, a0 = [], r.uniform(-np.pi, np.pi)
            for j, t in enumerate(tipler):
                a = a0 + j * 2 * np.pi / max(1, len(tipler)) + r.uniform(-0.4, 0.4)
                konumlar.append(govde + r.uniform(6, 12) * np.array([np.cos(a), np.sin(a)]))
        for j, (t, p) in enumerate(zip(tipler, konumlar)):
            d.mocap_pos[self.damla_mid[j]] = (p[0], p[1], 0.05)
            m.geom_rgba[self.damla_gid[j]] = RENK[t]
            self.damlalar.append((t, p))
        # gölge
        self.golge = None
        d.mocap_pos[self.golge_mid] = (0, 0, -80)
        if g in ("kacis", "gosteri") or "golgeler" in s:
            liste = s.get("golgeler")
            if liste is None:
                gercek = r.random() < 0.5
                liste = [dict(t0=r.uniform(0.3, 2.5), gercek=gercek)]
            self.golgeler = []
            for gl in liste:
                az = gl.get("az", r.uniform(-np.pi, np.pi)); el = gl.get("el", np.radians(r.uniform(15, 45)))
                hiz = gl.get("hiz", r.uniform(50, 90))
                ofs = 0.0 if gl["gercek"] else gl.get("ofs", r.uniform(4, 8))
                self.golgeler.append(dict(t0=gl["t0"], gercek=gl["gercek"], az=az, el=el, hiz=hiz, ofs=ofs,
                                          ofs_yon=r.uniform(0, 2 * np.pi)))
        else:
            self.golgeler = []
        self.t = 0.0
        self.kalkti = None
        self.tehdit = False
        self.durus_son = 0.0
        self.yakalandi = False
        self.kacti = False
        self.yedi = None
        self.yeme_t = None
        self.ilk_tat = None
        self.tatli = False
        self.aci_once = 0.0
        self.aci_hiz = 0.0
        self.esik_t = None
        self.hedefler = list(self.damlalar)
        self.d_onceki = self._hedef()[1]
        self.ilk_x = d.xpos[self.tb][:2].copy()
        self.yol = 0.0
        return self._gozlem(), {}

    # ---------- duyular ----------
    def _hedef(self):
        """Keşif yönlendirmesi: henüz tadına bakılmamış en yakın damla (tatlı/acı ayrımı yapmaz)."""
        p = self.d.xpos[self.hd][:2]
        if not self.hedefler:
            return None, 0.0
        q = min((q for _, q in self.hedefler), key=lambda q: np.linalg.norm(p - q))
        return q, float(np.linalg.norm(p - q))

    def _en_yakin(self):
        p = self.d.xpos[self.hd][:2]
        return min((np.linalg.norm(p - q) for _, q in self.damlalar), default=0.0)

    def _koku(self):
        out = []
        for k in range(2):
            c = []
            for a in self.ant:
                pa = self.d.xpos[a][:2]
                c.append(sum(KOKU[t][k] * np.exp(-np.linalg.norm(pa - q) / LAMBDA) for t, q in self.damlalar) + 1e-6)
            cm = 0.5 * (c[0] + c[1])
            out += [np.clip(np.log(cm) / 3 + 1, -1.5, 1.5), np.clip((c[0] - c[1]) / cm * 30, -3, 3)]
        return out

    def _tat(self):
        tat = np.zeros(2); deg = None
        for t, q in self.damlalar:
            for b in self.ayak:
                if np.linalg.norm(self.d.xpos[b][:2] - q) < DAMLA_R + 0.15 and self.d.xpos[b][2] < 1.0:
                    tat = np.maximum(tat, TATLAR[t]); deg = t
            if np.linalg.norm(self.d.xpos[self.hd][:2] - q) < DAMLA_R + 0.2:
                tat = np.maximum(tat, TATLAR[t]); deg = t
        return tat, deg

    def _golge_konum(self, gl, t):
        yon = np.array([np.cos(gl["el"]) * np.cos(gl["az"]), np.cos(gl["el"]) * np.sin(gl["az"]), np.sin(gl["el"])])
        dik = np.cross(yon, [0, 0, 1.0]); dik /= np.linalg.norm(dik); dik2 = np.cross(yon, dik)
        hedef = gl["hedef"] + gl["ofs"] * (np.cos(gl["ofs_yon"]) * dik + np.sin(gl["ofs_yon"]) * dik2)
        return hedef + yon * (28.0 - gl["hiz"] * (t - gl["t0"]))

    def _golge_guncelle(self):
        d = self.d
        aktif = None
        for gl in self.golgeler:
            if self.t >= gl["t0"] and self.t <= gl["t0"] + 56.0 / gl["hiz"]:
                if "hedef" not in gl:
                    gl["hedef"] = d.xpos[self.hd].copy()
                aktif = gl
        self.golge = aktif
        d.mocap_pos[self.golge_mid] = self._golge_konum(aktif, self.t) if aktif else (0, 0, -80)

    def _golge_gor(self):
        if self.golge is None:
            return 0.0, np.zeros(4)
        d = self.d
        v = d.mocap_pos[self.golge_mid] - d.xpos[self.hd]
        uz = np.linalg.norm(v)
        aci = 2 * np.arctan(GOLGE_R / max(uz, GOLGE_R + 1e-3))
        R = d.xmat[self.tb].reshape(3, 3)
        vb = R.T @ v
        az = np.arctan2(vb[1], vb[0])
        hiz = self.aci_hiz      # düzeltme: önceden aci_once güncellendikten sonra hesaplanıp hep 0 çıkıyordu
        return aci, np.array([aci * 2, np.clip(hiz / 10, -3, 3), np.sin(az), np.cos(az)])

    def _gozlem(self):
        d = self.d
        R = d.xmat[self.tb].reshape(3, 3)
        q = (d.qpos[self.qa] - self.q0)
        dq = np.clip(d.qvel[self.va] * 0.02, -5, 5)
        g = R.T @ np.array([0, 0, -1.0])
        v = np.clip(R.T @ d.qvel[0:3] / 20, -5, 5)
        w = np.clip(d.qvel[3:6] / 20, -5, 5)
        z = [d.xpos[self.tb][2] - 1.1]
        temas = self._temas()
        ritim = [np.sin(2 * np.pi * 10 * self.t), np.cos(2 * np.pi * 10 * self.t)]
        tat, _ = self._tat()
        _, gg = self._golge_gor()
        o = np.concatenate([q, dq, g, v, w, z, temas, ritim, self._koku(), tat, gg]).astype(np.float32)
        return np.clip(o, -10, 10)

    def _temas(self):
        temas = np.zeros(6)
        for c in self.d.contact[:self.d.ncon]:
            for g1, g2 in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if g1 == self.zemin and g2 in self.ayak_geom:
                    temas[self.ayak_geom[g2]] = 1.0
        return temas

    # ---------- adım ----------
    def _kalkis_alt(self, u, t_k):
        """Doğuştan kalkış programı (klipler.py ile aynı): t_k kalkıştan beri geçen süre."""
        d = self.d
        if t_k < 0.02:
            for i, v in self.itme.items():
                u[i] = v
        elif self.gorev == "gosteri" and t_k > 0.03:
            for i, ofs in self.katla:
                u[i] = self.notr[i] + ofs
        ph = 2 * np.pi * 60 * t_k
        for s in ("l", "r"):
            u[self.kanat[s][0]] = 1.3 + 0.9 * np.sin(ph)
            u[self.kanat[s][2]] = 0.6 * np.cos(ph)
        u[self.yapis] = 0
        d.xfrc_applied[:] = 0
        if t_k > 0.012:
            M, G, tb = self.M, self.G, self.tb
            up = d.xmat[tb].reshape(3, 3)[:, 2]
            v = d.cvel[tb][3:]; w_ = d.cvel[tb][:3]
            f = np.array([0, 0, 1.10 * M * G]) + 0.35 * M * G * np.append(self.kacis_yon, 0)
            d.xfrc_applied[tb, :3] = f - (0.30 * M * G / 150.0) * v
            d.xfrc_applied[tb, 3:] = 0.02 * M * G * np.cross(up, [0, 0, 1]) - 0.0008 * M * G * w_
        return u

    def step(self, a):
        m, d = self.m, self.d
        a = np.clip(np.asarray(a, float), -1, 1)
        u = self.notr.copy()
        u[self.bacak] += self.olcek * a[:42]
        u[self.yapis] = (a[42:48] > 0).astype(float)
        aci, _ = self._golge_gor()
        if self.kalkti is None and self.golge is not None and aci >= self.esik and self.esik_t is None:
            self.esik_t = self.t
        if self.refleks:
            istek = self.esik_t is not None and self.t >= self.esik_t + GF_GECIKME
        else:   # uçma yalnız kaçış öğrenilirken açılır (öncesinde kanat kasları "henüz yok")
            istek = a[48] > 0.5 and "kacis" in self.terimler
        if self.kalkti is None and istek:
            self.kalkti = self.t
            self.yedi = None            # (yalnız gösteride dolu olur) beslenmeyi bırakıp kalkar
            self.tehdit = self.golge is not None and self.golge["gercek"]   # kalkış anında gerçek gölge var mıydı
            if self.golge is not None:
                v = d.xpos[self.hd][:2] - d.mocap_pos[self.golge_mid][:2]
            else:
                v = d.xmat[self.tb].reshape(3, 3)[:2, 0]
            self.kacis_yon = v / (np.linalg.norm(v) + 1e-9)
        if self.yedi is not None:          # damlada: hortum beyin modelinin MN9 iziyle (gösteri)
            u = self.notr.copy(); u[self.yapis] = 1
            e = min(1.0, MN9[self.yedi] / 90.0) * min(1.0, (self.t - self.yeme_t) / 0.15)
            u[self.i_ir] -= 1.2 * e; u[self.i_ih] += 1.0 * e
        bozuk = False
        alt = 5 if self.kalkti is not None else N_ALT
        for _ in range(N_ALT // alt):
            if self.kalkti is not None:
                u = self._kalkis_alt(u, self.t - self.kalkti)
            d.ctrl[:] = u
            mj.mj_step(m, d, nstep=alt)
            self.t += alt * DT_FIZ
            self._golge_guncelle()
            if self.golge is not None:
                if np.linalg.norm(d.mocap_pos[self.golge_mid] - d.xpos[self.tb]) < GOLGE_R + 0.7:
                    self.yakalandi = True
            if d.warning[mj.mjtWarning.mjWARN_BADQACC].number:
                d.warning[mj.mjtWarning.mjWARN_BADQACC].number = 0
                bozuk = True
                break
        aci2, _ = self._golge_gor()
        R = d.xmat[self.tb].reshape(3, 3)
        dik = R[2, 2]
        z = d.xpos[self.tb][2]
        ileri = R[:2, 0] / (np.linalg.norm(R[:2, 0]) + 1e-9)
        hiz = d.qvel[0:3]
        tat, deg = self._tat()
        nt = int(self._temas().sum())
        # 2. deneme düzeltmesi: "ayakta" = dik, en az 3 ayak yerde, gövde havada değil (zıplayan takla hilesine karşı)
        ayakta = dik > 0.8 and 0.9 < z < 1.5 and nt >= 3
        # ----- ödül -----
        r = 0.0
        if "durus" in self.terimler:
            yuks = np.clip(z / 1.1, 0, 1) if z < 1.5 else max(0.0, 1.0 - 2.0 * (z - 1.5))
            rd = (0.4 * np.clip(dik, 0, 1) + 0.3 * yuks + 0.3 * min(nt, 3) / 3 - 0.02 * np.mean(a[:42] ** 2)
                  - 0.1 * min(np.linalg.norm(d.qvel[3:6]) / 20, 2) - 0.1 * min(abs(hiz[2]) / 20, 2))
            r += rd
            if self.kalkti is None:
                self.durus_son = rd
        if "yuru" in self.terimler and not self.damlalar and self.kalkti is None:
            # 3. deneme ara düzeltmesi (275 bin): hız ödülü 10 mm/s'de doyar; yoksa hızlı sekmek sağlam basmaktan kârlıydı
            r += (np.clip(np.dot(hiz[:2], ileri), -10, 10) / 10.0) * float(ayakta) - 0.02 * abs(R[:2, 1] @ hiz[:2]) / 10.0
            r += 0.5 * float(nt >= 3)          # 500 bin: sağlam basış ödülü (sınavın temas şartıyla hizalandı)
        # keşif yönlendirmesi: tadı bakılmamış en yakın damlaya (tat 1. denemesi: yönlendirmesiz sinek yerinde durdu)
        if "koku" in self.terimler and self.hedefler and self.kalkti is None:
            hq, dn = self._hedef()
            r += 10.0 * float(np.clip(self.d_onceki - dn, -0.03, 0.1)) * float(ayakta)  # +10 mm/s'de doyar, ceza en çok -0,3
            # yönelme: gövde damlaya döndükçe (değerlendirme damla yönünü kullanır, sinek yalnız koklar)
            yon = hq - d.xpos[self.hd][:2]
            r += 0.3 * float(ileri @ (yon / (np.linalg.norm(yon) + 1e-9))) * float(ayakta)
            self.d_onceki = dn
        bitti = False
        if deg is not None and self.kalkti is None:
            if self.ilk_tat is None:
                self.ilk_tat = deg
            self.tatli = self.tatli or deg == "seker"
            if self.gorev == "gosteri":
                if self.yedi is None:
                    self.yedi, self.yeme_t = deg, self.t
                    if deg != "seker":                  # acıysa hortum açılmaz, beslenme yok; devam
                        self.damlalar = [x for x in self.damlalar if x[0] != deg]; self.yedi = None
            else:
                # BEYİN MODELİNİN ÖDÜLÜ, her adım: değilen damlanın tadına MN9'un (hortum motor nöronu) ölçülen tepkisi.
                # (1. koku denemesi: değince bölüm bitiyordu; ayakta kalma ödülü yüzünden ulaşmak zarar ettiriyordu.)
                r += 1.0 * MN9_NORM[deg]
                if deg != "seker" and any(t == deg for t, _ in self.hedefler):
                    self.hedefler = [x for x in self.hedefler if x[0] != deg]
                    self.d_onceki = self._hedef()[1]
        if "kacis" in self.terimler:
            if self.kalkti is not None and abs(self.kalkti - self.t) < DT_KON + 1e-9:
                r -= 2.0
            if self.yakalandi:
                r -= 10.0; bitti = True
        if self.kalkti is not None and self.gorev != "gosteri":
            gecti = self.golge is None and all(self.t > gl["t0"] for gl in self.golgeler)
            if gecti or self.t - self.kalkti > 0.4:
                self.kacti = not self.yakalandi
                bitti = True
                if self.kacti:
                    # Uçuşla biten bölüm: kalkış anındaki duruş ödülü hızıyla kalan sürenin değeri (düşmüş sinek kalkarak
                    # kazanamaz). Tehdit yokken kalkış toplam -10 (kalkışta -2, burada -8); gerçek gölgede yalnız -2.
                    # (5. aşama 1-2. denemeler: değer uçurumu duruşu da bozdu / gölgesiz kalkış ödüllendi.)
                    n = max(0.0, (self.sure - self.t) / DT_KON)
                    r += max(0.0, self.durus_son) * (1 - 0.99 ** n) / (1 - 0.99)
                    if not self.tehdit:
                        r -= self.kalkis_bedeli - 2.0    # tehditsiz kalkışın toplam bedeli (varsayılan 10)
        if self.yakalandi:
            bitti = True
        # Düşmek bölümü bitirmez (1. deneme: bitirince keşif gürültüsü bölümleri 50 ms'ye indirdi, yürüme öğrenilemedi).
        # Düşen sinek ödül alamaz; önce doğrulmak zorundadır.
        if bozuk:
            bitti = True
        self.aci_hiz = (aci2 - self.aci_once) / DT_KON if self.golge is not None else 0.0
        self.aci_once = aci2
        kesik = self.t >= self.sure - 1e-9 and not bitti
        info = dict(gorev=self.gorev, ayakta=ayakta, z=float(z), dik=float(dik), tat=deg, kalkti=self.kalkti,
                    yakalandi=self.yakalandi, bozuk=bozuk, t=self.t, temas=nt, ilk_tat=self.ilk_tat, tatli=self.tatli,
                    x=d.xpos[self.tb].copy(), ileri_hiz=float(np.dot(hiz[:2], ileri)))
        return self._gozlem(), float(r), bool(bitti), bool(kesik), info
