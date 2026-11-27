"""Açlık analizi: enerji durumuna göre en iyi kaçış eşiği (durum bağımlı dinamik programlama, Houston & McNamara tarzı).

Seçenekler simülasyonda ölçülen doğuştan refleks eşikleridir (aynı 300 sahne): 20° (bağlantı şeması), 25-60°, ve
"hiç kalkma" (sonsuz eşik). Her seçeneğin karşılaşma başına sonuç olasılıkları ölçümden gelir (gerçek gölge oranı 0,5):
ölüm (gerçek gölgede yakalanma), kalkıp kurtulma, kalıp kurtulma. Sahte gölgeye temas ölüm sayılmaz (bkz. omur.py).

Enerji modeli (tam sayı birim; 10 birim = bir karşılaşmalık metabolizma):
  - Sinek her karşılaşmada 10 birim harcar; enerjisi 0'a inerse açlıktan ölür. Depo en çok E birim.
  - Kalırsa öğünü bitirir: +M birim. Kalkarsa öğünün b kesrini kaybeder: +M(1-b).
  - b bedel taramasından: B=10, 30, 60, 100 -> b = 0, 0,12, 0,30, 0,53 (omur_ozet.json).
Amaç: T karşılaşma sonra (üreme) hayatta olma olasılığını en çoklamak. Politika t=0'daki (ufka uzak) karardır.
  python aclik.py  -> aclik_ozet.json, aclik.png, aclik.pdf
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
T_DIR = os.path.join(HERE, "tarama")
MET, M, E, T = 10, 20, 50, 100
BS = (10, 30, 60, 100)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "axes.labelweight": "bold",
                     "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
                     "font.size": 10})


def oku(yol):
    r = list(csv.DictReader(open(yol)))
    a = lambda k: np.array([int(x[k]) for x in r])
    return a("gercek"), a("kalkti"), a("yakalandi")


def secenekler():
    """ad -> (eşik derecesi, ölüm, kalkıp kurtulma, kalıp kurtulma, gerçekte kaçış, sahtede kalkış)"""
    yollar = [(20.0, os.path.join(T_DIR, "refleks_b10.csv"))]
    for f in sorted(os.listdir(T_DIR)):
        if f.startswith("esik_") and f.endswith(".csv"):
            yollar.append((float(f[5:-4]), os.path.join(T_DIR, f)))
    yollar.append((np.inf, os.path.join(T_DIR, "hic_b10.csv")))
    out = []
    for d, y in sorted(yollar):
        g, k, c = oku(y)
        c = c * g                                       # ölüm yalnız gerçek gölgede
        pg = 0.5
        olum = pg * c[g == 1].mean()
        kalk = pg * (k * (1 - c))[g == 1].mean() + (1 - pg) * k[g == 0].mean()
        out.append(dict(esik=d, olum=float(olum), kalk=float(kalk), kal=float(1 - olum - kalk),
                        kacis=float(1 - c[g == 1].mean()), sahte_kalkis=float(k[g == 0].mean()), n=int(len(g))))
    return out


def dp(sec, b, met=MET, m=M, e_max=E, T=T):
    """V[t][e]: t'de e enerjiyle T'ye sağ ulaşma olasılığı. Döndürür: politika (e -> seçenek indeksi, t=0), V0."""
    V = np.ones(e_max + 1); V[0] = 0.0
    kazan_kal, kazan_kalk = m, int(round(m * (1 - b)))
    pol = None
    for _ in range(T):
        Q = np.zeros((len(sec), e_max + 1))
        for i, s in enumerate(sec):
            for e in range(1, e_max + 1):
                e_kal = min(e_max, max(0, e - met + kazan_kal))
                e_kalk = min(e_max, max(0, e - met + kazan_kalk))
                Q[i, e] = s["kal"] * V[e_kal] + s["kalk"] * V[e_kalk]   # ölüm: 0
        # eşitlikte daha düşük eşik (refleks tarafı) seçilir
        pol = np.argmax(Q + 1e-12 * np.arange(len(sec))[::-1, None], axis=0)
        V = Q.max(axis=0); V[0] = 0.0
    return pol, V


def sabit(sec_i, s, b, met=MET, m=M, e_max=E, T=T):
    return dp([s], b, met, m, e_max, T)[1]


def main():
    sec = secenekler()
    oz = json.load(open(os.path.join(HERE, "omur_ozet.json")))
    bler = {B: max(0.0, oz[str(B)]["c"] / oz[str(B)]["V"]) for B in BS}
    for s in sec:
        print(f"eşik {s['esik']:>5}: kaçış {s['kacis']:.3f} sahtede kalkış {s['sahte_kalkis']:.3f} | ölüm {s['olum']:.3f} kalk {s['kalk']:.3f}")
    out = dict(model=dict(MET=MET, M=M, E=E, T=T, gercek_oran=0.5), secenekler=sec, sonuc={}, duyarlilik={})
    for B in BS:
        b = bler[B]
        pol, V = dp(sec, b)
        sabitler = {s["esik"]: sabit(i, s, b) for i, s in enumerate(sec)}
        en_iyi_sabit = max(sabitler, key=lambda d: sabitler[d][E])
        out["sonuc"][B] = dict(b=b, politika_esik=[None if e == 0 else sec[pol[e]]["esik"] for e in range(E + 1)],
                               politika_kacis=[None if e == 0 else sec[pol[e]]["kacis"] for e in range(E + 1)],
                               V_durumlu=V.tolist(), V_refleks20=sabitler[20.0].tolist(),
                               en_iyi_sabit=en_iyi_sabit, V_en_iyi_sabit=sabitler[en_iyi_sabit].tolist(),
                               V_hic=sabitler[np.inf].tolist())
        esikler = [sec[pol[e]]["esik"] for e in range(1, E + 1)]
        print(f"B={B} b={b:.2f}: tok (E={E}) eşik {esikler[-1]}, aç (e=10) eşik {esikler[9]}, e=5 eşik {esikler[4]} | "
              f"T'ye sağ: durumlu {V[E]:.3f}, refleks20 {sabitler[20.0][E]:.3f}, en iyi sabit ({en_iyi_sabit}) "
              f"{sabitler[en_iyi_sabit][E]:.3f}, hiç {sabitler[np.inf][E]:.3f}")
    # duyarlılık: öğün büyüklüğü M
    for m in (15, 20, 30):
        out["duyarlilik"][m] = {}
        for B in BS:
            pol, V = dp(sec, bler[B], m=m)
            out["duyarlilik"][m][B] = dict(esik=[None if e == 0 else sec[pol[e]]["esik"] for e in range(E + 1)],
                                           V_durumlu=float(V[E]), V_refleks20=float(dp([sec[0]], bler[B], m=m)[1][E]))
        print(f"M={m}: " + " | ".join(f"B={B} eşik(e=5,10,25,50)=" + ",".join(str(out['duyarlilik'][m][B]['esik'][e]) for e in (5, 10, 25, 50))
                                      for B in BS))
    ciz(out, sec)
    json.dump(out, open(os.path.join(HERE, "aclik_ozet.json"), "w"), indent=1, ensure_ascii=False,
              default=lambda x: None)


B_CIZ = [(0.30, "b = 0,30 (B = 60)", "-", "#000000"), (0.53, "b = 0,53 (B = 100)", "-", "#404040"),
         (0.60, "b = 0,60", "--", "#404040"), (0.70, "b = 0,70", "--", "#8c8c8c")]


def ciz(out, sec):
    """(a) açlığa göre en iyi eşik, (b) öngörülen kaçış, (c) T'ye sağ ulaşma - öğün kaybı b sürekli.
    Seçeneklerin davranışı b'den bağımsız ölçüldüğü için b, simüle edilen B'lerin dışına da taşınabilir (ekolojik parametre)."""
    e = np.arange(1, E + 1) / MET                    # enerji deposu, karşılaşma cinsinden
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6))
    esik_list = [s["esik"] for s in sec]
    out["egriler"] = {}
    for b, et, ls, c in B_CIZ:
        pol, V = dp(sec, b)
        y = [pol[k] for k in range(1, E + 1)]
        axs[0].step(e, y, where="mid", ls=ls, color=c, lw=1.6, label=et)
        axs[1].step(e, [sec[k]["kacis"] for k in y], where="mid", ls=ls, color=c, lw=1.6)
        out["egriler"][b] = [sec[k]["esik"] for k in y]
    axs[0].set_yticks(range(len(esik_list)))
    axs[0].set_yticklabels([("kalkmaz" if not np.isfinite(d) else f"{d:g}°") for d in esik_list])
    axs[0].set_ylabel("En iyi kaçış eşiği")
    axs[0].set_title("(a) Açlığa göre en iyi eşik", fontweight="bold")
    axs[1].set_ylabel("Gerçek gölgeden kaçış olasılığı")
    axs[1].set_title("(b) Öngörülen kaçış", fontweight="bold")
    axs[1].set_ylim(0, 1.05)
    for ax in axs[:2]:
        ax.set_xlabel("Enerji deposu (karşılaşma)")
        ax.set_xlim(0, E / MET)
    bb = np.round(np.arange(0.40, 0.701, 0.01), 2)
    v_d, v_s, v_r = [], [], []
    for b in bb:
        v_d.append(max(dp(sec, b)[1][E], 1e-12))
        fx = [dp([s], b)[1][E] for s in sec]
        v_s.append(max(max(fx), 1e-12)); v_r.append(max(fx[0], 1e-12))
    out["hayatta"] = dict(b=bb.tolist(), durumlu=[float(x) for x in v_d], en_iyi_sabit=[float(x) for x in v_s],
                          refleks20=[float(x) for x in v_r])
    axs[2].plot(bb, v_r, "-", color="#000000", lw=1.8, label="Sabit refleks (20°)")
    axs[2].plot(bb, v_s, ":", color="#8c8c8c", lw=2.0, label="En iyi sabit eşik")
    axs[2].plot(bb, v_d, "--", color="#404040", lw=1.6, label="Açlığa göre ayarlanan eşik")
    axs[2].axvline(0.53, color="#8c8c8c", lw=0.8)
    axs[2].text(0.526, 2e-4, "B = 100", fontsize=9, ha="right")
    axs[2].set_xlabel("Kalkışta kaybedilen öğün kesri (b)")
    axs[2].set_ylabel(f"{T} karşılaşma sonra hayatta")
    axs[2].set_title("(c) Hayatta kalma", fontweight="bold")
    axs[2].set_yscale("log"); axs[2].set_ylim(1e-4, 1.6); axs[2].set_xlim(bb[0], bb[-1])
    axs[2].set_xticks([0.45, 0.50, 0.55, 0.60, 0.65, 0.70])
    h0, l0 = axs[0].get_legend_handles_labels(); h2, l2 = axs[2].get_legend_handles_labels()
    fig.legend(h0, l0, loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.36, 0.06))
    fig.legend(h2, l2, loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.32, wspace=0.42)
    fig.savefig(os.path.join(HERE, "aclik.png"), dpi=300); fig.savefig(os.path.join(HERE, "aclik.pdf"))
    print("aclik.png")


if __name__ == "__main__":
    main()
