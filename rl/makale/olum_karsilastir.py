"""Ölüm bedeli düzeltmesinin etkisi: eski öğrenen (ogren_b<B>, yakalanma yaklaşık -10) ile yeni öğrenen
(olum_k<K>_b<B>, yakalanma = -K öğün) ve bağlantı şeması refleksi karşılaştırılır.
Ölçüler omur.py ile aynı birimde: ölüm yalnız gerçek gölgede; V, c_B ve W(L) omur.py'den.
  python olum_karsilastir.py -> olum_ozet.json, olum.png, olum.pdf
"""
import json
import os
import sys

import numpy as np

sys.argv = [sys.argv[0]]                    # omur.py'nin argüman okumasını etkilemesin
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from omur import BS, T, W_dizi, W_sabit, birim, olum, oku, oranlar, yerel  # noqa: E402

L = np.array([100])


def det_ozet(yol, V, c):
    """Deterministik değerlendirme (deterministik.py, refleksle aynı 300 karşılaşma)."""
    d = oku(yol); g = d["g"] == 1; q, kk = oranlar(d)
    return dict(det_kacis=float(1 - olum(d)[g].mean()), det_sahte_kalkis=float(d["k"][~g].mean()),
                det_q=float(q), det_W100=float(W_sabit(q, kk, V, c, L)[0] / V))


def ozet(d, V, c):
    """Bir öğrenme kaydının ölçüleri. Son 1000 karşılaşma = öğrenmiş (sabit) politika (stokastik)."""
    y = olum(d)
    son = {k: v[-1000:] for k, v in d.items()}
    ys = olum(son)
    g = son["g"] == 1
    q, kk = oranlar(son)
    qy, ky = yerel(y.astype(float)), yerel((d["k"] * (1 - y)).astype(float))
    Wd, _ = W_dizi(qy, ky, V, c, L)
    return dict(ogrenirken_olum=int(y.sum()), karsilasma=int(len(y)),
                ilk1000_olum=int(y[:1000].sum()),
                kacis=float(1 - ys[g].mean()), sahte_kalkis=float(son["k"][~g].mean()),
                olgun_q=float(q), olgun_W100=float(W_sabit(q, kk, V, c, L)[0] / V),
                omur_icinde_W100=float(Wd[0] / V))


out = {}
for B in BS:
    V, c = birim(B)
    rf = oku(os.path.join(T, f"refleks_b{B}.csv"))
    g = rf["g"] == 1
    q, kk = oranlar(rf)
    o = dict(refleks=dict(kacis=float(1 - olum(rf)[g].mean()), sahte_kalkis=float(rf["k"][~g].mean()),
                          olgun_q=float(q), olgun_W100=float(W_sabit(q, kk, V, c, L)[0] / V)))
    for ad, sablon in (("eski", "ogren_b{B}_s{s}.csv"), ("yeni_k100", "olum_k100_b{B}_s{s}.csv"),
                       ("yeni_k10", "olum_k10_b{B}_s{s}.csv")):
        yollar = [os.path.join(T, sablon.format(B=B, s=s)) for s in (1, 2, 3)]
        if all(os.path.exists(y) for y in yollar):
            o[ad] = [ozet(oku(y), V, c) for y in yollar]
            for x, y in zip(o[ad], yollar):
                dy = os.path.join(T, "det_" + os.path.basename(y))
                if os.path.exists(dy):
                    x.update(det_ozet(dy, V, c))
    out[B] = o
    for ad in ("eski", "yeni_k100", "yeni_k10"):
        if ad in o:
            m = lambda k: np.mean([x[k] for x in o[ad]])
            print(f"B={B:>3} {ad:>9}: öğrenirken ölüm {[x['ogrenirken_olum'] for x in o[ad]]}, "
                  f"kaçış {m('kacis'):.3f}, sahte kalkış {m('sahte_kalkis'):.3f}, "
                  f"W100 olgun {m('olgun_W100'):.1f}, ömür içinde {m('omur_icinde_W100'):.1f}", flush=True)
            if all("det_W100" in x for x in o[ad]):
                print(f"          deterministik: kaçış {[round(x['det_kacis'], 3) for x in o[ad]]}, "
                      f"sahte kalkış {[round(x['det_sahte_kalkis'], 3) for x in o[ad]]}, W100 {[round(x['det_W100'], 1) for x in o[ad]]}", flush=True)
    r = o["refleks"]
    print(f"B={B:>3}   refleks: kaçış {r['kacis']:.3f}, sahte kalkış {r['sahte_kalkis']:.3f}, W100 {r['olgun_W100']:.1f}")
json.dump(out, open(os.path.join(HERE, "olum_ozet.json"), "w"), indent=1, ensure_ascii=False)

# --- şekil: B'ye göre (a) kaçış, (b) sahte kalkış, (c) öğrenirken ölüm, (d) W(100) ---
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "axes.labelweight": "bold",
                     "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
                     "font.size": 10, "hatch.linewidth": 0.6})
gruplar = [("refleks", "Bağlantı şeması refleksi", dict(color="#404040", edgecolor="black")),
           ("eski", "Öğrenen, eski ödül", dict(color="white", edgecolor="black", hatch="////")),
           ("yeni_k100", "Öğrenen, ölüm = 100 öğün", dict(color="#8c8c8c", edgecolor="black"))]
fig, axs = plt.subplots(1, 4, figsize=(11.5, 3.3))
x = np.arange(len(BS)); w = 0.26
paneller = (("kacis", "Gerçek gölgeden kaçış oranı", "(a) Kaçış"),
            ("sahte_kalkis", "Sahte gölgede kalkış oranı", "(b) Boşuna kalkış"),
            ("ogrenirken_olum", "Öğrenirken ölüm (sayı)", "(c) Öğrenmenin bedeli"),
            ("olgun_W100", "Ömür boyu besin, L = 100 (öğün)", "(d) Ömür boyu uygunluk"))
for ax, (anah, yl, bas) in zip(axs, paneller):
    for j, (ad, et, stil) in enumerate(gruplar):
        if ad == "refleks" and anah == "ogrenirken_olum":
            continue
        ort, alt, ust = [], [], []
        for B in BS:
            v = out[B][ad]
            vals = [v[anah]] if isinstance(v, dict) else [z[anah] for z in v]
            ort.append(np.mean(vals)); alt.append(np.mean(vals) - min(vals)); ust.append(max(vals) - np.mean(vals))
        ax.bar(x + (j - 1) * w, ort, w, yerr=[alt, ust], capsize=2, lw=0.8, label=et,
               error_kw=dict(lw=0.8, ecolor="black"), **stil)
    ax.set_xticks(x); ax.set_xticklabels([str(B) for B in BS]); ax.set_ylabel(yl)
    ax.set_title(bas, fontweight="bold")
axs[0].set_ylim(0, 1.05); axs[1].set_ylim(0, 1.05)
fig.supxlabel("Boşuna kalkışın bedeli B", fontweight="bold", y=0.11)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.subplots_adjust(left=0.05, right=0.99, top=0.9, bottom=0.27, wspace=0.38)
fig.savefig(os.path.join(HERE, "olum.png"), dpi=300); fig.savefig(os.path.join(HERE, "olum.pdf"))
print("olum.png")
