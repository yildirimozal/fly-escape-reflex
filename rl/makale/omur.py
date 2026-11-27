"""Ömür boyu uygunluk: stratejileri tek bir ölçüde (ömür boyunca toplanan besin) karşılaştırır.

Model (karşılaşma = bir gölge, yarısı gerçek; bkz. tarama.py):
  - Sinek ömrü boyunca L karşılaşma yaşar. t. karşılaşmada q_t olasılıkla yakalanır: ömür biter, o ve sonraki
    karşılaşmalardan kazanç yok.
  - Kurtulduğu her karşılaşmada bir öğünlük besin V kazanır. Kalktıysa (gölge gerçek de olsa sahte de olsa öğün yarıda
    kalır) c_B kaybeder.
  - Ömür boyu uygunluk W(L) = sum_t S_{t-1} * g_t,  S_t = prod_{s<=t} (1 - q_s),
    g_t = V * P(kurtuldu) - c_B * P(kalktı ve kurtuldu).
  V ve c_B simülatörün kendi biriminden ölçülür:
  V = hiç kalkmayan sineğin sahte gölgede kalıp öğününü bitirdiği bölümlerin ortalama getirisi.
  c_B = V - refleksin sahte gölgede kalktığı bölümlerin ortalama getirisi.
  Sonuç öğün cinsinden verilir (W / V).
Stratejiler:
  refleks     : bağlantı şeması refleksi (20° + 7 ms), tarama/refleks_b<B>.csv (300 karşılaşma)
  hic         : hiç kalkmayan sinek, tarama/hic_b<B>.csv
  olgun       : öğrenmeyi bitirmiş devre (son 1000 karşılaşmanın oranları, sabit politika; 3 tohum)
  omur_icinde : sıfırdan, kendi ömrü içinde öğrenen birey (öğrenme kaydındaki t. karşılaşmanın yerel oranları,
                101'lik ortalanmış pencere; 3 tohum)
  python omur.py                -> omur_ozet.json, omur.png, omur.pdf (ölüm yalnız gerçek gölgede)
  python omur.py sahte_oldurur  -> *_sahte_oldurur.* (duyarlılık: sahte gölgeye temas da ölüm)
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
T = os.path.join(HERE, "tarama")
BS = (10, 30, 60, 100)
import sys  # noqa: E402
SAHTE_OLDURMEZ = "sahte_oldurur" not in sys.argv
EK = "" if SAHTE_OLDURMEZ else "_sahte_oldurur"
LS = np.unique(np.round(np.logspace(0, np.log10(5000), 60)).astype(int))
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "axes.labelweight": "bold",
                     "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
                     "font.size": 10})


def oku(yol):
    r = list(csv.DictReader(open(yol)))
    a = lambda k: np.array([float(x[k]) for x in r])
    return dict(g=a("gercek").astype(int), k=a("kalkti").astype(int), y=a("yakalandi").astype(int), R=a("odul"))


def birim(B):
    h, rf = oku(os.path.join(T, f"hic_b{B}.csv")), oku(os.path.join(T, f"refleks_b{B}.csv"))
    V = h["R"][(h["g"] == 0) & (h["y"] == 0)].mean()
    kal = rf["R"][(rf["g"] == 0) & (rf["k"] == 1) & (rf["y"] == 0)].mean()
    return float(V), float(V - kal)


def olum(d):
    """Ölüm yalnız gerçek gölgede. Sahte gölge (zararsız cisim) 4-8 mm ıskalayarak geçer; kalkan sinek bazen onun yoluna
    uçup temas ediyor (simülasyon yan etkisi, refleks 2/143, öğrenen 5-11/~4500). Bunlar ölüm sayılmaz."""
    return d["y"] * d["g"] if SAHTE_OLDURMEZ else d["y"]


def oranlar(d):
    """Karşılaşma başına ölüm ve 'kalktı ve kurtuldu' olasılığı (sabit politika)."""
    y = olum(d)
    return y.mean(), (d["k"] * (1 - y)).mean()


def W_sabit(q, kk, V, c, L):
    g = V * (1 - q) - c * kk
    s = 1 - q
    return np.array([g * L_ if q == 0 else g * (1 - s ** L_) / q for L_ in L])


def W_dizi(q, kk, V, c, L):
    g = V * (1 - q) - c * kk
    S = np.concatenate([[1.0], np.cumprod(1 - q)[:-1]])   # t. karşılaşmanın başında hayatta olma
    W = np.cumsum(S * g)
    return np.array([W[L_ - 1] if L_ <= len(W) else np.nan for L_ in L]), S


def yerel(x, w=101):
    c = np.convolve(x, np.ones(w), "same"); n = np.convolve(np.ones_like(x, dtype=float), np.ones(w), "same")
    return c / n


out = {}
for B in BS:
    V, c = birim(B)
    o = dict(V=V, c=c, L=LS.tolist())
    for ad in ("refleks", "hic"):
        q, kk = oranlar(oku(os.path.join(T, f"{ad}_b{B}.csv")))
        o[ad] = dict(q=q, kalkis=kk, W=(W_sabit(q, kk, V, c, LS) / V).tolist())
    ol, oi = [], []
    for s in (1, 2, 3):
        d = oku(os.path.join(T, f"ogren_b{B}_s{s}.csv"))
        son = {k: v[-1000:] for k, v in d.items()}
        q, kk = oranlar(son)
        ol.append(dict(q=q, kalkis=kk, W=(W_sabit(q, kk, V, c, LS) / V).tolist()))
        y = olum(d)
        qy, ky = yerel(y.astype(float)), yerel((d["k"] * (1 - y)).astype(float))
        Wd, S = W_dizi(qy, ky, V, c, LS)
        yasam = float(np.sum(S))                       # beklenen ömür (karşılaşma), kayıt boyu ile sınırlı
        oi.append(dict(W=(Wd / V).tolist(), beklenen_omur=yasam, S_100=float(S[min(99, len(S) - 1)]),
                       S_1000=float(S[min(999, len(S) - 1)])))
    o["olgun"], o["omur_icinde"] = ol, oi
    out[B] = o
    i100 = int(np.argmin(abs(LS - 100)))
    print(f"B={B}: V={V:.1f} c={c:.1f} | L=100 öğün: refleks {o['refleks']['W'][i100]:.1f}, hic {o['hic']['W'][i100]:.1f}, "
          f"olgun {np.mean([x['W'][i100] for x in ol]):.1f}, ömür içinde {np.mean([x['W'][i100] for x in oi]):.1f} "
          f"| ömür içinde beklenen ömür {[round(x['beklenen_omur'], 1) for x in oi]}", flush=True)
json.dump(out, open(os.path.join(HERE, f"omur_ozet{EK}.json"), "w"), indent=1, ensure_ascii=False)

fig, axs = plt.subplots(1, 4, figsize=(11, 3.4), sharey=True)
for ax, B, harf in zip(axs, BS, "abcd"):
    o = out[B]
    ax.plot(LS, o["refleks"]["W"], "-", color="#000000", lw=1.8, label="Bağlantı şeması refleksi")
    for anah, stil, et in (("olgun", "-", "Öğrenmiş devre (sabit, 3 tohum)"),
                           ("omur_icinde", "--", "Kendi ömründe öğrenen (3 tohum)")):
        M = np.array([x["W"] for x in o[anah]])
        ax.fill_between(LS, np.nanmin(M, 0), np.nanmax(M, 0), color="#8c8c8c", alpha=0.35, lw=0)
        ax.plot(LS, np.nanmean(M, 0), stil, color="#404040", lw=1.3, label=et)
    ax.plot(LS, o["hic"]["W"], ":", color="#8c8c8c", lw=1.6, label="Hiç kalkmayan")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(f"({harf}) B = {B}  (öğün kaybı {max(0.0, o['c'] / o['V']):.2f})", fontweight="bold")
    ax.set_xlim(1, 5000)
axs[0].set_ylabel("Ömür boyu besin (öğün)")
fig.supxlabel("Ömür uzunluğu (gölge karşılaşması)", fontweight="bold", y=0.1)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.25, wspace=0.08)
fig.savefig(os.path.join(HERE, f"omur{EK}.png"), dpi=300); fig.savefig(os.path.join(HERE, f"omur{EK}.pdf"))
print(f"omur{EK}.png")
