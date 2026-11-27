"""bioRxiv makalesinin İngilizce şekilleri. Yalnız var olan sonuçları okur (yeni deney yok).
Girdiler: ../tarama/*.csv, ../olum_ozet.json, ../omur_ozet.json, ../aclik.py (DP), ../../../sonuclar/{aclik_golge,seker_esik,seker_esik_test}.json,
kareler/kare_{1,2,3}.png (kare_ciz.py).
  python sekiller.py   (sinek_govde_env)  -> figures/Fig{1..5}.png/pdf, figures/FigS1.png/pdf
"""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MAK = os.path.dirname(HERE)
TAR = os.path.join(MAK, "tarama")
SON = os.path.join(os.path.dirname(os.path.dirname(MAK)), "sonuclar")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, MAK)
from aclik import E, MET, T, dp, secenekler  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "axes.labelweight": "bold",
                     "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
                     "font.size": 10, "hatch.linewidth": 0.6, "axes.titlesize": 10.5})
KOYU, ORTA = "#404040", "#8c8c8c"
BS = (10, 30, 60, 100)


def kaydet(fig, ad):
    fig.savefig(os.path.join(OUT, ad + ".png"), dpi=300)
    fig.savefig(os.path.join(OUT, ad + ".pdf"))
    plt.close(fig)
    print(ad)


def baslik(ax, t):
    ax.set_title(t, fontweight="bold", loc="left")


def oku(yol):
    r = list(csv.DictReader(open(yol)))
    a = lambda k: np.array([float(x[k]) for x in r])
    return dict(g=a("gercek").astype(int), k=a("kalkti").astype(int), y=a("yakalandi").astype(int))


# ---------------- Figure 1: task and reflex threshold family ----------------
def fig1():
    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.25], hspace=0.28, wspace=0.05)
    kj = json.load(open(os.path.join(HERE, "kareler", "kare.json")))
    etiket = [("(a) Looming object, 10°", "kare_1"), ("(b) Threshold crossed", "kare_2"), ("(c) Takeoff + 70 ms", "kare_3")]
    for i, (t, ad) in enumerate(etiket):
        ax = fig.add_subplot(gs[0, i])
        im = mpimg.imread(os.path.join(HERE, "kareler", ad + ".png"))
        h, w = im.shape[:2]
        ax.imshow(im[int(0.0 * h):int(0.88 * h), int(0.05 * w):int(0.95 * w)])
        ax.set_xticks([]); ax.set_yticks([])
        baslik(ax, t)
        ax.text(0.03, 0.05, f"t = {kj[ad]['t'] - 0.6:.2f} s", transform=ax.transAxes, color="white", fontsize=8.5)
    ax = fig.add_subplot(gs[1, :])
    sec = secenekler()
    x = np.arange(len(sec)); w = 0.38
    kac = [s["kacis"] for s in sec]; bos = [s["sahte_kalkis"] for s in sec]
    ax.bar(x - w / 2, kac, w, color=KOYU, edgecolor="black", lw=0.8, label="Escape from a real (striking) object")
    ax.bar(x + w / 2, bos, w, color="white", edgecolor="black", hatch="////", lw=0.8, label="Takeoff for a harmless (passing) object")
    for xi, (a, b) in enumerate(zip(kac, bos)):
        ax.text(xi - w / 2, a + 0.02, f"{a:.2f}", ha="center", fontsize=7.5)
        ax.text(xi + w / 2, b + 0.02, f"{b:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([("never" if not np.isfinite(s["esik"]) else f"{s['esik']:g}°") for s in sec])
    ax.set_xlabel("Takeoff threshold (angular size of the object on the eye)")
    ax.set_ylabel("Probability per encounter")
    ax.set_ylim(0, 1.15)
    baslik(ax, "(d) Hard-wired reflex family, 300 encounters per threshold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.17)
    kaydet(fig, "Fig1")


# ---------------- Figure 2: learner vs reflex with corrected death cost ----------------
def fig2():
    d = json.load(open(os.path.join(MAK, "olum_ozet.json")))
    gruplar = [("refleks", "Hard-wired reflex (20° + 7 ms)", dict(color=KOYU, edgecolor="black")),
               ("eski", "Learner, death ≈ 1 meal", dict(color="white", edgecolor="black", hatch="////")),
               ("yeni_k100", "Learner, death = 100 meals", dict(color=ORTA, edgecolor="black"))]
    paneller = (("kacis", "Escape probability\n(striking object)", "(a) Escape"),
                ("sahte_kalkis", "Takeoff probability\n(harmless object)", "(b) False alarms"),
                ("ogrenirken_olum", "Deaths during training", "(c) Cost of learning"),
                ("olgun_W100", "Lifetime food, L = 100 (meals)", "(d) Lifetime fitness"))
    fig, axs = plt.subplots(2, 2, figsize=(7, 5.9)); axs = axs.ravel()
    x = np.arange(len(BS)); w = 0.26
    for ax, (anah, yl, bas) in zip(axs, paneller):
        for j, (ad, et, stil) in enumerate(gruplar):
            if ad == "refleks" and anah == "ogrenirken_olum":
                ax.bar(x + (j - 1) * w, [0] * len(BS), w, label=et, **stil)
                continue
            ort, alt, ust = [], [], []
            for B in BS:
                v = d[str(B)][ad]
                vals = [v[anah]] if isinstance(v, dict) else [z[anah] for z in v]
                ort.append(np.mean(vals)); alt.append(np.mean(vals) - min(vals)); ust.append(max(vals) - np.mean(vals))
            ax.bar(x + (j - 1) * w, ort, w, yerr=[alt, ust], capsize=2, lw=0.8, label=et,
                   error_kw=dict(lw=0.8, ecolor="black"), **stil)
        ax.set_xticks(x); ax.set_xticklabels([str(B) for B in BS]); ax.set_ylabel(yl)
        baslik(ax, bas)
    axs[0].set_ylim(0, 1.08); axs[1].set_ylim(0, 1.08)
    fig.supxlabel("Cost of a false alarm, B (simulator units)", fontweight="bold", y=0.085)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.005))
    fig.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.17, wspace=0.38, hspace=0.38)
    kaydet(fig, "Fig2")


# ---------------- Figure 3: learning dynamics at B = 100 ----------------
def kayan(x, w=501):
    c = np.convolve(x, np.ones(w), "valid") / w
    return np.arange(w // 2, w // 2 + len(c)), c


def fig3():
    fig, axs = plt.subplots(2, 2, figsize=(7, 5.6)); axs = axs.ravel()
    stiller = [("ogren_b100", "Learner, death ≈ 1 meal", dict(color=ORTA, ls="--")),
               ("olum_k10_b100", "Learner, death = 10 meals", dict(color=KOYU, ls=":")),
               ("olum_k100_b100", "Learner, death = 100 meals", dict(color="black", ls="-"))]
    for on, et, st in stiller:
        for s in (1, 2, 3):
            r = oku(os.path.join(TAR, f"{on}_s{s}.csv"))
            olum = r["y"] * r["g"]
            n = np.arange(1, len(olum) + 1)
            lab = et if s == 1 else None
            axs[0].plot(n, np.cumsum(olum), lw=1.2, label=lab, **st)
            gi = np.where(r["g"] == 1)[0]; fi = np.where(r["g"] == 0)[0]
            xs, ys = kayan(1.0 - olum[gi].astype(float), 251)
            axs[1].plot(gi[xs], ys, lw=1.1, **st)
            xs, ys = kayan(r["k"][fi].astype(float), 251)
            axs[2].plot(fi[xs], ys, lw=1.1, **st)
    rf = oku(os.path.join(TAR, "refleks_b100.csv"))
    axs[0].axhline(0, color="black", lw=2.2, label="Hard-wired reflex")
    axs[1].axhline(1 - (rf["y"] * rf["g"])[rf["g"] == 1].mean(), color="black", lw=2.2)
    axs[2].axhline(rf["k"][rf["g"] == 0].mean(), color="black", lw=2.2)
    axs[0].set_ylabel("Cumulative deaths"); baslik(axs[0], "(a) Deaths during training")
    axs[1].set_ylabel("Escape probability\n(striking object)"); baslik(axs[1], "(b) Escape")
    axs[2].set_ylabel("Takeoff probability\n(harmless object)"); baslik(axs[2], "(c) False alarms")
    axs[1].set_ylim(0.5, 1.02); axs[2].set_ylim(0.5, 1.02)
    for ax in axs[:3]:
        ax.set_xlim(0, 10500); ax.set_xlabel("Training encounter")
    h, l = axs[0].get_legend_handles_labels()
    axs[3].axis("off")
    axs[3].legend(h, l, loc="center", frameon=False, title="B = 100", title_fontproperties=dict(weight="bold"))
    fig.subplots_adjust(left=0.11, right=0.98, top=0.95, bottom=0.09, wspace=0.38, hspace=0.42)
    kaydet(fig, "Fig3")


# ---------------- Figure 4: state-dependent threshold (hunger) ----------------
def fig4():
    sec = secenekler()
    e = np.arange(1, E + 1) / MET
    fig, axs = plt.subplots(2, 2, figsize=(7, 5.8)); axs = axs.ravel()
    for b, et, ls, c in [(0.30, "b = 0.30 (B = 60)", "-", "#000000"), (0.53, "b = 0.53 (B = 100)", "-", KOYU),
                         (0.60, "b = 0.60", "--", KOYU), (0.70, "b = 0.70", "--", ORTA)]:
        pol, _ = dp(sec, b)
        y = [pol[k] for k in range(1, E + 1)]
        axs[0].step(e, y, where="mid", ls=ls, color=c, lw=1.6, label=et)
        axs[1].step(e, [sec[k]["kacis"] for k in y], where="mid", ls=ls, color=c, lw=1.6)
    axs[0].set_yticks(range(len(sec)))
    axs[0].set_yticklabels([("never" if not np.isfinite(s["esik"]) else f"{s['esik']:g}°") for s in sec])
    axs[0].set_ylabel("Optimal takeoff threshold"); baslik(axs[0], "(a) Optimal threshold")
    axs[1].set_ylabel("Escape probability\n(striking object)"); baslik(axs[1], "(b) Predicted escape"); axs[1].set_ylim(0, 1.05)
    for ax in axs[:2]:
        ax.set_xlabel("Energy reserve (encounters)"); ax.set_xlim(0, E / MET)
    h = json.load(open(os.path.join(MAK, "aclik_ozet.json")))["hayatta"]
    bb = np.array(h["b"])
    axs[2].plot(bb, np.maximum(h["refleks20"], 1e-12), "-", color="black", lw=1.8, label="Fixed reflex (20°)")
    axs[2].plot(bb, np.maximum(h["en_iyi_sabit"], 1e-12), ":", color=ORTA, lw=2.0, label="Best fixed threshold")
    axs[2].plot(bb, np.maximum(h["durumlu"], 1e-12), "--", color=KOYU, lw=1.6, label="State-dependent threshold")
    axs[2].axvline(0.53, color=ORTA, lw=0.8); axs[2].text(0.526, 2e-4, "B = 100", fontsize=9, ha="right")
    axs[2].set_xlabel("Fraction of a meal lost by takeoff (b)")
    axs[2].set_ylabel(f"Survival to encounter {T}")
    baslik(axs[2], "(c) Survival")
    axs[2].set_yscale("log"); axs[2].set_ylim(1e-4, 1.6); axs[2].set_xlim(bb[0], bb[-1])
    axs[2].set_xticks([0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]); axs[2].set_xticklabels(["0.40", "", "0.50", "", "0.60", "", "0.70"])
    h0, l0 = axs[0].get_legend_handles_labels(); h2, l2 = axs[2].get_legend_handles_labels()
    axs[3].axis("off")
    lg = axs[3].legend(h0, l0, loc="upper center", frameon=False, title="Panels (a) and (b)", title_fontproperties=dict(weight="bold"))
    axs[3].add_artist(lg)
    axs[3].legend(h2, l2, loc="lower center", frameon=False, title="Panel (c)", title_fontproperties=dict(weight="bold"))
    fig.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.09, wspace=0.42, hspace=0.42)
    kaydet(fig, "Fig4")


# ---------------- Figure 5: whole-brain model ----------------
def fig5():
    ag = json.load(open(os.path.join(SON, "aclik_golge.json")))
    se = json.load(open(os.path.join(SON, "seker_esik.json")))
    st = json.load(open(os.path.join(SON, "seker_esik_test.json")))
    fig, axs = plt.subplots(2, 2, figsize=(7, 6.2)); axs = axs.ravel()
    kos = [("seker", "Sugar"), ("lplc2", "Looming"), ("seker_lplc2", "Sugar +\nlooming"), ("aci_lplc2", "Bitter +\nlooming")]
    x = np.arange(len(kos)); w = 0.38
    gf = [np.mean(ag[k]["gf"]) for k, _ in kos]; mn = [np.mean(ag[k]["mn9"]) for k, _ in kos]
    axs[0].bar(x - w / 2, gf, w, color=KOYU, edgecolor="black", lw=0.8, label="Giant fiber (DNp01)")
    axs[0].bar(x + w / 2, mn, w, color="white", edgecolor="black", hatch="////", lw=0.8, label="MN9")
    for xi, (a, b) in enumerate(zip(gf, mn)):
        axs[0].text(xi - w / 2, a + 3, f"{a:.0f}", ha="center", fontsize=7.5)
        axs[0].text(xi + w / 2, b + 3, f"{b:.1f}" if b < 1 else f"{b:.0f}", ha="center", fontsize=7.5)
    axs[0].set_xticks(x); axs[0].set_xticklabels([k for _, k in kos], fontsize=8.5)
    axs[0].set_ylabel("Firing rate (Hz)"); axs[0].set_ylim(0, 185)
    baslik(axs[0], "(a) Saturating inputs (100 Hz)")
    axs[0].legend(loc="upper left", ncol=2, frameon=False, fontsize=8.5)
    hz = sorted({int(k.split("_")[1]) for k in se})
    for kosul, et, stil in (("yalniz", "Looming only", dict(color="black", marker="o", mfc="black")),
                            ("seker", "Looming + sugar", dict(color=KOYU, marker="s", mfc="white", ls="--"))):
        g = [se[f"{kosul}_{h}"]["gf"] for h in hz]
        axs[1].plot(hz, [np.mean(v["ateslenen_deneme"]) for v in g], lw=1.3, ms=4.5, label=et, **stil)
        axs[2].plot(hz, [np.mean(v["deneme_sayim"]) for v in g], lw=1.3, ms=4.5, label=et, **stil)
        gec = [np.nanmean([u if u is not None else np.nan for u in v["ilk_ms_medyan"]]) for v in g]
        axs[3].plot(hz, gec, lw=1.3, ms=4.5, label=et, **stil)
    for h_ in hz:
        if st[str(h_)]["p_sayim"] < 0.05:
            yy = max(np.mean(se[f"yalniz_{h_}"]["gf"]["deneme_sayim"]), np.mean(se[f"seker_{h_}"]["gf"]["deneme_sayim"]))
            axs[2].text(h_, yy + 1.5, "**", ha="center", fontsize=10)
    axs[1].set_ylabel("P(giant fiber fires in trial)"); axs[1].set_ylim(-0.03, 1.08); baslik(axs[1], "(b) Firing probability")
    axs[2].set_ylabel("Giant fiber spikes per trial"); baslik(axs[2], "(c) Spike count")
    axs[3].set_ylabel("First spike latency (ms)"); baslik(axs[3], "(d) Latency")
    for ax in axs[1:]:
        ax.set_xticks(hz); ax.set_xlabel("LPLC2 input rate (Hz)")
    axs[1].legend(loc="lower right", frameon=False, fontsize=8.5)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.08, wspace=0.34, hspace=0.42)
    kaydet(fig, "Fig5")


# ---------------- Figure S1: lifetime fitness curves (naive death cost) ----------------
def figS1():
    out = json.load(open(os.path.join(MAK, "omur_ozet.json")))
    fig, axs = plt.subplots(2, 2, figsize=(7, 6.0), sharey=True, sharex=True); axs = axs.ravel()
    for ax, B, harf in zip(axs, BS, "abcd"):
        o = out[str(B)]; LS = np.array(o["L"])
        ax.plot(LS, o["refleks"]["W"], "-", color="black", lw=1.8, label="Hard-wired reflex")
        for anah, stil, et in (("olgun", "-", "Trained learner, fixed policy"),
                               ("omur_icinde", "--", "Learner trained within its own life")):
            M = np.array([x["W"] for x in o[anah]], dtype=float)
            ax.fill_between(LS, np.nanmin(M, 0), np.nanmax(M, 0), color=ORTA, alpha=0.35, lw=0)
            ax.plot(LS, np.nanmean(M, 0), stil, color=KOYU, lw=1.3, label=et)
        ax.plot(LS, o["hic"]["W"], ":", color=ORTA, lw=1.6, label="Never takes off")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1, 5000)
        baslik(ax, f"({harf}) B = {B}, b = {max(0.0, o['c'] / o['V']):.2f}")
    for ax in (axs[0], axs[2]):
        ax.set_ylabel("Lifetime food (meals)")
    for ax in (axs[2], axs[3]):
        ax.set_xlabel("Lifespan (encounters)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.005))
    fig.subplots_adjust(left=0.11, right=0.98, top=0.95, bottom=0.17, wspace=0.12, hspace=0.3)
    kaydet(fig, "FigS1")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, figS1):
        f()
