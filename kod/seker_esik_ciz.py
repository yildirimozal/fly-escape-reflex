"""seker_esik.json -> sonuclar/seker_esik.png/pdf: dev lif eşik eğrisi, şekersiz ve şekerli.
(a) GF'nin ateşlediği deneme oranı, (b) GF ateşleme hızı, (c) ilk ateşleme gecikmesi; iki GF'nin ortalaması.
  python seker_esik_ciz.py   (matplotlib için sinek_govde_env)"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402

S = Path(__file__).resolve().parent.parent / "sonuclar"
d = json.load(open(S / "seker_esik.json"))
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "axes.labelweight": "bold",
                     "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
                     "font.size": 10})
hz = sorted({int(k.split("_")[1]) for k in d})
fig, axs = plt.subplots(1, 3, figsize=(10, 3.3))
for kosul, et, stil in (("yalniz", "Yalnız gölge (LPLC2)", dict(color="#000000", marker="o", mfc="#000000")),
                        ("seker", "Gölge + şeker tadı", dict(color="#404040", marker="s", mfc="white", ls="--"))):
    x = [h for h in hz if f"{kosul}_{h}" in d]
    g = [d[f"{kosul}_{h}"]["gf"] for h in x]
    axs[0].plot(x, [np.mean(v["ateslenen_deneme"]) for v in g], lw=1.4, ms=5, label=et, **stil)
    axs[1].plot(x, [np.mean(v["hiz_hz"]) for v in g], lw=1.4, ms=5, **stil)
    gec = [np.mean([u for u in v["ilk_ms_medyan"] if u is not None]) if any(u is not None for u in v["ilk_ms_medyan"])
           else np.nan for v in g]
    axs[2].plot(x, gec, lw=1.4, ms=5, **stil)
axs[0].set_ylabel("Dev lifin ateşlediği deneme oranı"); axs[0].set_ylim(-0.03, 1.03)
axs[1].set_ylabel("Dev lif ateşleme hızı (Hz)")
axs[2].set_ylabel("İlk ateşleme (ms, medyan)")
for ax, b in zip(axs, ("(a) Ateşleme olasılığı", "(b) Ateşleme hızı", "(c) Gecikme")):
    ax.set_title(b, fontweight="bold"); ax.set_xticks(hz)
# istatistik: her hızda deneme başına GF ateşleme sayısı (iki GF toplamı), şekersiz vs şekerli, Mann-Whitney U (iki yönlü)
test = {}
for h in hz:
    a, b = d.get(f"yalniz_{h}"), d.get(f"seker_{h}")
    if a and b and "deneme_sayim" in a["gf"]:
        x, y = np.array(a["gf"]["deneme_sayim"]), np.array(b["gf"]["deneme_sayim"])
        p = float(mannwhitneyu(x, y, alternative="two-sided").pvalue)
        ia = np.array([v for v in a["gf"]["deneme_ilk_ms"] if v is not None]); ib = np.array([v for v in b["gf"]["deneme_ilk_ms"] if v is not None])
        pg = float(mannwhitneyu(ia, ib, alternative="two-sided").pvalue) if len(ia) and len(ib) else None
        test[h] = dict(yalniz_ort=float(x.mean()), seker_ort=float(y.mean()), fark_yuzde=float(100 * (y.mean() - x.mean()) / max(x.mean(), 1e-9)),
                       p_sayim=p, yalniz_ilk_ms=float(np.median(ia)) if len(ia) else None,
                       seker_ilk_ms=float(np.median(ib)) if len(ib) else None, p_ilk=pg)
        print(f"{h:>3} Hz: GF sayım/deneme {x.mean():.1f} -> {y.mean():.1f} ({test[h]['fark_yuzde']:+.0f}%), p={p:.3f} | "
              f"ilk ateşleme {test[h]['yalniz_ilk_ms']} -> {test[h]['seker_ilk_ms']} ms, p={pg}")
json.dump(test, open(S / "seker_esik_test.json", "w"), indent=1)
fig.supxlabel("Gölge şiddeti: LPLC2 uyarım hızı (Hz)", fontweight="bold", y=0.11)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.27, wspace=0.35)
fig.savefig(S / "seker_esik.png", dpi=300); fig.savefig(S / "seker_esik.pdf")
print("seker_esik.png")
