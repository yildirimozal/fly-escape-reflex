"""Sinek beyni deneyleri: ortak tanimlar (Shiu vd. 2024 modeli, FlyWire v783)."""
from pathlib import Path
import pandas as pd
D = Path(__file__).resolve().parent.parent / "indirilen"   # Connectivity_783.parquet, Completeness_783.csv, neuron_annotations.tsv (see README)
SONUC = Path(__file__).resolve().parent.parent / "sonuclar"
COMP, CON = D / "Completeness_783.csv", D / "Connectivity_783.parquet"
_an = None
def an():
    global _an
    if _an is None:
        _an = pd.read_csv(D / "neuron_annotations.tsv", sep="\t", low_memory=False).set_index("root_id")
    return _an
def tip(alan, deger, taraf=None):
    a = an(); s = a[a[alan] == deger]
    if taraf: s = s[s.side == taraf]
    comp = pd.read_csv(COMP, index_col=0)
    return [int(i) for i in s.index if i in comp.index]
MN9 = [i for i in tip("cell_type", "CB0701")]           # MN9 (yutak sinirinde hortum motor noronu), iki taraf
SEKER = tip("cell_sub_class", "sugar")
ACI = tip("cell_sub_class", "bitter")
