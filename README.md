# Learning to escape costs lives

Code, trained models and result files for the preprint

> Yıldırım, Ö. (2026). *Learning to escape costs lives: a hard-wired looming reflex outperforms reinforcement learning in an embodied fruit fly model.* bioRxiv (link to be added after posting).

An adult *Drosophila* body (NeuroMechFly v2, MuJoCo) forages while looming objects approach. Half of them strike the fly and half pass within 4–8 mm.
A hard-wired reflex (20° angular threshold + 7 ms giant fiber latency taken from the whole-brain model of Shiu et al., 2024) is compared with PPO learning circuits trained on the same encounters.
A state-dependent dynamic programme predicts when hunger should raise the escape threshold, and the FlyWire whole-brain leaky integrate-and-fire model is used to test whether sugar input changes giant fiber firing.

Code comments and file names are partly in Turkish. The tables below give the English meaning of the main names.

## Repository layout

| Path | Content |
|---|---|
| `rl/dunya.py` | Embodied world: body, looming objects (`golge`), sugar drops, takeoff programme, rewards (Gymnasium env `Sinek`) |
| `rl/kacis_devresi.py` | Escape-circuit environment (`KacisOrtam`): frozen locomotion policy + 2-action (wait / take off) decision every 10 ms |
| `rl/mufredat.py` | Curriculum used to train the locomotion and foraging policy (SAC) |
| `rl/model/` | MJCF body model and meshes (derived from NeuroMechFly v2, see below) |
| `rl/modeller/beyin_G4_tat.zip` | Trained, frozen locomotion and foraging policy (SAC, Stable-Baselines3) |
| `rl/makale/dogustan.py`, `tarama.py` | Encounter task (`Ortam`), learners with the original reward (`ogren`), reflex (`refleks`) and never-takeoff fly (`hic`) |
| `rl/makale/olum.py` | Learners with the corrected death cost (capture = −K meals) |
| `rl/makale/esik_olc.py` | Reflex threshold family (20–60°) on the same 300 encounters |
| `rl/makale/deterministik.py` | Deterministic (greedy) evaluation of trained learners on the reflex encounters |
| `rl/makale/olasilik.py` | Learned takeoff probability per 10 ms step (background and peak) → `olasilik_ozet.json` (Table S1) |
| `rl/makale/omur.py` | Lifetime fitness *W(L)* |
| `rl/makale/olum_karsilastir.py` | Comparison of reflex and learners → `olum_ozet.json` |
| `rl/makale/aclik.py` | State-dependent dynamic programme (hunger) → `aclik_ozet.json` |
| `rl/makale/tarama/` | Per-encounter result files (`*.csv`) and trained PPO models (`*.zip`) |
| `rl/makale/biorxiv/sekiller.py`, `kare_ciz.py` | Figures of the preprint (`figures/`) and the rendered frames of Figure 1 (`kareler/`) |
| `kod/` | Whole-brain model experiments (`model.py`, `utils.py` from Shiu et al., 2024) |
| `sonuclar/` | Summary results of the brain-model experiments (`*.json`) |

Main result files: `rl/makale/olum_ozet.json` (Figure 2, Tables 1 and S1), `rl/makale/olasilik_ozet.json` (Table S1), `rl/makale/aclik_ozet.json` (Figure 4), `sonuclar/aclik_golge.json` and `sonuclar/seker_esik*.json` (Figure 5, Table 2).

Columns of the per-encounter CSV files: `bolum` encounter index, `adim` training step, `gercek` striking object (1) or harmless object (0), `kalkti` took off, `yakalandi` captured, `odul` episode return.
File names: `ogren_b<B>_s<seed>` learner with the original reward, `olum_k<K>_b<B>_s<seed>` learner with death cost K meals, `refleks_b<B>` reflex, `hic_b<B>` never-takeoff fly, `esik_<deg>` reflex with another threshold, `det_*` deterministic evaluation.

## Installation

Python 3.11. Install the packages in `requirements.txt`. We used two environments, one for the body simulation and learning and one for the brain model, but a single environment also works.

Brain-model data are not included. Put these files into `indirilen/`:
- `Connectivity_783.parquet` and `Completeness_783.csv` from https://github.com/philshiu/Drosophila_brain_model
- `neuron_annotations.tsv` (rename of `Supplemental_file1_neuron_annotations.tsv`) from https://github.com/flyconnectome/flywire_annotations

## Reproducing the results

Run from `rl/makale/` unless stated otherwise. Each learner run takes about 10–20 min on a laptop CPU.

```bash
python tarama.py refleks 300 10            # reflex, 300 encounters (also: hic)
python esik_olc.py 25 30 35 40 50 60       # threshold family
python tarama.py ogren 1 100000 100        # learner, original reward (seed, steps, B)
python olum.py ogren 1 100000 100 100      # learner, death cost K = 100 meals (seed, steps, B, K)
python deterministik.py olum_k100_b100_s1  # greedy evaluation of a trained learner
python olasilik.py                         # learned takeoff probabilities of all trained learners
python omur.py && python aclik.py && python olum_karsilastir.py
python biorxiv/sekiller.py                 # figures
```

`tarama_kos.sh` and `olum_kos.sh` run all costs and seeds (set `PYTHON` to your interpreter).

Brain model (run from `kod/`): `python deneyler.py` (baseline runs), `python aclik_golge.py` (saturating inputs), `python seker_esik.py tara 3 5 7 10 15 20 25` (threshold test) and `python seker_esik_ciz.py` (statistics).

## Third-party material

- `kod/model.py`, `kod/utils.py`: Shiu, P. K. et al. (2024), *Nature* 634, 210–219, https://github.com/philshiu/Drosophila_brain_model, MIT License (`kod/LICENSE_Shiu_et_al_MIT.txt`).
- `rl/model/`: body model and meshes derived from NeuroMechFly v2 / FlyGym (Wang-Chen et al., 2024, *Nature Methods* 21, 2353–2362), https://github.com/NeLy-EPFL/flygym, Apache License 2.0.
- FlyWire connectome v783: Dorkenwald et al. (2024) and Schlegel et al. (2024), *Nature* 634. Data are obtained from the original sources and are not redistributed here.

## License

Code in this repository is released under the MIT License (`LICENSE`), except for the third-party material listed above.
