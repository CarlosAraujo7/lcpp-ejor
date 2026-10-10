# LCPP — Laser Cutting Path Planning Problem

Code, instances and result files for the paper that formulates laser cutting path planning (LCPP) as a
geometric specialization of the *Precedence-Constrained Generalized Traveling Salesman Problem* (PCGTSP),
at the intersection with the *Hierarchical Chinese Postman Problem* (HCPP).

The proposed method combines spatial decomposition (QuadTree), a KD-tree proximity graph, a minimum spanning
tree, a hierarchical traversal with precedence handled through the lowest common ancestor (LCA), and a
post-construction local search on the routing tree (edge swap + iterated local search). It is evaluated
against PCGLNS (Kuryavtsev & Khachay) and against a VNS re-implemented from Zhang et al. (2024).

The objective (FO) is the total processing time: cutting time at `v_cut = 100/6` plus non-cutting
(laser off) movement time at `v_air = 400`, both measured with the Chebyshev metric.

> Paper: *citation to be added after publication.*

**Data.** All instances used in the paper and the complete results (per instance and per seed) are in this repository.
See [`DATA.md`](DATA.md) for where each file is, the formats and units, and a description of every result column;
`SHA256SUMS.txt` lists checksums of all data files (`sha256sum -c SHA256SUMS.txt`).

## Repository layout

```
src/                       Core of the heuristic
  heuristic_core.py          construction pipeline, LCA/precedence traversals, objective function
  busca_local_arvore.py      tree local search (true MST, cheap edge swap, ILS)
  graph_routing.py, quadtree.py, geometry.py, local_search_tree.py, timer.py
  main.py                    runs the base heuristic over every instance, writes results/heuristic/
  data/load_instances.py     instance loader (segment files -> polygons)

pcgtsp_comparison/         Empirical comparison pipeline (produces the numbers in the paper)
  converter.py               LCPP instance -> .pcglns (PCGTSP) conversion
  run_comparison.py          PCGLNS wrapper (Julia) and precedence checkers
  comparacao_tres_vias.py    original heuristic vs. heuristic with precedence vs. PCGLNS (hole instances)
  heuristica_direta_pcgtsp.py  heuristic applied directly to abstract PCGTSP (.pcglns) instances
  ils_perturbacao_v19.py     final ILS with adaptive restarts (PCGTSPLIB instances)
  busca_local_v3.py, dp_selecao_cidade.py   local-search operators and DP city selection used by the ILS
  vns_zhang_fiel.py          VNS baseline, faithful to Zhang et al. (2024)
  rerun_*.py                 experiment drivers (see below)
  convert_zhang_pack.py      .pack (Zhang et al.) -> segment files
  agregar_tudo.py            consolidates all methods per instance
  significancia_estatistica.py   Wilcoxon / effect size / bootstrap CI
  ablacao_densidade_precedencia.py   sensitivity to precedence density and DAG width
  *.csv                      raw and consolidated results (see "Result files")

instances/
  generated/                 Set I:   282 synthetic instances
  original/                  Set III: 50 literature layouts (25 layouts x {packing, separated}), ESICUP-derived
  zhang2024/                 6 extra instances from Zhang et al. (2024): lcpp/ (segment files), test/ (.pack)
  comparar_282_vs_50_direto.py     piece-level shape-signature overlap between Set I and Set III
  verificar_proveniencia_esicup.py same audit against the raw ESICUP catalogs (not bundled, see below)

external_tools/pcglns/     PCGLNS solver (Julia, Apache-2.0) and the PCGTSPLIB instances (Set II, 33 instances)
```

Comments and docstrings in the source code are written in Portuguese.

## Requirements

* Python 3.10+ with `pip install -r requirements.txt`.
* Julia **1.6** to run PCGLNS (`juliaup add 1.6`). Newer Julia releases (e.g. 1.12) crash with a GC error on
  this solver. PCGLNS is only needed to regenerate the PCGLNS reference values.

The Python wrappers call Julia as `julia +1.6 --check-bounds=no external_tools/pcglns/runPCGLNS.jl ...`.
This can be configured with environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `LCPP_JULIA` | `julia` | Julia executable |
| `LCPP_JULIA_CHANNEL` | `+1.6` | juliaup channel (`none` = pass no channel, e.g. a plain Julia 1.6 install) |
| `LCPP_TMP_DIR` | `<system tmp>/pcgtsp_run` | scratch directory for `.pcglns` / `.tour` files |

## Quick start

```bash
pip install -r requirements.txt
python src/main.py          # base heuristic over all instances; reports and plots go to results/heuristic/
```

From Python:

```python
import sys
sys.path.insert(0, "src")
from data.load_instances import load_instance
import heuristic_core as hc

name, layout = load_instance("instances/generated/art_1_regular_sep.txt")
cfg = {"root_strategy": "nearest", "child_order": "num_filhos",
       "mst_strategy": "standard", "vertex_strategy": "nearest", "time_limit": 60.0}
print(hc.run_heuristic(list(layout), cfg)["fo_total"])
```

Instance files are plain text: the first line holds the number of segments `N`, followed by `N` lines
`x1 y1 x2 y2`. Segments are grouped into closed polygons by endpoint connectivity.

## Instance sets

| Set | Instances | Location | Origin |
|---|---|---|---|
| I | 282 synthetic layouts (regular, irregular, with holes, ESICUP-inspired) | `instances/generated/` | generated for this work |
| II | 33 PCGTSPLIB instances (abstract distance matrices) | `external_tools/pcglns/PCGTSPLIB/` (`.pcgtsp`), `PCGLNS_PCGTSP/` (`.pcglns`) | Salman, Ekstedt & Damaschke (2020) |
| III | 50 literature layouts (25 layouts x {packing, separated}) | `instances/original/` | ESICUP, as assembled by Amaro Junior et al. (2023) |
| extra | 6 layouts | `instances/zhang2024/` | Zhang et al. (2024) |

`PCGTSPLIB/` contains 33 of the 39 PCGTSPLIB instances; `rbg150a` onwards (4-21 MB each) were left out.

The raw ESICUP catalogs (XML) are not bundled. To run `instances/verificar_proveniencia_esicup.py`, download
them from <https://github.com/ESICUP/datasets> (`2d_irregular`) into `instances/external/Set1_ESICUP/`.

## Reproducing the experiments

<!-- TODO(authors): ils_perturbacao_v19.py imports _otimizar_ate_convergir_v3 / _v5 from ils_perturbacao_v6.py
     and ils_perturbacao_v12.py, which are NOT in the source repository (they were not in LCPP-main either).
     rerun_v19_perf_chunk.py depends on it. Add those two modules before publishing; delete this comment after. -->

All drivers write one CSV row per instance (or per instance and seed), incrementally, and resume from where
they stopped if the output file already exists. Run them from the repository root.

| Experiment | Command | Output reference in `pcgtsp_comparison/` |
|---|---|---|
| Set I: heuristic vs. PCGLNS | `python pcgtsp_comparison/rerun_frente1_282_chunk.py 0 282 out.csv` | `rerun_frente1_282_FINAL.csv` |
| Set III: heuristic vs. PCGLNS | `python pcgtsp_comparison/rerun_50_originais_chunk.py 0 50 out.csv` | `originais_50_FULL.csv` |
| Set II: heuristic (33 instances x 10 seeds) | `python pcgtsp_comparison/rerun_v19_perf_chunk.py 0 33 out.csv` | `rerun_v19_perf_10seeds_FULL.csv` |
| Set II: VNS (33 instances x 10 seeds) | `python pcgtsp_comparison/rerun_vns_fiel_chunk.py 0 33 out.csv` | `vns_33x10_fiel_FULL.csv` |
| Sets I and III: VNS (10 seeds) | `python pcgtsp_comparison/rerun_vns_geometricas.py out.csv` | `vns_282_50_geometricas.csv` |
| Zhang et al. instances (3 methods) | `python pcgtsp_comparison/rerun_zhang6.py out.csv` | `resultado_zhang6.csv` |
| Consolidation per instance | `python pcgtsp_comparison/agregar_tudo.py` | `completo_por_instancia_*.csv`, `estatisticas_heur_vs_vns_4conjuntos.csv` |
| Statistical tests | `python pcgtsp_comparison/significancia_estatistica.py` | `significancia_estatistica.csv` |
| Precedence-density sensitivity | `python pcgtsp_comparison/ablacao_densidade_precedencia.py` | `ablacao_densidade_precedencia.csv` |
| Provenance audit: Set I vs. Set III piece overlap | `python instances/comparar_282_vs_50_direto.py` | printed to the terminal |

`rerun_vns_geometricas.py` processes instances in parallel (2 workers) and stops after a wall-clock limit per
call (second argument, default 150 s); call it again to resume.

The consolidation, statistics and sensitivity steps only read files already in the repository, so they can be
re-run without Julia. The heuristic runs use a 3 s budget for the local search; PCGLNS runs on Sets I and III
use a 20 s limit.

### Result files (`pcgtsp_comparison/`)

Every column is documented in [`DATA.md`](DATA.md).

| File | Content |
|---|---|
| `rerun_frente1_282_FINAL.csv` | Set I, one row per instance: heuristic, PCGLNS, structural and precedence checks |
| `originais_50_FULL.csv` | Set III, same columns plus `representacao` |
| `rerun_v19_perf_10seeds_FULL.csv` | Set II, heuristic, 10 seeds per instance |
| `vns_33x10_fiel_FULL.csv` | Set II, VNS, 10 seeds per instance |
| `comparacao_pcgtsplib_real.csv` | Set II, PCGLNS reference values |
| `vns_282_50_geometricas.csv` | VNS on Sets I and III, 10 seeds per instance |
| `resultado_zhang6.csv` | the six Zhang et al. instances (heuristic, PCGLNS, VNS) |
| `completo_por_instancia_{282,50,33,6zhang}.csv` | consolidated table per instance |
| `estatisticas_heur_vs_vns_4conjuntos.csv` | heuristic vs. VNS tests on the four collections |
| `significancia_estatistica.csv` | heuristic vs. PCGLNS (and VNS) tests: Wilcoxon, rank-biserial r, bootstrap CI |
| `ablacao_densidade_precedencia.csv` | precedence density / DAG width vs. gap to PCGLNS |

## Third-party material

* `external_tools/pcglns/` — PCGLNS, Kuryavtsev & Khachay, <https://github.com/AndreiKud/PCGLNS>
  (Apache-2.0, see `external_tools/pcglns/LICENSE`).
* PCGTSPLIB — Salman, Ekstedt & Damaschke (2020).
* ESICUP datasets — <https://github.com/ESICUP/datasets>.
* Zhang, Yao, Liu, Wei & Zhang (2024), *Heuristic approaches for the cutting path problem*, Expert Systems with
  Applications 237 — instances and the VNS algorithm re-implemented in `vns_zhang_fiel.py`.

## Citation

*To be added after publication.*

## License

The code of this repository is released under the [MIT License](LICENSE). Unless noted otherwise, the same license
covers the instances and result files produced by the authors (Set I and the files in `pcgtsp_comparison/`).
Third-party material keeps its original terms: the bundled PCGLNS solver (Apache-2.0, see
`external_tools/pcglns/LICENSE`), PCGTSPLIB, the ESICUP-derived layouts and the instances of Zhang et al. (2024);
please consult the original sources.
