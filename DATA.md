# Data guide

Everything needed to inspect or re-run the experiments is in this repository: **all instances** and the **complete results** (per instance and per seed). This file documents where each item is, the file formats, the units, and every column of every result file. Integrity checksums are in `SHA256SUMS.txt`.

## 1. Instances

| Set | Content | Files | Location |
|---|---|---|---|
| I | 282 synthetic layouts (regular, irregular, with holes, literature-inspired) | 282 `.txt` | `instances/generated/` |
| II | PCGTSPLIB instances (abstract distance matrices with precedence) | 33 `.pcgtsp` + 33 `.pcglns` | `external_tools/pcglns/PCGTSPLIB/` (original), `external_tools/pcglns/PCGLNS_PCGTSP/` (solver format) |
| III | 50 literature layouts (25 layouts x packing / separated) | 25 + 25 `.txt` | `instances/original/packing/`, `instances/original/separated/` |
| extra | 6 layouts of Zhang et al. (2024) | 6 `.txt` + 6 `.pack` | `instances/zhang2024/lcpp/` (segments), `instances/zhang2024/test/` (original `.pack`) |

**Segment files (`.txt`, Sets I, III and Zhang).** The first line is the number of segments `N`; the next `N` lines are `x1 y1 x2 y2`. Segments are grouped into closed polygons by endpoint connectivity (`src/data/load_instances.py`).

**Names in Set I (`instances/generated/`).** `art_<k>_<regular|irregular|hole>_<pack|sep>.txt` are the 150 artificial layouts (25 base patterns x 3 geometric types x pack/sep); `<name>[_mod]_<pack|sep>.txt` are the 132 literature-inspired layouts built from 33 ESICUP designs (`_mod` = stochastically modified variant). `pack` = pieces touching, `sep` = separated.

**`.pcgtsp` / `.pcglns` (Set II).** PCGTSPLIB format (Salman, Ekstedt & Damaschke, 2020) and the same instances converted to the PCGLNS input format (`GTSP_SETS`, `EDGE_WEIGHT_SECTION`, `GTSP_SET_SECTION`, `GTSP_SET_ORDERING`, `START_GROUP_SECTION`). `pcgtsp_comparison/converter.py` produces `.pcglns` files from the geometric layouts as well.

The PCGTSPLIB subset has 33 of the 39 instances; `rbg150a` onwards (4-21 MB each) are not included and were not used.

## 2. Objective and units

* **Geometric sets (I, III, Zhang):** the objective (FO) is total time = cutting length / `v_cut` + non-cutting travel / `v_air`, with `v_cut = 100/6` and `v_air = 400`, using the Chebyshev metric in the layout's own coordinate units. The cutting term is constant for a layout (`cut_time_fixed`). PCGLNS operates on a matrix of 10 x Chebyshev distances, so its tour cost is converted as `fo_pcglns = (tour_cost / 10) / v_air + cut_time_fixed`.
* **PCGTSPLIB (Set II):** the objective is the tour cost in the instance's own distance-matrix units; there is no cutting term.
* **Gap convention:** `delta = 100 * (FO_heuristic - FO_reference) / FO_reference`; negative values mean the heuristic is better.
* **Budgets and seeds:** 3 s per run for the heuristic's local search, the ILS and the VNS; 20 s for PCGLNS on the geometric sets; seeds 0-9 where results are per seed. Set I and Set III heuristic rows use seed 0.

## 3. Result files (`pcgtsp_comparison/`)

Columns marked `*_indep` / `*recompute*` come from an independent re-computation of the objective, feasibility and precedence from the produced path or tour; they should show zero differences and zero violations.

### `rerun_frente1_282_FINAL.csv`

Set I (282 synthetic instances): final heuristic vs. PCGLNS, one row per instance. (282 rows)

| Column | Meaning |
|---|---|
| `instance` | Instance name (file name without extension). |
| `n_polygons` | Number of contours (groups) after preprocessing (hole extraction, intersection resolution, QuadTree clustering). |
| `n_pares_pi` | Number of precedence pairs in Pi (hole before its parent contour), detected by geometric containment. |
| `fo_baseline` | Objective (FO) of the construction-only heuristic in its original configuration (no local search; does not enforce Pi). |
| `t_baseline` | Runtime of the construction-only heuristic (s). |
| `fo_forte` | Objective of the final heuristic (construction + tree local search, 3 s budget, seed 0). |
| `t_forte` | Runtime of the final heuristic (s). |
| `fo_forte_recomputado_indep` | Objective of the final path recomputed independently from its segments. |
| `diff_fo_recompute` | |fo_forte_recomputado_indep - fo_forte| (0 = consistent). |
| `ganho_pct_vs_baseline` | 100 * (fo_baseline - fo_forte) / fo_baseline. |
| `estrut_duplicadas` | Structural check on the final path: physical edges cut more than once. |
| `estrut_faltando` | Structural check: required edges never cut. |
| `estrut_extras` | Structural check: cut edges that are not required. |
| `pi_checados` | Precedence pairs checked on the final path. |
| `pi_violacoes` | Precedence pairs violated on the final path. |
| `n_kicks` | ILS perturbations (kicks) attempted within the budget. |
| `n_kick_improvements` | Kicks that improved the solution. |
| `edge_swap_evals` | Candidate evaluations of the edge-swap refinement. |
| `edge_swap_improvements` | Improving edge swaps applied. |
| `erro` | Error message if a step failed (empty otherwise). In Set III, the four single-contour 'packing' layouts (instance_01_2pol/3pol/4pol/6pol) show 'pcglns: sem arquivo de saida': PCGLNS does not accept a GTSP with a single group, so they have no PCGLNS reference. |
| `fo_pcglns` | PCGLNS objective converted to the same unit: (tour_cost/10)/v_air + cut_time_fixed. Empty if PCGLNS refused the instance. |
| `delta_pct_novo` | 100 * (fo_forte - fo_pcglns) / fo_pcglns (negative = the heuristic is better). |
| `categoria` | regular | irregular | hole (with precedence) | esicup_nomeada (literature-inspired). |

### `originais_50_FULL.csv`

Set III (50 literature layouts = 25 layouts x {packing, separated}): final heuristic vs. PCGLNS. (50 rows)

| Column | Meaning |
|---|---|
| `instance` | Layout name (file name without extension). |
| `representacao` | 'packing' (touching pieces) or 'separated'. |
| `n_polygons` | Number of contours (groups) after preprocessing (hole extraction, intersection resolution, QuadTree clustering). |
| `n_pares_pi` | Number of precedence pairs in Pi (hole before its parent contour), detected by geometric containment. |
| `fo_baseline` | Objective (FO) of the construction-only heuristic in its original configuration (no local search; does not enforce Pi). |
| `t_baseline` | Runtime of the construction-only heuristic (s). |
| `fo_forte` | Objective of the final heuristic (construction + tree local search, 3 s budget, seed 0). |
| `t_forte` | Runtime of the final heuristic (s). |
| `fo_forte_recomputado_indep` | Objective of the final path recomputed independently from its segments. |
| `diff_fo_recompute` | |fo_forte_recomputado_indep - fo_forte| (0 = consistent). |
| `ganho_pct_vs_baseline` | 100 * (fo_baseline - fo_forte) / fo_baseline. |
| `estrut_duplicadas` | Structural check on the final path: physical edges cut more than once. |
| `estrut_faltando` | Structural check: required edges never cut. |
| `estrut_extras` | Structural check: cut edges that are not required. |
| `pi_checados` | Precedence pairs checked on the final path. |
| `pi_violacoes` | Precedence pairs violated on the final path. |
| `n_kicks` | ILS perturbations (kicks) attempted within the budget. |
| `n_kick_improvements` | Kicks that improved the solution. |
| `edge_swap_evals` | Candidate evaluations of the edge-swap refinement. |
| `edge_swap_improvements` | Improving edge swaps applied. |
| `erro` | Error message if a step failed (empty otherwise). In Set III, the four single-contour 'packing' layouts (instance_01_2pol/3pol/4pol/6pol) show 'pcglns: sem arquivo de saida': PCGLNS does not accept a GTSP with a single group, so they have no PCGLNS reference. |
| `fo_pcglns` | PCGLNS objective converted to the same unit: (tour_cost/10)/v_air + cut_time_fixed. Empty if PCGLNS refused the instance. |
| `delta_pct_novo` | 100 * (fo_forte - fo_pcglns) / fo_pcglns (negative = the heuristic is better). |

### `rerun_v19_perf_10seeds_FULL.csv`

Set II (33 PCGTSPLIB instances): final ILS heuristic, 10 seeds per instance, 3 s per run. Objective = tour cost in the instance's own distance-matrix units. (330 rows)

| Column | Meaning |
|---|---|
| `instance` | PCGTSPLIB instance name. |
| `seed` | Random seed (0-9). |
| `fo_ils_v19_perf` | Tour cost found by the ILS. |
| `t` | Runtime (s). |
| `perturb_aceitas` | Accepted perturbations. |
| `perturb_tentadas` | Attempted perturbations. |
| `restarts` | Restarts from the anchor pool. |
| `ancoras_criadas` | Anchors created for the pool. |
| `comps_bimodais_detectados` | Components flagged as bimodal by the adaptive restart rule. |
| `viol_reportado` | Precedence violations reported by the algorithm. |
| `custo_recomputado_indep` | Tour cost recomputed independently from the tour. |
| `viol_recomputado_indep` | Precedence violations recomputed independently from the tour. |
| `dup_grupo` | True if some group is visited more than once. |
| `faltando_grupo` | True if some group is not visited. |
| `diff_fo` | |fo_ils_v19_perf - custo_recomputado_indep|. |

### `vns_33x10_fiel_FULL.csv`

Set II: VNS baseline (faithful to Zhang et al., 2024), 10 seeds per instance, 3 s per run. (330 rows)

| Column | Meaning |
|---|---|
| `instance` | PCGTSPLIB instance name. |
| `seed` | Random seed (0-9). |
| `fo_vns` | Tour cost found by the VNS (instance units). |
| `t` | Runtime (s). |
| `shakes_aceitos` | Accepted shake moves. |
| `shakes_tentados` | Attempted shake moves. |
| `viol_reportado` | Precedence violations reported by the algorithm. |
| `custo_recomputado_indep` | Tour cost recomputed independently from the tour. |
| `viol_recomputado_indep` | Precedence violations recomputed independently from the tour. |
| `dup_grupo` | True if some group is visited more than once. |
| `faltando_grupo` | True if some group is not visited. |
| `diff_fo` | |fo_vns - custo_recomputado_indep|. |

### `comparacao_pcgtsplib_real.csv`

Set II: PCGLNS reference values (used by the consolidation) and the constructive heuristic without local search. (33 rows)

| Column | Meaning |
|---|---|
| `instance` | PCGTSPLIB instance name. |
| `n_grupos` | Number of groups (clusters). |
| `n_pares` | Number of precedence pairs. |
| `fo_direta` | Tour cost of the constructive heuristic without local search (instance units). |
| `viol_direta` | Precedence violations of that tour. |
| `t_direta` | Its runtime (s). |
| `fo_pcglns` | PCGLNS tour cost (instance units). |
| `t_pcglns` | PCGLNS runtime (s). |
| `delta_pct` | 100 * (fo_direta - fo_pcglns) / fo_pcglns. |
| `erro` | Error message (empty if none). |

### `vns_282_50_geometricas.csv`

VNS baseline on Sets I and III, 10 seeds per instance (3 s per run), one row per (instance, seed). (3320 rows)

| Column | Meaning |
|---|---|
| `pool` | 'sintetica' (Set I) or 'esicup' (Set III). |
| `instance` | Instance name; Set III uses '<layout>__packing' / '<layout>__separated'. |
| `seed` | Random seed (0-9). |
| `fo_vns_raw` | VNS tour cost in the units of the .pcglns matrix (10 x Chebyshev distance). |
| `fo_vns` | VNS objective in time units: (fo_vns_raw/10)/v_air + cut_time_fixed. |
| `t` | Runtime (s). |
| `shakes_aceitos` | Accepted shake moves. |
| `shakes_tentados` | Attempted shake moves. |
| `viol_reportado` | Precedence violations reported by the algorithm. |
| `custo_recomputado_indep_raw` | Tour cost recomputed independently (raw units). |
| `viol_recomputado_indep` | Precedence violations recomputed independently. |
| `dup_grupo` | True if some group is visited more than once. |
| `faltando_grupo` | True if some group is not visited. |
| `diff_fo_raw` | |fo_vns_raw - custo_recomputado_indep_raw|. |
| `cut_time_fixed` | Constant cutting time of the layout, sum(L_i / v_cut). |
| `erro` | Error message (empty if none). |

### `resultado_zhang6.csv`

The six extra layouts of Zhang et al. (2024): heuristic, PCGLNS and VNS (10 seeds each). (6 rows)

| Column | Meaning |
|---|---|
| `instance` | Instance name. |
| `n_polygons` | Number of contours after preprocessing. |
| `n_pares_pi` | Precedence pairs. |
| `fo_baseline` | Construction-only heuristic objective. |
| `t_baseline` | Its runtime (s). |
| `fo_forte_min` | Minimum objective of the final heuristic over 10 seeds. |
| `fo_forte_media` | Mean objective of the final heuristic over 10 seeds. |
| `fo_forte_max` | Maximum objective of the final heuristic over 10 seeds. |
| `t_forte_media` | Mean runtime of the final heuristic (s). |
| `max_diff_fo_recompute` | Largest |recomputed - reported| objective difference over the seeds. |
| `estrut_duplicadas` | Sum over seeds of edges cut more than once. |
| `estrut_faltando` | Sum over seeds of required edges never cut. |
| `estrut_extras` | Sum over seeds of cut edges that are not required. |
| `pi_violacoes_total` | Sum over seeds of precedence violations. |
| `fo_pcglns` | PCGLNS objective (time units). |
| `fo_vns_min` | Minimum VNS objective over 10 seeds. |
| `fo_vns_media` | Mean VNS objective over 10 seeds. |
| `fo_vns_max` | Maximum VNS objective over 10 seeds. |
| `vns_viol_total` | Sum over seeds of VNS precedence violations. |
| `vns_dup_total` | Seeds in which the VNS tour visits a group twice. |
| `vns_falt_total` | Seeds in which the VNS tour misses a group. |
| `vns_max_diff_fo` | Largest |recomputed - reported| VNS cost difference (raw units). |
| `erro` | Error message (empty if none). |

### `completo_por_instancia_282.csv`

Set I consolidated: rerun_frente1_282_FINAL.csv joined with the VNS statistics. (282 rows)

| Column | Meaning |
|---|---|
| `instance` | Instance name (file name without extension). |
| `n_polygons` | Number of contours (groups) after preprocessing (hole extraction, intersection resolution, QuadTree clustering). |
| `n_pares_pi` | Number of precedence pairs in Pi (hole before its parent contour), detected by geometric containment. |
| `fo_baseline` | Objective (FO) of the construction-only heuristic in its original configuration (no local search; does not enforce Pi). |
| `t_baseline` | Runtime of the construction-only heuristic (s). |
| `fo_forte` | Objective of the final heuristic (construction + tree local search, 3 s budget, seed 0). |
| `t_forte` | Runtime of the final heuristic (s). |
| `fo_forte_recomputado_indep` | Objective of the final path recomputed independently from its segments. |
| `diff_fo_recompute` | |fo_forte_recomputado_indep - fo_forte| (0 = consistent). |
| `ganho_pct_vs_baseline` | 100 * (fo_baseline - fo_forte) / fo_baseline. |
| `estrut_duplicadas` | Structural check on the final path: physical edges cut more than once. |
| `estrut_faltando` | Structural check: required edges never cut. |
| `estrut_extras` | Structural check: cut edges that are not required. |
| `pi_checados` | Precedence pairs checked on the final path. |
| `pi_violacoes` | Precedence pairs violated on the final path. |
| `n_kicks` | ILS perturbations (kicks) attempted within the budget. |
| `n_kick_improvements` | Kicks that improved the solution. |
| `edge_swap_evals` | Candidate evaluations of the edge-swap refinement. |
| `edge_swap_improvements` | Improving edge swaps applied. |
| `erro` | Error message if a step failed (empty otherwise). In Set III, the four single-contour 'packing' layouts (instance_01_2pol/3pol/4pol/6pol) show 'pcglns: sem arquivo de saida': PCGLNS does not accept a GTSP with a single group, so they have no PCGLNS reference. |
| `fo_pcglns` | PCGLNS objective converted to the same unit: (tour_cost/10)/v_air + cut_time_fixed. Empty if PCGLNS refused the instance. |
| `delta_pct_novo` | 100 * (fo_forte - fo_pcglns) / fo_pcglns (negative = the heuristic is better). |
| `categoria` | regular | irregular | hole (with precedence) | esicup_nomeada (literature-inspired). |
| `pool` | Source of the instance in the VNS file: 'sintetica' (Set I) or 'esicup' (Set III). |
| `fo_vns_min` | Minimum VNS objective over the 10 seeds. |
| `fo_vns_media` | Mean VNS objective over the 10 seeds. |
| `fo_vns_max` | Maximum VNS objective over the 10 seeds. |
| `fo_vns_std` | Standard deviation of the VNS objective over the 10 seeds. |
| `n_seeds` | Number of VNS seeds (10). |
| `delta_pcglns_pct` | 100 * (heuristic - fo_pcglns) / fo_pcglns. |
| `delta_vns_pct` | 100 * (heuristic - fo_vns_media) / fo_vns_media (negative = the heuristic is better). |

### `completo_por_instancia_50.csv`

Set III consolidated: originais_50_FULL.csv joined with the VNS statistics. (50 rows)

| Column | Meaning |
|---|---|
| `instance` | Layout name (file name without extension). |
| `representacao` | 'packing' (touching pieces) or 'separated'. |
| `n_polygons` | Number of contours (groups) after preprocessing (hole extraction, intersection resolution, QuadTree clustering). |
| `n_pares_pi` | Number of precedence pairs in Pi (hole before its parent contour), detected by geometric containment. |
| `fo_baseline` | Objective (FO) of the construction-only heuristic in its original configuration (no local search; does not enforce Pi). |
| `t_baseline` | Runtime of the construction-only heuristic (s). |
| `fo_forte` | Objective of the final heuristic (construction + tree local search, 3 s budget, seed 0). |
| `t_forte` | Runtime of the final heuristic (s). |
| `fo_forte_recomputado_indep` | Objective of the final path recomputed independently from its segments. |
| `diff_fo_recompute` | |fo_forte_recomputado_indep - fo_forte| (0 = consistent). |
| `ganho_pct_vs_baseline` | 100 * (fo_baseline - fo_forte) / fo_baseline. |
| `estrut_duplicadas` | Structural check on the final path: physical edges cut more than once. |
| `estrut_faltando` | Structural check: required edges never cut. |
| `estrut_extras` | Structural check: cut edges that are not required. |
| `pi_checados` | Precedence pairs checked on the final path. |
| `pi_violacoes` | Precedence pairs violated on the final path. |
| `n_kicks` | ILS perturbations (kicks) attempted within the budget. |
| `n_kick_improvements` | Kicks that improved the solution. |
| `edge_swap_evals` | Candidate evaluations of the edge-swap refinement. |
| `edge_swap_improvements` | Improving edge swaps applied. |
| `erro` | Error message if a step failed (empty otherwise). In Set III, the four single-contour 'packing' layouts (instance_01_2pol/3pol/4pol/6pol) show 'pcglns: sem arquivo de saida': PCGLNS does not accept a GTSP with a single group, so they have no PCGLNS reference. |
| `fo_pcglns` | PCGLNS objective converted to the same unit: (tour_cost/10)/v_air + cut_time_fixed. Empty if PCGLNS refused the instance. |
| `delta_pct_novo` | 100 * (fo_forte - fo_pcglns) / fo_pcglns (negative = the heuristic is better). |
| `pool` | Source of the instance in the VNS file: 'sintetica' (Set I) or 'esicup' (Set III). |
| `instance_vns` | Instance key in vns_282_50_geometricas.csv ('<layout>__<representacao>'). |
| `fo_vns_min` | Minimum VNS objective over the 10 seeds. |
| `fo_vns_media` | Mean VNS objective over the 10 seeds. |
| `fo_vns_max` | Maximum VNS objective over the 10 seeds. |
| `fo_vns_std` | Standard deviation of the VNS objective over the 10 seeds. |
| `n_seeds` | Number of VNS seeds (10). |
| `instance_base` | Layout name without the representation suffix. |
| `delta_pcglns_pct` | 100 * (heuristic - fo_pcglns) / fo_pcglns. |
| `delta_vns_pct` | 100 * (heuristic - fo_vns_media) / fo_vns_media (negative = the heuristic is better). |

### `completo_por_instancia_33.csv`

Set II consolidated (mean over 10 seeds). (33 rows)

| Column | Meaning |
|---|---|
| `instance` | PCGTSPLIB instance name. |
| `fo_heur_media` | Mean tour cost of the final ILS heuristic over the 10 seeds. |
| `fo_pcglns` | PCGLNS tour cost. |
| `fo_vns_media` | Mean VNS tour cost over the 10 seeds. |
| `delta_pcglns_pct` | 100 * (fo_heur_media - fo_pcglns) / fo_pcglns. |
| `delta_vns_pct` | 100 * (fo_heur_media - fo_vns_media) / fo_vns_media. |

### `completo_por_instancia_6zhang.csv`

Zhang et al. layouts consolidated. (6 rows)

| Column | Meaning |
|---|---|
| `instance` | Instance name. |
| `n_polygons` | Number of contours after preprocessing. |
| `n_pares_pi` | Precedence pairs. |
| `fo_baseline` | Construction-only heuristic objective. |
| `t_baseline` | Its runtime (s). |
| `fo_forte_min` | Minimum objective of the final heuristic over 10 seeds. |
| `fo_forte_media` | Mean objective of the final heuristic over 10 seeds. |
| `fo_forte_max` | Maximum objective of the final heuristic over 10 seeds. |
| `t_forte_media` | Mean runtime of the final heuristic (s). |
| `max_diff_fo_recompute` | Largest |recomputed - reported| objective difference over the seeds. |
| `estrut_duplicadas` | Sum over seeds of edges cut more than once. |
| `estrut_faltando` | Sum over seeds of required edges never cut. |
| `estrut_extras` | Sum over seeds of cut edges that are not required. |
| `pi_violacoes_total` | Sum over seeds of precedence violations. |
| `fo_pcglns` | PCGLNS objective (time units). |
| `fo_vns_min` | Minimum VNS objective over 10 seeds. |
| `fo_vns_media` | Mean VNS objective over 10 seeds. |
| `fo_vns_max` | Maximum VNS objective over 10 seeds. |
| `vns_viol_total` | Sum over seeds of VNS precedence violations. |
| `vns_dup_total` | Seeds in which the VNS tour visits a group twice. |
| `vns_falt_total` | Seeds in which the VNS tour misses a group. |
| `vns_max_diff_fo` | Largest |recomputed - reported| VNS cost difference (raw units). |
| `erro` | Error message (empty if none). |
| `delta_pcglns_pct` | 100 * (fo_forte_media - fo_pcglns) / fo_pcglns. |
| `delta_vns_pct` | 100 * (fo_forte_media - fo_vns_media) / fo_vns_media. |

### `estatisticas_heur_vs_vns_4conjuntos.csv`

Heuristic vs. VNS on the four collections (one-sample Wilcoxon on delta_vns_pct). (4 rows)

| Column | Meaning |
|---|---|
| `conjunto` | Collection. |
| `n` | Number of instances. |
| `media_pct` | Mean of delta_vns_pct. |
| `mediana_pct` | Median of delta_vns_pct. |
| `n_neg` | Instances where the heuristic is better (delta < 0). |
| `n_pos` | Instances where the heuristic is worse (delta > 0). |
| `n_zero` | Ties. |
| `wilcoxon_p` | Two-sided Wilcoxon signed-rank p-value (H0: median = 0). |
| `rank_biserial_r` | Paired rank-biserial correlation (effect size). |

### `significancia_estatistica.csv`

Heuristic vs. PCGLNS (and VNS) per collection: Wilcoxon, effect size and bootstrap confidence interval. (10 rows)

| Column | Meaning |
|---|---|
| `comparacao` | Comparison label. |
| `n` | Number of instances. |
| `media_pct` | Mean percentage delta. |
| `mediana_pct` | Median percentage delta. |
| `desvio_pct` | Standard deviation of the percentage delta. |
| `n_neg` | Instances where the heuristic is better. |
| `n_pos` | Instances where the heuristic is worse. |
| `n_zero` | Ties. |
| `wilcoxon_stat` | Wilcoxon signed-rank statistic. |
| `wilcoxon_p` | Two-sided p-value (H0: median = 0). |
| `rank_biserial_r` | Paired rank-biserial correlation. |
| `ic95_lo` | Lower bound of the 95% bootstrap CI of the mean (20,000 resamples, seed 0). |
| `ic95_hi` | Upper bound of the 95% bootstrap CI of the mean. |

### `ablacao_densidade_precedencia.csv`

Sensitivity of the gap to precedence structure (instances with non-empty Pi). (59 rows)

| Column | Meaning |
|---|---|
| `conjunto` | '282-sinteticas' (Set I, category hole) or '50-esicup' (Set III). |
| `instance` | Instance name. |
| `n_polygons` | Number of contours. |
| `n_pares_pi` | Number of precedence pairs. |
| `densidade` | n_pares_pi / n_polygons. |
| `largura` | Width of the precedence DAG (size of the largest antichain, via Dilworth's theorem). |
| `delta_pcglns_pct` | 100 * (heuristic - PCGLNS) / PCGLNS. |

## 4. Reproducibility notes

* The per-seed and per-instance CSVs are the raw outputs behind every table of the paper; the consolidated files and the statistics are regenerated from them by `agregar_tudo.py`, `significancia_estatistica.py` and `ablacao_densidade_precedencia.py` (no solver needed).
* PCGLNS (ALNS under a wall-clock budget) is not exactly repeatable and time-limited components (the 3 s local search) depend on machine speed, so a fresh run can differ slightly from the stored values. The stored values are the ones reported in the paper.
* The solutions themselves (cutting paths and tours) are not stored, only objectives, timings and feasibility checks; `python src/main.py` regenerates paths and plots for the base heuristic, and the drivers in `pcgtsp_comparison/` regenerate the rest.

## 5. Provenance and third-party material

* Set I was generated for this work (pipeline of Nepomuceno de Carvalho et al., 2025) from pieces of the ESICUP collection and synthetic generators.
* Set III is the 50-instance benchmark of Amaro Junior et al. (2023), built from ESICUP pieces.
* Set II is the PCGTSPLIB of Salman, Ekstedt & Damaschke (2020), as distributed with the PCGLNS code (Kuryavtsev & Khachay, Apache-2.0).
* The six extra layouts come from the reproducibility capsule of Zhang et al. (2024).
* Please consult the original sources for their terms of use. Full references are in the paper and in the top-level `README.md`.
