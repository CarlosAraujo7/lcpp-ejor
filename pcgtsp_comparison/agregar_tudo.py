"""
agregar_tudo.py
================
Consolida heuristica + PCGLNS + VNS nos 4 conjuntos (282 sinteticas, 50
ESICUP, 33 PCGTSPLIB, 6 Zhang2024) em uma tabela por instancia, e roda
Wilcoxon+r (heuristica vs VNS) em cada conjunto onde ainda nao foi feito
(282 e 50 sao novos; 33 e 6 ja existiam, recalculados aqui so para
consistencia de metodologia/arredondamento).

Saidas:
  - completo_por_instancia_282.csv
  - completo_por_instancia_50.csv
  - completo_por_instancia_33.csv
  - completo_por_instancia_6zhang.csv
  - estatisticas_heur_vs_vns_4conjuntos.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PASTA = str(Path(__file__).resolve().parent.parent / "pcgtsp_comparison") + "/"


def rank_biserial(diffs):
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[diffs != 0]
    n = len(diffs)
    if n == 0:
        return 0.0, 0.0, 0.0
    ranks = pd.Series(np.abs(diffs)).rank().values
    w_pos = ranks[diffs > 0].sum()
    w_neg = ranks[diffs < 0].sum()
    return (w_pos - w_neg) / (n * (n + 1) / 2), w_pos, w_neg


def relatar(nome, deltas):
    deltas = np.asarray(deltas, dtype=float)
    n = len(deltas)
    n_neg, n_pos, n_zero = (deltas < 0).sum(), (deltas > 0).sum(), (deltas == 0).sum()
    if n >= 1 and not np.allclose(deltas, 0):
        try:
            stat, p = wilcoxon(deltas, zero_method="wilcox", alternative="two-sided", method="auto")
        except ValueError:
            stat, p = np.nan, np.nan
    else:
        stat, p = np.nan, np.nan
    r, w_pos, w_neg = rank_biserial(deltas)
    print(f"\n=== {nome} (n={n}) ===")
    print(f"  media={deltas.mean():+.4f}%  mediana={np.median(deltas):+.4f}%  desvio={deltas.std(ddof=1) if n>1 else 0:.4f}%")
    print(f"  sinais: neg={n_neg} pos={n_pos} zero={n_zero}  ({n_neg}/{n} vitorias/empates)")
    print(f"  Wilcoxon p={p}  effect size r={r:+.4f}")
    return {"conjunto": nome, "n": n, "media_pct": deltas.mean(), "mediana_pct": np.median(deltas),
            "n_neg": int(n_neg), "n_pos": int(n_pos), "n_zero": int(n_zero),
            "wilcoxon_p": p, "rank_biserial_r": r}


# ---------------------------------------------------------------------------
# 1) Agregar VNS por instancia (282+50) a partir do CSV por-seed
# ---------------------------------------------------------------------------
vns_geo = pd.read_csv(PASTA + "vns_282_50_geometricas.csv")
agg_geo = vns_geo.groupby(["pool", "instance"]).agg(
    fo_vns_min=("fo_vns", "min"), fo_vns_media=("fo_vns", "mean"), fo_vns_max=("fo_vns", "max"),
    fo_vns_std=("fo_vns", "std"), n_seeds=("fo_vns", "count"),
).reset_index()
assert (agg_geo["n_seeds"] == 10).all(), "alguma instancia sem 10 sementes"

# ---------------------------------------------------------------------------
# 2) Conjunto 282: junta com rerun_frente1_282_FINAL.csv (fo_forte, fo_pcglns)
# ---------------------------------------------------------------------------
f282 = pd.read_csv(PASTA + "rerun_frente1_282_FINAL.csv")
vns_282 = agg_geo[agg_geo["pool"] == "sintetica"].copy()
m282 = f282.merge(vns_282, left_on="instance", right_on="instance", how="inner")
assert len(m282) == 282, f"esperava 282, achei {len(m282)}"
m282["delta_pcglns_pct"] = 100 * (m282["fo_forte"] - m282["fo_pcglns"]) / m282["fo_pcglns"]
m282["delta_vns_pct"] = 100 * (m282["fo_forte"] - m282["fo_vns_media"]) / m282["fo_vns_media"]
m282.to_csv(PASTA + "completo_por_instancia_282.csv", index=False)

# ---------------------------------------------------------------------------
# 3) Conjunto 50: junta com originais_50_FULL.csv (fo_forte, fo_pcglns),
#    por (instance, representacao)
# ---------------------------------------------------------------------------
f50 = pd.read_csv(PASTA + "originais_50_FULL.csv")
vns_50 = agg_geo[agg_geo["pool"] == "esicup"].copy()
vns_50[["instance_base", "representacao"]] = vns_50["instance"].str.split("__", expand=True)
m50 = f50.merge(vns_50, left_on=["instance", "representacao"], right_on=["instance_base", "representacao"],
                 how="inner", suffixes=("", "_vns"))
assert len(m50) == 50, f"esperava 50, achei {len(m50)}"
m50["delta_pcglns_pct"] = 100 * (m50["fo_forte"] - m50["fo_pcglns"]) / m50["fo_pcglns"]
m50["delta_vns_pct"] = 100 * (m50["fo_forte"] - m50["fo_vns_media"]) / m50["fo_vns_media"]
m50.to_csv(PASTA + "completo_por_instancia_50.csv", index=False)

# ---------------------------------------------------------------------------
# 4) Conjunto 33 (ja existia): heuristica de rerun_v19_perf_10seeds_FULL.csv,
#    PCGLNS de comparacao_pcgtsplib_real.csv, VNS de vns_33x10_fiel_FULL.csv
# ---------------------------------------------------------------------------
heur33 = pd.read_csv(PASTA + "rerun_v19_perf_10seeds_FULL.csv")
heur33_agg = heur33.groupby("instance")["fo_ils_v19_perf"].mean().reset_index()
heur33_agg.columns = ["instance", "fo_heur_media"]
pcglns33 = pd.read_csv(PASTA + "comparacao_pcgtsplib_real.csv")
vns33 = pd.read_csv(PASTA + "vns_33x10_fiel_FULL.csv")
vns33_agg = vns33.groupby("instance")["fo_vns"].mean().reset_index()
vns33_agg.columns = ["instance", "fo_vns_media"]
m33 = heur33_agg.merge(pcglns33[["instance", "fo_pcglns"]], on="instance", how="inner")
m33 = m33.merge(vns33_agg, on="instance", how="inner")
assert len(m33) == 33, f"esperava 33, achei {len(m33)}"
m33["delta_pcglns_pct"] = 100 * (m33["fo_heur_media"] - m33["fo_pcglns"]) / m33["fo_pcglns"]
m33["delta_vns_pct"] = 100 * (m33["fo_heur_media"] - m33["fo_vns_media"]) / m33["fo_vns_media"]
m33.to_csv(PASTA + "completo_por_instancia_33.csv", index=False)

# ---------------------------------------------------------------------------
# 5) Conjunto 6 Zhang (ja existia, so re-formata)
# ---------------------------------------------------------------------------
z6 = pd.read_csv(PASTA + "resultado_zhang6.csv")
z6["delta_pcglns_pct"] = 100 * (z6["fo_forte_media"] - z6["fo_pcglns"]) / z6["fo_pcglns"]
z6["delta_vns_pct"] = 100 * (z6["fo_forte_media"] - z6["fo_vns_media"]) / z6["fo_vns_media"]
z6.to_csv(PASTA + "completo_por_instancia_6zhang.csv", index=False)

# ---------------------------------------------------------------------------
# 6) Estatisticas heur-vs-VNS nos 4 conjuntos
# ---------------------------------------------------------------------------
resultados = []
resultados.append(relatar("282 sinteticas (heur vs VNS)", m282["delta_vns_pct"].values))
resultados.append(relatar("50 ESICUP (heur vs VNS)", m50["delta_vns_pct"].values))
resultados.append(relatar("33 PCGTSPLIB (heur vs VNS)", m33["delta_vns_pct"].values))
resultados.append(relatar("6 Zhang2024 (heur vs VNS)", z6["delta_vns_pct"].values))

pd.DataFrame(resultados).to_csv(PASTA + "estatisticas_heur_vs_vns_4conjuntos.csv", index=False)
print("\nSalvo: estatisticas_heur_vs_vns_4conjuntos.csv")
