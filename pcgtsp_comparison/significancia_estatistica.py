"""
significancia_estatistica.py
==============================
Testes de significancia estatistica da comparacao heuristica proposta vs.
PCGLNS (e vs. VNS), por conjunto de instancias.

Para cada comparacao, `relatar` reporta: media, mediana e desvio do delta
percentual; contagem de sinais; Wilcoxon signed-rank de uma amostra contra 0
(H0: a mediana do delta percentual e' zero, i.e., sem vantagem sistematica);
tamanho de efeito (correlacao rank-biserial pareada) e IC 95% bootstrap da
media. Delta percentual = 100 * (FO_heuristica - FO_referencia) / FO_referencia
(negativo = a heuristica proposta e' melhor).

Entradas (ja' presentes neste diretorio; as de 33 instancias sao geradas por
`agregar_tudo.py`):
  - rerun_frente1_282_FINAL.csv      282 sinteticas: `delta_pct_novo`, `categoria`
  - completo_por_instancia_33.csv    33 PCGTSPLIB: media de 10 seeds por instancia
                                     (heuristica, PCGLNS, VNS)
  - originais_50_FULL.csv            50 de literatura (ESICUP / Amaro et al. 2023)

Saida: significancia_estatistica.csv (ou o caminho dado como 1o argumento).

Uso: python significancia_estatistica.py [saida.csv]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PASTA = Path(__file__).resolve().parent


def rank_biserial(diffs):
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[diffs != 0]
    n = len(diffs)
    ranks = pd.Series(np.abs(diffs)).rank().values
    w_pos = ranks[diffs > 0].sum()
    w_neg = ranks[diffs < 0].sum()
    total = n * (n + 1) / 2
    assert abs((w_pos + w_neg) - total) < 1e-6
    r = (w_pos - w_neg) / total
    return r, w_pos, w_neg


def bootstrap_ci_mean(x, n_boot=20000, alpha=0.05, seed=0):
    x = np.asarray(x, dtype=float)
    r = np.random.default_rng(seed)
    boots = np.array([r.choice(x, size=len(x), replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi


def relatar(nome, deltas):
    deltas = np.asarray(deltas, dtype=float)
    n = len(deltas)
    n_neg = (deltas < 0).sum()
    n_pos = (deltas > 0).sum()
    n_zero = (deltas == 0).sum()
    stat, p = wilcoxon(deltas, zero_method="wilcox", alternative="two-sided", method="auto")
    r, w_pos, w_neg = rank_biserial(deltas)
    lo, hi = bootstrap_ci_mean(deltas)
    print(f"\n=== {nome} (n={n}) ===")
    print(f"  media={deltas.mean():+.4f}%  mediana={np.median(deltas):+.4f}%  "
          f"desvio={deltas.std(ddof=1):.4f}%")
    print(f"  sinais: neg={n_neg} pos={n_pos} zero={n_zero}")
    print(f"  Wilcoxon signed-rank (H0: mediana=0): statistic={stat:.2f}  p={p:.3e}")
    print(f"  W+ (soma ranks positivos)={w_pos:.1f}  W- (soma ranks negativos)={w_neg:.1f}")
    print(f"  effect size (correlacao rank-biserial pareada) r={r:+.4f}")
    print(f"  IC 95% bootstrap da media: [{lo:+.4f}%, {hi:+.4f}%]")
    return {
        "comparacao": nome, "n": n, "media_pct": deltas.mean(), "mediana_pct": np.median(deltas),
        "desvio_pct": deltas.std(ddof=1), "n_neg": n_neg, "n_pos": n_pos, "n_zero": n_zero,
        "wilcoxon_stat": stat, "wilcoxon_p": p, "rank_biserial_r": r,
        "ic95_lo": lo, "ic95_hi": hi,
    }


def main():
    saida = Path(sys.argv[1]) if len(sys.argv) > 1 else PASTA / "significancia_estatistica.csv"
    resultados = []

    # Conjunto I: 282 instancias sinteticas (heuristica com busca local forte vs. PCGLNS)
    df1 = pd.read_csv(PASTA / "rerun_frente1_282_FINAL.csv")
    resultados.append(relatar("Conjunto I -- TOTAL (282 instancias sinteticas)", df1["delta_pct_novo"]))
    for cat, g in df1.groupby("categoria"):
        resultados.append(relatar(f"Conjunto I -- categoria {cat} (n={len(g)})", g["delta_pct_novo"]))

    # Conjunto II: 33 instancias PCGTSPLIB (media de 10 seeds por instancia)
    df2 = pd.read_csv(PASTA / "completo_por_instancia_33.csv")
    resultados.append(relatar("Conjunto II (PCGTSPLIB) -- TOTAL (33 instancias, media de 10 seeds)",
                              df2["delta_pcglns_pct"]))

    # p43.2 e p43.3 sao as 2 instancias bimodais (duas bacias de atracao de
    # qualidade muito diferente); reporta-se tambem a exclusao simetrica.
    df2_sem_bimodais = df2[~df2["instance"].isin(["p43.2", "p43.3"])]
    resultados.append(relatar("Conjunto II (PCGTSPLIB) -- excluindo p43.2 e p43.3 (n=31)",
                              df2_sem_bimodais["delta_pcglns_pct"]))

    # Segundo baseline: VNS fiel a Zhang et al. (2024), mesmo protocolo
    # (33 instancias x 10 seeds, 3 s por execucao).
    delta_vns_vs_pcglns = 100.0 * (df2["fo_vns_media"] - df2["fo_pcglns"]) / df2["fo_pcglns"]
    resultados.append(relatar("Conjunto II (PCGTSPLIB) -- VNS (2o baseline) vs. PCGLNS (n=33, media de 10 seeds)",
                              delta_vns_vs_pcglns))
    # delta_vns_pct > 0 => a heuristica proposta teve FO MAIOR (pior) que o VNS.
    resultados.append(relatar("Conjunto II (PCGTSPLIB) -- heuristica proposta vs. VNS (n=33, medias de 10 seeds)",
                              df2["delta_vns_pct"]))

    # Conjunto III: 50 instancias de literatura (ESICUP / Amaro et al. 2023).
    # 4 delas (instance_01_2pol/3pol/4pol/6pol, representacao "packing") reduzem,
    # apos o pre-processamento geometrico (tratar_furos + resolver_intersecoes +
    # clusterizar_quadtree), a um UNICO contorno (n_polygons=1): o PCGLNS recusa
    # essas instancias na fonte ("must have more than 1 set") e a comparacao e'
    # estruturalmente inaplicavel. Ficam fora do delta/Wilcoxon, nos dados brutos.
    df3 = pd.read_csv(PASTA / "originais_50_FULL.csv")
    df3_validas = df3[df3["erro"].isna()]
    assert len(df3) == 50 and len(df3_validas) == 46
    resultados.append(relatar("Conjunto III -- 50 instancias de literatura (ESICUP/Amaro2023), "
                              "n=46 validas p/ PCGLNS", df3_validas["delta_pct_novo"]))

    pd.DataFrame(resultados).to_csv(saida, index=False)
    print(f"\nSalvo em {saida}")


if __name__ == "__main__":
    main()
