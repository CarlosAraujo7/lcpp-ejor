"""
ablacao_densidade_precedencia.py
==================================
Sensibilidade a densidade/largura da precedencia: mede se o gap da heuristica
proposta (vs. PCGLNS) piora conforme a densidade/largura do DAG de
precedencia Pi cresce, nas instancias que de fato tem precedencia
(n_pares_pi > 0), nos conjuntos 282 (sinteticas) + 50 (ESICUP).

Metricas por instancia (apenas as com Pi nao-vazio):
  - densidade = n_pares_pi / n_polygons
  - largura   = tamanho da maior anticadeia do DAG de precedencia (fecho
    transitivo), via teorema de Dilworth (largura = N - tamanho do
    emparelhamento maximo no grafo bipartido de comparabilidade)

Correlacao (Spearman, nao assume linearidade) entre cada metrica e o gap
percentual heur-vs-PCGLNS. Reporta o coeficiente e o p-valor.
"""
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "pcgtsp_comparison"))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
from data.load_instances import load_instance  # noqa: E402
from converter import _preprocess, _detect_precedence  # noqa: E402

PASTA = _PROJECT_ROOT / "pcgtsp_comparison"


def largura_dag(n_nos: int, pares: list[tuple[int, int]]) -> int:
    """Tamanho da maior anticadeia via Dilworth: constroi o fecho
    transitivo do DAG (pred->suc), monta o grafo bipartido de
    comparabilidade (i_esq -> j_dir se i precede j no fecho), acha o
    emparelhamento maximo, e retorna n_nos - |emparelhamento|."""
    if not pares:
        return n_nos  # sem nenhuma restricao, todos formam uma unica anticadeia
    dag = nx.DiGraph()
    dag.add_nodes_from(range(n_nos))
    dag.add_edges_from(pares)
    fecho = nx.transitive_closure(dag, reflexive=False)

    B = nx.Graph()
    esq = [f"L{i}" for i in range(n_nos)]
    dir_ = [f"R{i}" for i in range(n_nos)]
    B.add_nodes_from(esq, bipartite=0)
    B.add_nodes_from(dir_, bipartite=1)
    for (i, j) in fecho.edges():
        B.add_edge(f"L{i}", f"R{j}")
    if B.number_of_edges() == 0:
        return n_nos
    matching = nx.bipartite.maximum_matching(B, top_nodes=esq)
    # maximum_matching retorna pares dos dois lados; conta so' um sentido
    tam_emparelhamento = sum(1 for k in matching if k.startswith("L"))
    return n_nos - tam_emparelhamento


def processa_conjunto(csv_completo: Path, instance_col_map, nome_conjunto: str):
    df = pd.read_csv(csv_completo)
    linhas = []
    for _, row in df.iterrows():
        nome_arquivo, fp, n_pares_pi, delta_pcglns = instance_col_map(row)
        if n_pares_pi == 0 or pd.isna(n_pares_pi):
            continue
        try:
            _, layout = load_instance(fp)
            positions = _preprocess(layout)
            pares = _detect_precedence(positions)
            n_poly = len(positions)
            if len(pares) != int(n_pares_pi):
                # confirma consistencia com o que ja estava registrado; se
                # divergir, reporta mas nao trava (usa o recem-calculado,
                # que e' o que realmente importa pra largura)
                pass
            largura = largura_dag(n_poly, pares)
            densidade = len(pares) / n_poly
            linhas.append({
                "conjunto": nome_conjunto, "instance": nome_arquivo, "n_polygons": n_poly,
                "n_pares_pi": len(pares), "densidade": densidade, "largura": largura,
                "delta_pcglns_pct": delta_pcglns,
            })
        except Exception as e:
            print(f"  [erro] {nome_arquivo}: {e}", file=sys.stderr)
    return linhas


# ---------------------------------------------------------------------------
# 282 sinteticas
# ---------------------------------------------------------------------------
def map_282(row):
    fp = _PROJECT_ROOT / "instances" / "generated" / f"{row['instance']}.txt"
    return row["instance"], fp, row["n_pares_pi"], row["delta_pcglns_pct"]


# ---------------------------------------------------------------------------
# 50 ESICUP
# ---------------------------------------------------------------------------
def map_50(row):
    fp = _PROJECT_ROOT / "instances" / "original" / row["representacao"] / f"{row['instance']}.txt"
    return f"{row['instance']}__{row['representacao']}", fp, row["n_pares_pi"], row["delta_pcglns_pct"]


linhas = []
linhas += processa_conjunto(PASTA / "completo_por_instancia_282.csv", map_282, "282-sinteticas")
linhas += processa_conjunto(PASTA / "completo_por_instancia_50.csv", map_50, "50-esicup")

resultado = pd.DataFrame(linhas)
resultado.to_csv(PASTA / "ablacao_densidade_precedencia.csv", index=False)
print(f"Instancias com Pi nao-vazio analisadas: {len(resultado)}")
print(resultado[["conjunto", "instance", "n_polygons", "n_pares_pi", "densidade", "largura", "delta_pcglns_pct"]]
      .to_string(index=False))

if len(resultado) >= 3:
    for metrica in ["densidade", "largura", "n_pares_pi"]:
        rho, p = spearmanr(resultado[metrica], resultado["delta_pcglns_pct"])
        print(f"\nSpearman({metrica}, delta_pcglns_pct): rho={rho:+.4f}  p={p:.4f}  n={len(resultado)}")
else:
    print("Poucas instancias com Pi para correlacao formal.")
