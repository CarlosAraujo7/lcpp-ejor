"""
rerun_50_originais_chunk.py <indice_inicial> <indice_final> <saida.csv>
=======================================================================
Roda a heuristica final (construcao original + busca local forte -- MST
verdadeira + edge-swap + ILS, `busca_local_arvore.py`) contra o PCGLNS nas
50 instancias de `instances/original/` (25 layouts x {packing, separated}),
o conjunto de literatura (ESICUP / Amaro et al. 2023), distinto das 282
sinteticas proprias e das 33 PCGTSPLIB abstratas.

Mesmo padrao de `rerun_frente1_282_chunk.py`: mesma metodologia de
run_comparison.py, mesmas colunas (mais uma, `representacao`, para
distinguir packing/separated ja que os nomes de arquivo se repetem entre
as duas subpastas), gravacao incremental e resumivel.

Uso: python3 rerun_50_originais_chunk.py <ini> <fim> <saida.csv>
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pcgtsp_comparison"))
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
from data.load_instances import load_instance  # noqa: E402
import heuristic_core as hc  # noqa: E402
from converter import convert_instance, SCALE  # noqa: E402
import busca_local_arvore as bla  # noqa: E402
from run_comparison import (  # noqa: E402
    _detect_precedence_positions, run_pcglns, check_pcglns_precedence_violations,
    TMP_DIR, JULIA_BIN, PCGLNS_SCRIPT,
)

INSTANCES_DIR = _PROJECT_ROOT / "instances" / "original"
TIME_BUDGET_FORTE = 3.0
PCGLNS_MAX_TIME = 20.0


def processa_uma(fp: Path, representacao: str) -> dict:
    nome = fp.stem
    row = {"instance": nome, "representacao": representacao, "erro": ""}
    try:
        _, layout_original = load_instance(fp)

        # baseline: heuristica original (config 'original', sem busca local forte)
        _ORIGINAL_CFG = {
            "root_strategy": "nearest", "child_order": "num_filhos",
            "mst_strategy": "standard", "vertex_strategy": "nearest", "time_limit": 60.0,
        }
        t0 = time.time()
        baseline = hc.run_heuristic(list(layout_original), _ORIGINAL_CFG)
        t_baseline = time.time() - t0
        fo_baseline = baseline["fo_total"]

        # posicoes pos-processamento (furos extraidos, intersecoes resolvidas, quadtree)
        positions = hc.clusterizar_quadtree(
            hc.resolver_intersecoes(hc.tratar_furos(list(layout_original)))
        )
        pares_pi = _detect_precedence_positions(positions)

        # raiz: mesma logica da heuristica original, mas sobre o grafo completo
        # (consistente com a MST verdadeira que a busca local forte usa)
        G_full_raiz = hc._build_full_graph(positions)
        raiz = hc.escolher_raiz_nearest(G_full_raiz, positions)

        # busca local forte: MST completa -> edge-swap -> ILS (3s)
        t0 = time.time()
        forte = bla.busca_local_forte(positions, raiz, pares_pi, time_limit=TIME_BUDGET_FORTE, seed=0)
        t_forte = time.time() - t0
        fo_forte = forte["fo_total"]

        # verificacao independente: recomputa FO do zero a partir do caminho
        fo_recompute = hc.calcular_fo(forte["caminho"])
        diff_fo = abs(fo_recompute - fo_forte)

        # verificacao estrutural: cada aresta fisica cortada exatamente uma vez
        cortes = [seg for seg, tag in forte["caminho"] if tag == "c"]

        def _key(seg):
            a = tuple(round(v, 6) for v in seg.coords[0])
            b = tuple(round(v, 6) for v in seg.coords[1])
            return (min(a, b), max(a, b))

        from collections import Counter
        contagem = Counter(_key(s) for s in cortes)
        esperadas = Counter()
        for pol in positions:
            coords = list(pol.exterior.coords)
            for i in range(len(coords) - 1):
                esperadas[_key_from_pts(coords[i], coords[i + 1])] += 1
        duplicadas = sum(1 for k, v in contagem.items() if v > 1)
        faltando = sum(1 for k in esperadas if k not in contagem)
        extras = sum(1 for k in contagem if k not in esperadas)

        # violacoes de Pi
        from run_comparison import check_our_precedence_violations
        pi_checados, pi_violacoes = check_our_precedence_violations(
            layout_original, forte["caminho"], positions, pares_pi
        )

        ganho_pct = 100.0 * (fo_baseline - fo_forte) / fo_baseline if fo_baseline else 0.0

        # PCGLNS
        pcglns_in = TMP_DIR / f"orig_{representacao}_{nome}.pcglns"
        pcglns_out = TMP_DIR / f"orig_{representacao}_{nome}.tour"
        stats = convert_instance(fp, pcglns_in)
        res = run_pcglns(pcglns_in, PCGLNS_MAX_TIME, pcglns_out)
        if not res["ok"]:
            row["erro"] = f"pcglns: {res.get('error')}"
            fo_pcglns = None
            delta_pct_novo = None
        else:
            fo_pcglns = (res["tour_cost_scaled"] / SCALE) / hc._V_AIR + stats["cut_time_fixed"]
            delta_pct_novo = 100.0 * (fo_forte - fo_pcglns) / fo_pcglns if fo_pcglns else None

        row.update({
            "n_polygons": len(positions),
            "n_pares_pi": len(pares_pi),
            "fo_baseline": fo_baseline,
            "t_baseline": round(t_baseline, 3),
            "fo_forte": fo_forte,
            "t_forte": round(t_forte, 3),
            "fo_forte_recomputado_indep": fo_recompute,
            "diff_fo_recompute": diff_fo,
            "ganho_pct_vs_baseline": ganho_pct,
            "estrut_duplicadas": duplicadas,
            "estrut_faltando": faltando,
            "estrut_extras": extras,
            "pi_checados": pi_checados,
            "pi_violacoes": pi_violacoes,
            "n_kicks": forte["meta"]["n_kicks"],
            "n_kick_improvements": forte["meta"]["n_kick_improvements"],
            "edge_swap_evals": forte["meta"].get("edge_swap_evals", 0),
            "edge_swap_improvements": forte["meta"].get("edge_swap_improvements", 0),
            "fo_pcglns": fo_pcglns,
            "delta_pct_novo": delta_pct_novo,
        })
    except Exception as e:
        row["erro"] = str(e)
    return row


def _key_from_pts(a, b):
    a = tuple(round(v, 6) for v in a)
    b = tuple(round(v, 6) for v in b)
    return (min(a, b), max(a, b))


FIELDNAMES = ["instance", "representacao", "n_polygons", "n_pares_pi", "fo_baseline", "t_baseline",
              "fo_forte", "t_forte", "fo_forte_recomputado_indep", "diff_fo_recompute",
              "ganho_pct_vs_baseline", "estrut_duplicadas", "estrut_faltando", "estrut_extras",
              "pi_checados", "pi_violacoes", "n_kicks", "n_kick_improvements", "edge_swap_evals",
              "edge_swap_improvements", "erro", "fo_pcglns", "delta_pct_novo"]


def main():
    ini, fim, saida_path = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    packing = sorted((INSTANCES_DIR / "packing").glob("*.txt"))
    separated = sorted((INSTANCES_DIR / "separated").glob("*.txt"))
    assert len(packing) == 25 and len(separated) == 25, \
        f"esperado 25+25, achei {len(packing)}+{len(separated)}"
    todas = [(fp, "packing") for fp in packing] + [(fp, "separated") for fp in separated]
    assert len(todas) == 50
    lote = todas[ini:fim]
    saida = Path(saida_path)

    ja_feito = set()
    if saida.exists():
        with open(saida) as f:
            for row in csv.DictReader(f):
                ja_feito.add((row["instance"], row["representacao"]))

    precisa_header = not saida.exists()
    with open(saida, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if precisa_header:
            w.writeheader()
        for fp, representacao in lote:
            nome = fp.stem
            if (nome, representacao) in ja_feito:
                continue
            t0 = time.time()
            row = processa_uma(fp, representacao)
            w.writerow(row)
            f.flush()
            print(f"{representacao:<10} {nome:<35} fo_forte={row.get('fo_forte')}  "
                  f"fo_pcglns={row.get('fo_pcglns')}  delta={row.get('delta_pct_novo')}  "
                  f"pi_viol={row.get('pi_violacoes')}  erro={row.get('erro')!r}  "
                  f"(wall={time.time()-t0:.2f}s)", flush=True)
    print(f"Lote [{ini}:{fim}] concluido -> {saida}")


if __name__ == "__main__":
    main()
