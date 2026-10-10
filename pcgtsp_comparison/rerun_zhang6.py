"""
rerun_zhang6.py
================
Roda os 3 metodos (heuristica proposta, PCGLNS, VNS fiel ao Zhang et al.
2024) sobre as 6 instancias reais do capsule Code Ocean do Zhang2024
(instances/zhang2024/lcpp/*.txt, convertidas de .pack via
convert_zhang_pack.py), TODAS pelo nosso proprio pipeline de construcao
(candidatos = vertices de poligono, distancia Chebyshev*SCALE, mesmo
pre-processamento hc.tratar_furos+resolver_intersecoes+clusterizar_quadtree
usado em todo o resto do projeto).

Isso evita o problema de comparar numeros publicados de papers diferentes
(escalas de coordenada e construcao de candidatos incompativeis, ja
verificado nao-comparavel) -- aqui os 3 metodos rodam
sobre EXATAMENTE os mesmos dados de distancia, entao a comparacao e justa
por construcao.

Mesmo orcamento de tempo (3s) e protocolo de verificacao independente das
demais execucoes (10 seeds para heuristica e para VNS).

Uso: python3 rerun_zhang6.py <saida.csv>
"""
from __future__ import annotations
import csv
import sys
import time
from collections import Counter
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "pcgtsp_comparison"))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
from data.load_instances import load_instance  # noqa: E402
import heuristic_core as hc  # noqa: E402
from converter import convert_instance, SCALE  # noqa: E402
import busca_local_arvore as bla  # noqa: E402
from run_comparison import (  # noqa: E402
    _detect_precedence_positions, run_pcglns, TMP_DIR,
    check_our_precedence_violations,
)
from heuristica_direta_pcgtsp import parse_pcglns, _dist, _viol_ciclica  # noqa: E402
from vns_zhang_fiel import resolver_com_vns_fiel  # noqa: E402

INSTANCES_DIR = _PROJECT_ROOT / "instances" / "zhang2024" / "lcpp"
TIME_BUDGET = 3.0
PCGLNS_MAX_TIME = 20.0
N_SEEDS = 10

INSTANCIAS = ["blaz2_s", "dighe1", "dighe1_s", "dighe2", "fu", "poly1a"]


def _key_from_pts(a, b):
    a = tuple(round(v, 6) for v in a)
    b = tuple(round(v, 6) for v in b)
    return (min(a, b), max(a, b))


def roda_heuristica_e_pcglns(fp: Path) -> dict:
    nome = fp.stem
    row = {"instance": nome, "erro": ""}
    _, layout_original = load_instance(fp)

    _ORIGINAL_CFG = {
        "root_strategy": "nearest", "child_order": "num_filhos",
        "mst_strategy": "standard", "vertex_strategy": "nearest", "time_limit": 60.0,
    }
    t0 = time.time()
    baseline = hc.run_heuristic(list(layout_original), _ORIGINAL_CFG)
    t_baseline = time.time() - t0
    fo_baseline = baseline["fo_total"]

    positions = hc.clusterizar_quadtree(
        hc.resolver_intersecoes(hc.tratar_furos(list(layout_original)))
    )
    pares_pi = _detect_precedence_positions(positions)

    G_full_raiz = hc._build_full_graph(positions)
    raiz = hc.escolher_raiz_nearest(G_full_raiz, positions)

    fos_forte, ts_forte, viols = [], [], []
    melhor = None
    for seed in range(N_SEEDS):
        t0 = time.time()
        forte = bla.busca_local_forte(positions, raiz, pares_pi, time_limit=TIME_BUDGET, seed=seed)
        t_forte = time.time() - t0
        fo_forte = forte["fo_total"]
        fo_recompute = hc.calcular_fo(forte["caminho"])
        diff_fo = abs(fo_recompute - fo_forte)

        cortes = [seg for seg, tag in forte["caminho"] if tag == "c"]
        contagem = Counter(_key_from_pts(*s.coords) for s in cortes)
        esperadas = Counter()
        for pol in positions:
            coords = list(pol.exterior.coords)
            for i in range(len(coords) - 1):
                esperadas[_key_from_pts(coords[i], coords[i + 1])] += 1
        duplicadas = sum(1 for k, v in contagem.items() if v > 1)
        faltando = sum(1 for k in esperadas if k not in contagem)
        extras = sum(1 for k in contagem if k not in esperadas)

        pi_checados, pi_violacoes = check_our_precedence_violations(
            layout_original, forte["caminho"], positions, pares_pi
        )

        fos_forte.append(fo_forte)
        ts_forte.append(t_forte)
        viols.append((duplicadas, faltando, extras, diff_fo, pi_violacoes))
        if melhor is None or fo_forte < melhor:
            melhor = fo_forte

    fo_forte_media = sum(fos_forte) / len(fos_forte)
    max_diff_fo = max(v[3] for v in viols)
    total_dup = sum(v[0] for v in viols)
    total_falt = sum(v[1] for v in viols)
    total_extra = sum(v[2] for v in viols)
    total_pi_viol = sum(v[4] for v in viols)

    pcglns_in = TMP_DIR / f"zhang6_{nome}.pcglns"
    pcglns_out = TMP_DIR / f"zhang6_{nome}.tour"
    stats = convert_instance(fp, pcglns_in)
    res = run_pcglns(pcglns_in, PCGLNS_MAX_TIME, pcglns_out)
    if not res["ok"]:
        row["erro"] = f"pcglns: {res.get('error')}"
        fo_pcglns = None
    else:
        fo_pcglns = (res["tour_cost_scaled"] / SCALE) / hc._V_AIR + stats["cut_time_fixed"]

    row.update({
        "n_polygons": len(positions),
        "n_pares_pi": len(pares_pi),
        "fo_baseline": fo_baseline,
        "t_baseline": round(t_baseline, 3),
        "fo_forte_min": melhor,
        "fo_forte_media": fo_forte_media,
        "fo_forte_max": max(fos_forte),
        "t_forte_media": round(sum(ts_forte) / len(ts_forte), 3),
        "max_diff_fo_recompute": max_diff_fo,
        "estrut_duplicadas": total_dup,
        "estrut_faltando": total_falt,
        "estrut_extras": total_extra,
        "pi_violacoes_total": total_pi_viol,
        "fo_pcglns": fo_pcglns,
        "pcglns_path": str(pcglns_in),
        "cut_time_fixed": stats["cut_time_fixed"],
    })
    return row


def roda_vns_fiel(pcglns_path: str, nome: str, cut_time_fixed: float) -> list:
    """
    IMPORTANTE: r['fo'] (e o custo recomputado independente) vem em unidades
    RAW da matriz .pcglns (SCALE=10 * Chebyshev), a mesma base de
    res['tour_cost_scaled'] do PCGLNS ANTES da conversao aplicada em
    roda_heuristica_e_pcglns (fo_pcglns = (raw/SCALE)/V_AIR + cut_time_fixed).
    Sem essa mesma conversao aqui, fo_vns fica numa escala totalmente
    diferente de fo_forte/fo_pcglns (raw vs. tempo total) -- aplicamos a
    MESMA formula para manter os 3 metodos comparaveis.
    """
    parsed = parse_pcglns(pcglns_path)
    linhas = []
    for seed in range(N_SEEDS):
        r = resolver_com_vns_fiel(pcglns_path, time_budget=TIME_BUDGET, seed=seed)
        tour = r["tour"]
        matrix = parsed["matrix"]
        custo_indep_raw = sum(_dist(matrix, tour[i], tour[(i + 1) % len(tour)]) for i in range(len(tour)))
        viol_indep = _viol_ciclica(tour, parsed["city_group"], parsed["pares"], parsed["start_group"])
        grupos_visitados = [parsed["city_group"][c] for c in tour]
        dup = len(grupos_visitados) != len(set(grupos_visitados))
        faltando = len(set(grupos_visitados)) != len(parsed["group_cities"])
        fo_vns_raw = r["fo"]
        fo_vns_tempo = (fo_vns_raw / SCALE) / hc._V_AIR + cut_time_fixed
        linhas.append({
            "instance": nome, "seed": seed, "fo_vns_raw": fo_vns_raw, "fo_vns": fo_vns_tempo,
            "viol_indep": viol_indep, "dup_grupo": dup, "faltando_grupo": faltando,
            "diff_fo": abs(fo_vns_raw - custo_indep_raw),
        })
    return linhas


FIELDNAMES = ["instance", "n_polygons", "n_pares_pi", "fo_baseline", "t_baseline",
              "fo_forte_min", "fo_forte_media", "fo_forte_max", "t_forte_media",
              "max_diff_fo_recompute", "estrut_duplicadas", "estrut_faltando", "estrut_extras",
              "pi_violacoes_total", "fo_pcglns",
              "fo_vns_min", "fo_vns_media", "fo_vns_max", "vns_viol_total",
              "vns_dup_total", "vns_falt_total", "vns_max_diff_fo", "erro"]


def main():
    # uso: rerun_zhang6.py <saida.csv> [nome1 nome2 ...]
    # sem nomes extras -> roda todas as INSTANCIAS que ainda nao estao no csv
    saida_path = sys.argv[1]
    saida = Path(saida_path)
    pedidas = sys.argv[2:] if len(sys.argv) > 2 else INSTANCIAS

    ja_feito = set()
    if saida.exists():
        with open(saida) as f:
            for row in csv.DictReader(f):
                ja_feito.add(row["instance"])

    precisa_header = not saida.exists()
    with open(saida, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if precisa_header:
            w.writeheader()
        for nome in pedidas:
            if nome in ja_feito:
                print(f"{nome:<10} ja feito, pulando", flush=True)
                continue
            fp = INSTANCES_DIR / f"{nome}.txt"
            t0 = time.time()
            row = roda_heuristica_e_pcglns(fp)
            vns_linhas = roda_vns_fiel(row["pcglns_path"], nome, row["cut_time_fixed"])
            fos_vns = [l["fo_vns"] for l in vns_linhas]
            row["fo_vns_min"] = min(fos_vns)
            row["fo_vns_media"] = sum(fos_vns) / len(fos_vns)
            row["fo_vns_max"] = max(fos_vns)
            row["vns_viol_total"] = sum(l["viol_indep"] for l in vns_linhas)
            row["vns_dup_total"] = sum(1 for l in vns_linhas if l["dup_grupo"])
            row["vns_falt_total"] = sum(1 for l in vns_linhas if l["faltando_grupo"])
            row["vns_max_diff_fo"] = max(l["diff_fo"] for l in vns_linhas)
            w.writerow(row)
            f.flush()
            print(f"{nome:<10} n_poly={row['n_polygons']:>3}  "
                  f"fo_heur(min/media)={row['fo_forte_min']:.2f}/{row['fo_forte_media']:.2f}  "
                  f"fo_pcglns={row['fo_pcglns']}  "
                  f"fo_vns(min/media)={row['fo_vns_min']:.2f}/{row['fo_vns_media']:.2f}  "
                  f"estrut_viol={row['estrut_duplicadas']+row['estrut_faltando']+row['estrut_extras']}  "
                  f"pi_viol={row['pi_violacoes_total']}  vns_viol={row['vns_viol_total']}  "
                  f"(wall={time.time()-t0:.1f}s)", flush=True)
    print(f"Concluido -> {saida}")


if __name__ == "__main__":
    main()
