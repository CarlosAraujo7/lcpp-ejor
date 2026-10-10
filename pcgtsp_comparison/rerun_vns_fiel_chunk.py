"""
rerun_vns_fiel_chunk.py <indice_inicial> <indice_final> <arquivo_saida.csv>
=============================================================================
Roda o VNS FIEL a Zhang et al. 2024 (vns_zhang_fiel.py) nas 33 instancias
PCGTSPLIB x 10 seeds (orcamento de 3s por execucao, com verificacao
independente do tour: violacoes de Pi, grupos duplicados/faltando e FO
recomputada), gravando uma linha por (instancia, seed). Gravacao incremental
e resumivel. Saida de referencia: vns_33x10_fiel_FULL.csv.

Uso: python rerun_vns_fiel_chunk.py <ini> <fim> <arquivo_saida.csv>
"""
import sys
import csv
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pcgtsp_comparison"))
from heuristica_direta_pcgtsp import parse_pcglns, _dist, _viol_ciclica
from vns_zhang_fiel import resolver_com_vns_fiel

PASTA = (Path(__file__).resolve().parent.parent / "external_tools/pcglns/PCGLNS_PCGTSP")

TODAS = ["ESC07", "ESC12", "ESC25", "ESC47", "ESC63", "ESC78", "br17.10", "br17.12",
         "ft53.1", "ft53.2", "ft53.3", "ft53.4", "ft70.1", "ft70.2", "ft70.3", "ft70.4",
         "kro124p.1", "kro124p.2", "kro124p.3", "kro124p.4", "p43.1", "p43.2", "p43.3", "p43.4",
         "prob.100", "prob.42", "rbg048a", "rbg050c", "rbg109a",
         "ry48p.1", "ry48p.2", "ry48p.3", "ry48p.4"]

ini, fim, saida_path = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
instancias = TODAS[ini:fim]
saida = Path(saida_path)

fieldnames = ["instance", "seed", "fo_vns", "t", "shakes_aceitos", "shakes_tentados",
              "viol_reportado", "custo_recomputado_indep", "viol_recomputado_indep",
              "dup_grupo", "faltando_grupo", "diff_fo"]

ja_feito = set()
if saida.exists():
    with open(saida) as f:
        for row in csv.DictReader(f):
            ja_feito.add((row["instance"], row["seed"]))

precisa_header = not saida.exists()
with open(saida, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    if precisa_header:
        w.writeheader()
    for nome in instancias:
        fp = PASTA / f"{nome}.pcglns"
        parsed = parse_pcglns(str(fp))
        for seed in range(10):
            if (nome, str(seed)) in ja_feito:
                continue
            t0 = time.time()
            r = resolver_com_vns_fiel(str(fp), time_budget=3.0, seed=seed)
            tour = r["tour"]
            matrix = parsed["matrix"]
            custo_indep = sum(_dist(matrix, tour[i], tour[(i + 1) % len(tour)]) for i in range(len(tour)))
            viol_indep = _viol_ciclica(tour, parsed["city_group"], parsed["pares"], parsed["start_group"])
            grupos_visitados = [parsed["city_group"][c] for c in tour]
            dup = len(grupos_visitados) != len(set(grupos_visitados))
            faltando = len(set(grupos_visitados)) != len(parsed["group_cities"])
            row = {
                "instance": nome, "seed": seed, "fo_vns": r["fo"], "t": r["t"],
                "shakes_aceitos": r["shakes_aceitos"], "shakes_tentados": r["shakes_tentados"],
                "viol_reportado": r["viol"], "custo_recomputado_indep": custo_indep,
                "viol_recomputado_indep": viol_indep, "dup_grupo": dup, "faltando_grupo": faltando,
                "diff_fo": abs(r["fo"] - custo_indep),
            }
            w.writerow(row)
            f.flush()
            print(f"{nome:<12} seed={seed}  fo={r['fo']:.1f}  viol={r['viol']}/{viol_indep}  "
                  f"dup={dup}  faltando={faltando}  t={r['t']:.2f}s  (wall={time.time()-t0:.2f}s)", flush=True)
print(f"Chunk [{ini}:{fim}] concluido -> {saida}")
