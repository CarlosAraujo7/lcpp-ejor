"""
rerun_vns_geometricas.py [saida.csv] [max_segundos]
=====================================================
Roda o VNS fiel a Zhang et al. 2024 (vns_zhang_fiel.py) nas 282 instancias
sinteticas (instances/generated/) e nas 50 instancias ESICUP/Amaro2023
(instances/original/{packing,separated}/) -- os mesmos conjuntos em que a
heuristica proposta e o PCGLNS foram executados (rerun_frente1_282_FINAL.csv,
originais_50_FULL.csv), de modo que todos os metodos rodem em todas as
instancias.

Conversao de unidade (a mesma usada para as 6 instancias de Zhang): fo_vns
bruto esta em unidades RAW da matriz .pcglns; (raw/SCALE)/V_AIR +
cut_time_fixed o coloca na mesma escala de fo_forte/fo_pcglns.

Paralelizado (multiprocessing, 2 workers),
escrita incremental e resumivel por instancia (nao por seed -- cada
instancia so e' marcada como feita apos suas 10 sementes completarem,
para nao deixar uma instancia pela metade se cortado no meio).

Saida: uma linha por (instancia, seed) -- mesmo padrao de
vns_33x10_fiel_FULL.csv. A agregacao por instancia e' feita depois, num
script separado.
"""
from __future__ import annotations
import csv
import multiprocessing as mp
import os
import sys
import tempfile
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "pcgtsp_comparison"))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
N_SEEDS = 10
TIME_BUDGET = 3.0

GENERATED_DIR = _PROJECT_ROOT / "instances" / "generated"
ORIGINAL_DIR = _PROJECT_ROOT / "instances" / "original"

FIELDNAMES = ["pool", "instance", "seed", "fo_vns_raw", "fo_vns", "t",
              "shakes_aceitos", "shakes_tentados", "viol_reportado",
              "custo_recomputado_indep_raw", "viol_recomputado_indep",
              "dup_grupo", "faltando_grupo", "diff_fo_raw", "cut_time_fixed", "erro"]


def build_instance_list():
    itens = []
    for fp in sorted(GENERATED_DIR.glob("*.txt")):
        itens.append(("sintetica", fp.stem, fp))
    for fp in sorted((ORIGINAL_DIR / "packing").glob("*.txt")):
        itens.append(("esicup", f"{fp.stem}__packing", fp))
    for fp in sorted((ORIGINAL_DIR / "separated").glob("*.txt")):
        itens.append(("esicup", f"{fp.stem}__separated", fp))
    return itens


def processa_instancia(args):
    """Roda no worker: converte p/ .pcglns, roda 10 sementes de VNS, verifica."""
    pool, nome, fp = args
    import heuristic_core as hc
    from converter import convert_instance, SCALE
    from heuristica_direta_pcgtsp import parse_pcglns, _dist, _viol_ciclica
    from vns_zhang_fiel import resolver_com_vns_fiel

    linhas = []
    try:
        tmp_dir = Path(os.environ.get("LCPP_TMP_DIR", Path(tempfile.gettempdir()) / "pcgtsp_run")) / "vns_geo"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        pcglns_path = tmp_dir / f"{pool}_{nome}.pcglns"
        stats = convert_instance(fp, pcglns_path)
        cut_time_fixed = stats["cut_time_fixed"]
        parsed = parse_pcglns(str(pcglns_path))
        matrix = parsed["matrix"]

        for seed in range(N_SEEDS):
            t0 = time.time()
            r = resolver_com_vns_fiel(str(pcglns_path), time_budget=TIME_BUDGET, seed=seed)
            tour = r["tour"]
            custo_indep_raw = sum(
                _dist(matrix, tour[i], tour[(i + 1) % len(tour)]) for i in range(len(tour))
            )
            viol_indep = _viol_ciclica(tour, parsed["city_group"], parsed["pares"], parsed["start_group"])
            grupos_visitados = [parsed["city_group"][c] for c in tour]
            dup = len(grupos_visitados) != len(set(grupos_visitados))
            faltando = len(set(grupos_visitados)) != len(parsed["group_cities"])
            fo_vns_raw = r["fo"]
            fo_vns_tempo = (fo_vns_raw / SCALE) / hc._V_AIR + cut_time_fixed
            linhas.append({
                "pool": pool, "instance": nome, "seed": seed,
                "fo_vns_raw": fo_vns_raw, "fo_vns": fo_vns_tempo, "t": r["t"],
                "shakes_aceitos": r["shakes_aceitos"], "shakes_tentados": r["shakes_tentados"],
                "viol_reportado": r["viol"], "custo_recomputado_indep_raw": custo_indep_raw,
                "viol_recomputado_indep": viol_indep, "dup_grupo": dup, "faltando_grupo": faltando,
                "diff_fo_raw": abs(fo_vns_raw - custo_indep_raw), "cut_time_fixed": cut_time_fixed,
                "erro": "",
            })
    except Exception as e:
        linhas = [{
            "pool": pool, "instance": nome, "seed": s, "fo_vns_raw": "", "fo_vns": "", "t": "",
            "shakes_aceitos": "", "shakes_tentados": "", "viol_reportado": "",
            "custo_recomputado_indep_raw": "", "viol_recomputado_indep": "",
            "dup_grupo": "", "faltando_grupo": "", "diff_fo_raw": "", "cut_time_fixed": "",
            "erro": str(e),
        } for s in range(N_SEEDS)]
    return nome, linhas


def main():
    saida_path = sys.argv[1] if len(sys.argv) > 1 else "vns_282_50_geometricas.csv"
    max_segundos = float(sys.argv[2]) if len(sys.argv) > 2 else 150.0
    saida = Path(saida_path)

    todas = build_instance_list()

    ja_feito = set()
    if saida.exists():
        contagem = {}
        with open(saida) as f:
            for row in csv.DictReader(f):
                contagem[row["instance"]] = contagem.get(row["instance"], 0) + 1
        # so considera "feito" se tem as 10 sementes (senao pode ter sido cortado)
        ja_feito = {nome for nome, c in contagem.items() if c >= N_SEEDS}

    pendentes = [(pool, nome, fp) for pool, nome, fp in todas if nome not in ja_feito]
    print(f"Total={len(todas)}  ja_feito={len(ja_feito)}  pendentes={len(pendentes)}", flush=True)
    if not pendentes:
        print("Nada a fazer.")
        return

    precisa_header = not saida.exists()
    t_inicio = time.time()
    n_processadas = 0

    with open(saida, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if precisa_header:
            w.writeheader()
            f.flush()

        with mp.Pool(2) as pool_mp:
            for nome, linhas in pool_mp.imap_unordered(processa_instancia, pendentes):
                for linha in linhas:
                    w.writerow(linha)
                f.flush()
                n_processadas += 1
                erro = linhas[0]["erro"]
                if erro:
                    print(f"[{n_processadas}/{len(pendentes)}] {nome:<40} ERRO: {erro}", flush=True)
                else:
                    fos = [l["fo_vns"] for l in linhas]
                    print(f"[{n_processadas}/{len(pendentes)}] {nome:<40} "
                          f"fo_vns(min/media)={min(fos):.2f}/{sum(fos)/len(fos):.2f}  "
                          f"viol={sum(l['viol_recomputado_indep'] for l in linhas)}", flush=True)
                if time.time() - t_inicio > max_segundos:
                    print(f"Tempo limite do lote atingido ({max_segundos}s), parando para retomar depois.",
                          flush=True)
                    pool_mp.terminate()
                    break

    print(f"Lote concluido -> {saida} ({n_processadas} instancias processadas nesta chamada)")


if __name__ == "__main__":
    main()
