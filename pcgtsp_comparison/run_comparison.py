"""
run_comparison.py
==================
Orquestra a comparacao de qualidade de solucao entre a nossa heuristica
(src/heuristic_core.py, config 'original') e o PCGLNS (solver de referencia
para PCGTSP), nas MESMAS instancias (apos o mesmo pre-processamento),
convertidas via converter.py.

Para cada instancia:
  1. Roda a nossa heuristica (config 'original') -> FO_nossa, tempo, e
     verificacao de quantos pares de precedencia REAL (Pi, calculado via
     geometria) ela viola.
  2. Converte a instancia para .pcglns (mesmo pre-processamento).
  3. Roda o PCGLNS via subprocess (Julia 1.6) com um orcamento de tempo fixo.
  4. Converte o custo de tour do PCGLNS (Chebyshev escalado, so movimento em
     vazio) de volta para a MESMA unidade de FO da nossa heuristica:
        FO_pcglns = (tour_cost / SCALE) / v_air + cut_time_fixed
     onde cut_time_fixed = soma dos tempos de corte de cada contorno
     (constante, independente da ordem de corte).
  5. Verifica (independentemente, lendo o arquivo .tour) se o PCGLNS de fato
     respeitou Pi -- deve ser sempre 0 violacoes, por construcao do GTSP+SOP.
  6. Registra tudo num CSV.

Uso:
    python3 run_comparison.py <glob_de_instancias> <max_time_pcglns_seg> <saida.csv>

Configuracao do Julia (variaveis de ambiente, todas opcionais):
    LCPP_JULIA          executavel do Julia             (padrao: "julia", no PATH)
    LCPP_JULIA_CHANNEL  canal do juliaup                (padrao: "+1.6"; "none" = nao passa canal)
    LCPP_TMP_DIR        pasta de arquivos temporarios   (padrao: <tmp do sistema>/pcgtsp_run)
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
from data.load_instances import load_instance  # noqa: E402
import heuristic_core as hc  # noqa: E402
from converter import convert_instance, SCALE  # noqa: E402

# O PCGLNS roda em Julia 1.6 (a versao 1.12 trava com erro de GC neste codigo).
JULIA_BIN = os.environ.get("LCPP_JULIA", "julia")
JULIA_CHANNEL = os.environ.get("LCPP_JULIA_CHANNEL", "+1.6")
if JULIA_CHANNEL.lower() == "none":
    JULIA_CHANNEL = ""
PCGLNS_SCRIPT = str(_PROJECT_ROOT / "external_tools" / "pcglns" / "runPCGLNS.jl")
TMP_DIR = Path(os.environ.get("LCPP_TMP_DIR", Path(tempfile.gettempdir()) / "pcgtsp_run"))
TMP_DIR.mkdir(parents=True, exist_ok=True)

_ORIGINAL_CFG = {
    "root_strategy": "nearest",
    "child_order": "num_filhos",
    "mst_strategy": "standard",
    "vertex_strategy": "nearest",
    "time_limit": 60.0,
}


def _detect_precedence_positions(positions) -> List[Tuple[int, int]]:
    from converter import _detect_precedence
    return _detect_precedence(positions)


def _find_polygon_owner(seg, positions) -> int | None:
    def key(a, b):
        a = tuple(round(v, 6) for v in a)
        b = tuple(round(v, 6) for v in b)
        return (min(a, b), max(a, b))
    k = key(seg.coords[0], seg.coords[1])
    for idx, pol in enumerate(positions):
        coords = list(pol.exterior.coords)
        for i in range(len(coords) - 1):
            if key(coords[i], coords[i + 1]) == k:
                return idx
    return None


def check_our_precedence_violations(layout_original, caminho, positions, precedence_pairs) -> Tuple[int, int]:
    """Retorna (n_pares_checados, n_violacoes) para a nossa heuristica.

    Nota: segmentos de comprimento ~0 (vertices quase-duplicados na geometria de
    entrada, ruido de ponto flutuante ~1e-11, comum em cantos onde dois
    poligonos se tocam) sao ignorados aqui -- eles nao representam corte
    fisico algum (contribuem 0 pra FO de qualquer forma) mas podem colidir,
    por chave de coordenada arredondada, com o vertice-fantasma equivalente
    de OUTRO poligono que toca no mesmo canto, causando uma atribuicao de
    dono ambigua/incorreta (`_find_polygon_owner` pega o primeiro indice que
    bate) e um falso positivo de violacao de Pi. Confirmado em
    art_6_hole_pack: sem este filtro, aparecia 1 violacao (furo=8, pai=6)
    causada exatamente por esse artefato -- o par Pi real era respeitado
    (a fronteira de verdade do poligono 6, com comprimento > 0, so e cortada
    bem depois do furo 8, como esperado)."""
    primeiro, ultimo = {}, {}
    for pos_idx, (seg, tag) in enumerate(caminho):
        if tag != "c":
            continue
        if seg.length < 1e-6:
            continue
        owner = _find_polygon_owner(seg, positions)
        if owner is not None:
            primeiro.setdefault(owner, pos_idx)
            ultimo[owner] = pos_idx
    violacoes = 0
    checados = 0
    for furo_idx, pai_idx in precedence_pairs:
        if furo_idx not in ultimo or pai_idx not in primeiro:
            continue
        checados += 1
        if ultimo[furo_idx] >= primeiro[pai_idx]:
            violacoes += 1
    return checados, violacoes


def run_pcglns(pcglns_path: Path, max_time: float, out_path: Path) -> Dict:
    t0 = time.time()
    try:
        subprocess.run(
            [JULIA_BIN, *([JULIA_CHANNEL] if JULIA_CHANNEL else []), "--check-bounds=no", PCGLNS_SCRIPT,
             str(pcglns_path), f"-max_time={int(max_time)}", "-verbose=0",
             f"-output={out_path}"],
            capture_output=True, text=True, timeout=max_time + 60,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "subprocess timeout"}
    elapsed = time.time() - t0

    if not out_path.exists():
        return {"ok": False, "error": "sem arquivo de saida"}

    text = out_path.read_text()
    m_cost = re.search(r"Tour Cost\s*:\s*([\-\d.]+)", text)
    m_tour = re.search(r"Tour\s*:\s*\[([^\]]+)\]", text)
    m_time = re.search(r"Solver Time\s*:\s*([\d.]+)", text)
    if not m_cost or not m_tour:
        return {"ok": False, "error": "nao conseguiu parsear saida"}

    tour = [int(x.strip()) for x in m_tour.group(1).split(",")]
    return {
        "ok": True,
        "tour_cost_scaled": float(m_cost.group(1)),
        "tour": tour,
        "solver_time": float(m_time.group(1)) if m_time else elapsed,
        "wall_time": elapsed,
    }


def check_pcglns_precedence_violations(pcglns_path: Path, tour: List[int]) -> Tuple[int, int]:
    """Le o .pcglns (GTSP_SET_SECTION + GTSP_SET_ORDERING) e verifica se o
    tour devolvido pelo PCGLNS respeita a ordem exigida entre grupos."""
    text = pcglns_path.read_text()
    lines = text.split("\n")
    city_group, prec = {}, []
    state = None
    for line in lines:
        s = line.strip()
        if s.startswith("GTSP_SET_SECTION"):
            state = "sets"; continue
        if s.startswith("GTSP_SET_ORDERING"):
            state = "order"; continue
        if s.startswith("START_GROUP_SECTION") or s.startswith("EOF"):
            state = None; continue
        if state == "sets" and s:
            nums = [int(x) for x in s.split()]
            grp = nums[0]
            for c in nums[1:]:
                if c == -1: break
                city_group[c] = grp
        elif state == "order" and s:
            nums = [int(x) for x in s.split()]
            pred = nums[0]
            for suc in nums[1:]:
                if suc == -1: break
                prec.append((pred, suc))
    group_pos = {}
    for pos, city in enumerate(tour):
        group_pos[city_group[city]] = pos
    violacoes = 0
    for pred, suc in prec:
        if pred in group_pos and suc in group_pos and group_pos[pred] > group_pos[suc]:
            violacoes += 1
    return len(prec), violacoes


def compare_one(instance_path: Path, max_time_pcglns: float) -> Dict:
    name = instance_path.name
    result = {"instance": name}

    # ---- nossa heuristica ----
    _, layout_original = load_instance(instance_path)
    t0 = time.time()
    our = hc.run_heuristic(list(layout_original), _ORIGINAL_CFG)
    our_time = time.time() - t0

    positions = hc.clusterizar_quadtree(
        hc.resolver_intersecoes(hc.tratar_furos(list(layout_original)))
    )
    prec_pairs = _detect_precedence_positions(positions)
    n_checked, n_viol = check_our_precedence_violations(
        layout_original, our["caminho"], positions, prec_pairs
    )

    result.update({
        "n_groups": len(positions),
        "n_precedence_pairs": len(prec_pairs),
        "our_fo": our["fo_total"],
        "our_time_s": round(our_time, 3),
        "our_prec_checked": n_checked,
        "our_prec_violations": n_viol,
    })

    # ---- PCGLNS ----
    pcglns_in = TMP_DIR / f"{instance_path.stem}.pcglns"
    pcglns_out = TMP_DIR / f"{instance_path.stem}.tour"
    stats = convert_instance(instance_path, pcglns_in)
    res = run_pcglns(pcglns_in, max_time_pcglns, pcglns_out)

    if not res["ok"]:
        result.update({"pcglns_ok": False, "pcglns_error": res.get("error")})
        return result

    fo_pcglns = (res["tour_cost_scaled"] / SCALE) / hc._V_AIR + stats["cut_time_fixed"]
    n_prec_total, n_prec_viol = check_pcglns_precedence_violations(pcglns_in, res["tour"])

    result.update({
        "pcglns_ok": True,
        "pcglns_n_cities": stats["n_cities"],
        "pcglns_fo": round(fo_pcglns, 4),
        "pcglns_solver_time_s": res["solver_time"],
        "pcglns_prec_checked": n_prec_total,
        "pcglns_prec_violations": n_prec_viol,
        "gap_pct": round(100.0 * (fo_pcglns - our["fo_total"]) / our["fo_total"], 2) if our["fo_total"] else None,
    })
    return result


def main():
    import glob
    import csv

    pattern = sys.argv[1]
    max_time = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
    out_csv = sys.argv[3] if len(sys.argv) > 3 else "comparacao_pcglns.csv"

    files = sorted(Path(p) for p in glob.glob(pattern))
    print(f"{len(files)} instancias a processar, orcamento PCGLNS={max_time}s cada")

    rows = []
    for i, fp in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {fp.name} ...", flush=True)
        try:
            row = compare_one(fp, max_time)
        except Exception as e:
            row = {"instance": fp.name, "error": str(e)}
        rows.append(row)
        print(f"    {row}", flush=True)

    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResultados salvos em {out_csv}")


if __name__ == "__main__":
    main()
