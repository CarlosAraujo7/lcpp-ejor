"""
comparacao_tres_vias.py
=========================
Comparação completa em 3 vias, nas instâncias com furo:

  A) nossa heurística ORIGINAL   (enforce_precedence=False) -- rápida, mas
     sem garantia de Π (ver nota no topo de src/heuristic_core.py).
  B) nossa heurística COM Π      (enforce_precedence=True)  -- garante Π,
     custo incremental quase nulo.
  C) PCGLNS (solver de referência p/ PCGTSP, Julia)          -- garante Π
     por construção (é uma restrição rígida do próprio PCGTSP).

Reaproveita run_comparison.py (conversão + subprocess Julia) e
heuristic_core.py (heurística nossa, as duas variantes).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
from data.load_instances import load_instance  # noqa: E402
import heuristic_core as hc  # noqa: E402
from converter import convert_instance, SCALE  # noqa: E402
from run_comparison import run_pcglns, check_pcglns_precedence_violations, TMP_DIR  # noqa: E402


def _seg_key(a, b):
    a = tuple(round(v, 6) for v in a)
    b = tuple(round(v, 6) for v in b)
    return (min(a, b), max(a, b))


def find_owner(seg, positions):
    k = _seg_key(seg.coords[0], seg.coords[1])
    for idx, pol in enumerate(positions):
        coords = list(pol.exterior.coords)
        for i in range(len(coords) - 1):
            if _seg_key(coords[i], coords[i + 1]) == k:
                return idx
    return None


def contar_violacoes(caminho, positions, pares):
    primeiro, ultimo = {}, {}
    for pos_idx, (seg, tag) in enumerate(caminho):
        if tag != "c":
            continue
        owner = find_owner(seg, positions)
        if owner is not None:
            primeiro.setdefault(owner, pos_idx)
            ultimo[owner] = pos_idx
    viol, checados = 0, 0
    for furo, pai in pares:
        if furo not in ultimo or pai not in primeiro:
            continue
        checados += 1
        if ultimo[furo] >= primeiro[pai]:
            viol += 1
    return checados, viol


def comparar(fp, max_time_pcglns=15.0):
    name, layout = load_instance(fp)
    positions = hc.clusterizar_quadtree(hc.resolver_intersecoes(hc.tratar_furos(list(layout))))
    pares = hc._detectar_precedencia(positions)

    # A) original
    t0 = time.time()
    r_a = hc.run_heuristic(list(layout), {**hc._DEFAULT_CONFIG, "enforce_precedence": False, "time_limit": 60.0})
    t_a = time.time() - t0
    chk_a, v_a = contar_violacoes(r_a["caminho"], positions, pares)

    # B) corrigida
    t0 = time.time()
    r_b = hc.run_heuristic(list(layout), {**hc._DEFAULT_CONFIG, "enforce_precedence": True, "time_limit": 60.0})
    t_b = time.time() - t0
    chk_b, v_b = contar_violacoes(r_b["caminho"], positions, pares)

    # C) PCGLNS
    pcglns_in = TMP_DIR / f"{Path(fp).stem}.pcglns"
    pcglns_out = TMP_DIR / f"{Path(fp).stem}.tour"
    stats = convert_instance(fp, pcglns_in)
    res = run_pcglns(pcglns_in, max_time_pcglns, pcglns_out)
    if res["ok"]:
        fo_c = (res["tour_cost_scaled"] / SCALE) / hc._V_AIR + stats["cut_time_fixed"]
        _, v_c = check_pcglns_precedence_violations(pcglns_in, res["tour"])
        t_c = res["solver_time"]
    else:
        fo_c, v_c, t_c = None, None, None

    print(f"\n{Path(fp).name}  ({len(pares)} pares Π)")
    print(f"  A) nossa ORIGINAL   FO={r_a['fo_total']:>9.4f}  viol={v_a:>2}/{chk_a}  t={t_a:.2f}s")
    print(f"  B) nossa CORRIGIDA  FO={r_b['fo_total']:>9.4f}  viol={v_b:>2}/{chk_b}  t={t_b:.2f}s  "
          f"(delta vs A: {100*(r_b['fo_total']-r_a['fo_total'])/r_a['fo_total']:+.2f}%)")
    if fo_c is not None:
        print(f"  C) PCGLNS           FO={fo_c:>9.4f}  viol={v_c:>2}/{stats['n_precedence_pairs']}  t={t_c:.2f}s  "
              f"(delta vs B: {100*(fo_c-r_b['fo_total'])/r_b['fo_total']:+.2f}%)")
    else:
        print("  C) PCGLNS           FALHOU")


if __name__ == "__main__":
    for fp in sys.argv[1:]:
        comparar(str(_PROJECT_ROOT / fp) if not fp.startswith("/") else fp)
