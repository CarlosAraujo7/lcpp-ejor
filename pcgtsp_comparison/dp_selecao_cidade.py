"""
Busca local (fase 1): substitui a escolha GULOSA de cidade representante por
grupo por uma escolha OTIMA via DP, mantendo a ORDEM DE GRUPOS fixa (a mesma
que ja respeita Pi via mecanismo LCA). Como a ordem de grupos nao muda, Pi
continua satisfeita automaticamente -- essa fase nao pode introduzir violacao.

Formulacao: dado grupos g_1..g_k na ordem fixa, escolher c_i em candidatos(g_i)
minimizando soma_{i=1}^{k-1} dist(c_i,c_{i+1}) + dist(c_k,c_1) (fecha o ciclo).
Isso e' um shortest-path em grafo em camadas, exceto pelo termo de fechamento
que acopla a primeira e ultima escolha -- resolvido iterando sobre todas as
candidatas da primeira posicao como "ancora" (grupos tem poucas cidades, ~5
em media, entao e' barato).
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pcgtsp_comparison"))
import networkx as nx
from heuristica_direta_pcgtsp import (
    parse_pcglns, construir_grafo_grupos, construir_arvore_grupos,
    ordem_de_visita_grupos, escolher_cidades_e_fo, _dist, INF, _viol_ciclica,
)

PASTA = (Path(__file__).resolve().parent.parent / "external_tools/pcglns/PCGLNS_PCGTSP")


def escolher_cidades_e_fo_dp(ordem_grupos, parsed):
    group_cities = parsed["group_cities"]
    matrix = parsed["matrix"]
    k = len(ordem_grupos)
    if k == 0:
        return [], 0.0
    cands = [group_cities[g] for g in ordem_grupos]
    if k == 1:
        return [cands[0][0]], 0.0

    melhor_fo_global = INF
    melhor_caminho_global = None

    for c1 in cands[0]:
        prev_layer = {c1: 0.0}
        parent = [None] + [dict() for _ in range(k - 1)]
        for i in range(1, k):
            cur_layer = {}
            par_i = parent[i]
            for c in cands[i]:
                melhor, melhor_prev = INF, None
                for cp, custo_prev in prev_layer.items():
                    d = _dist(matrix, cp, c)
                    if d == INF:
                        continue
                    total = custo_prev + d
                    if total < melhor:
                        melhor, melhor_prev = total, cp
                if melhor_prev is not None:
                    cur_layer[c] = melhor
                    par_i[c] = melhor_prev
            prev_layer = cur_layer
            if not prev_layer:
                break

        for c_last, custo in prev_layer.items():
            fecho = _dist(matrix, c_last, c1)
            if fecho == INF:
                continue
            total = custo + fecho
            if total < melhor_fo_global:
                melhor_fo_global = total
                caminho = [c_last]
                for i in range(k - 1, 0, -1):
                    caminho.append(parent[i][caminho[-1]])
                caminho.reverse()
                melhor_caminho_global = caminho

    if melhor_caminho_global is None:
        raise RuntimeError(
            f"DP nao encontrou NENHUMA realizacao de custo finito para esta ordem de grupos "
            f"({len(ordem_grupos)} grupos) -- indica que a busca local produziu uma ordem sem "
            f"nenhum par de cidades finito entre grupos consecutivos em algum ponto. Nao deve "
            f"cair em fallback silencioso."
        )

    return melhor_caminho_global, melhor_fo_global


def resolver_com_dp(path):
    t0 = time.time()
    parsed = parse_pcglns(path)
    G = construir_grafo_grupos(parsed)
    fo_total, tour_total = 0.0, []
    viol = 0
    for comp in nx.connected_components(G):
        sub = G.subgraph(comp)
        mst = nx.minimum_spanning_tree(sub, weight="weight")
        raiz = parsed["start_group"] if parsed["start_group"] in comp else next(iter(comp))
        arvore = construir_arvore_grupos(mst, raiz)
        pares_comp = [(f, p) for f, p in parsed["pares"] if f in comp and p in comp]
        ordem_grupos = ordem_de_visita_grupos(arvore, raiz, pares_comp)
        cidades, fo = escolher_cidades_e_fo_dp(ordem_grupos, parsed)
        fo_total += fo
        tour_total.extend(cidades)
        # checagem por componente, rotacionada pra raiz (ver
        # _ordem_e_factivel_ciclica).
        viol += _viol_ciclica(cidades, parsed["city_group"], pares_comp, raiz)
    t_total = time.time() - t0

    return {"fo": fo_total, "tour": tour_total, "viol": viol, "t": t_total}


if __name__ == "__main__":
    from heuristica_direta_pcgtsp import resolver_direto
    print(f"{'instancia':<15} {'fo_guloso':>12} {'fo_dp':>12} {'ganho%':>8} {'viol_dp':>7} {'t_dp':>7}")
    ganhos = []
    for fp in sorted(PASTA.glob("*.pcglns")):
        r_guloso = resolver_direto(str(fp))
        r_dp = resolver_com_dp(str(fp))
        ganho = 100 * (r_guloso["fo"] - r_dp["fo"]) / r_guloso["fo"] if r_guloso["fo"] else 0.0
        ganhos.append(ganho)
        flag = "" if r_dp["viol"] == 0 else "  <-- VIOLACAO!!"
        print(f"{fp.stem:<15} {r_guloso['fo']:>12.1f} {r_dp['fo']:>12.1f} {ganho:>7.2f}% {r_dp['viol']:>7} {r_dp['t']:>6.3f}s{flag}")
    print(f"\nGanho medio da DP sobre o guloso: {sum(ganhos)/len(ganhos):.2f}%  (min={min(ganhos):.2f}%, max={max(ganhos):.2f}%)")
