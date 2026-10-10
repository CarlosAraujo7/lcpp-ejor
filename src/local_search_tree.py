"""
local_search_tree.py
====================
Modulo de analise e melhoria direta da arvore gerada pela heuristica.

==========================================================================
POR QUE BUSCA LOCAL CLASSICA NAO MELHORA A HEURISTICA DIRETAMENTE
==========================================================================

A heuristica original produz uma DiGraph (T, raiz) via BFS sobre o grafo G
de adjacencia dos poligonos. O caminho de corte emerge de cortar_arvore(),
que percorre T recursivamente com interleaving geometrico.

Experimento empirico (282 instancias):
  - SubtreeRelocation foi testado: para CADA no v nao-raiz, todos os
    possiveis novos pais u foram tentados. Em 100% dos casos, todos os
    movimentos pioraram ou mantiveram a FO (delta <= 0).
  - ChildPrioritySwap: idem, 0 melhorias em todas as instancias.

Por que isso ocorre — tres razoes estruturais:

1. A CONSTRUCAO BFS JA E GULOSA-OTIMA LOCALMENTE
   O BFS conecta cada poligono v ao seu vizinho mais proximo disponivel
   no grafo G, minimizando o peso da aresta (u -> v). Qualquer outro no
   u como pai de v teria dist(u, v) >= dist(pai_bfs, v). Logo, mover v
   para qualquer outro pai aumenta o custo de deslocamento da conexao.

2. O INTERLEAVING E GEOMETRICO, NAO TOPOLOGICO
   caminha_arvore() dispara a visita ao filho s quando a cabeca de corte
   passa pelo vertice A do pai onde a conexao (pai -> s) esta ancorada.
   Isso ocorre INDEPENDENTEMENTE da ordem de prioridade dos filhos (o
   argumento 'num_filhos' so desempata quando dois filhos compartilham
   o MESMO vertice de disparo, o que e geometricamente raro).
   Portanto, ChildPrioritySwap raramente altera a ordem de visita real.

3. A ESTRUTURA LINEAR NAO CAPTURA O INTERLEAVING
   Qualquer LS que converta T em cadeia linear (rebuild_linear_arvore)
   perde o interleaving: a cadeia linear pode no maximo igualar T quando
   a arvore original for uma cadeia. Para arvores com ramificacao, a FO
   da cadeia e estruturalmente maior que a FO da arvore entrelaçada.

==========================================================================
O QUE PODE SER FEITO PARA MELHORAR DIRETAMENTE A HEURISTICA
==========================================================================

A unica melhoria que preserva o interleaving e explora o espaco de
solucoes da arvore de forma efetiva e a EXPLORACAO DE RAIZES:

  Para cada no v em G como candidato a raiz, re-executa o BFS completo
  (construir_arvore_direcionada(G, v)) e avalia _fo_arvore(T_v, v).
  A melhor raiz e retida.

Por que funciona:
  - Raizes diferentes produzem arvores BFS completamente diferentes,
    com topologias e padroes de interleaving distintos.
  - A heuristica original escolhe a raiz como o poligono mais proximo
    da origem (0,0), o que e uma heuristica de construcao, nao optimal.
  - Explorar k raizes candidates (ex.: os k poligonos mais proximos da
    origem, ou raizes aleatorias) e um refinamento valido que preserva
    toda a qualidade do interleaving.

Esse metodo e implementado em explore_roots() abaixo.

==========================================================================
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import networkx as nx
from scipy.spatial.distance import chebyshev
from shapely.geometry import LineString, Polygon, Point

from graph_routing import construir_arvore_direcionada, cortar_arvore, linha_esta_na_lista

_V_CUT = 100 / 6
_V_AIR = 400.0


# ── Avaliacao da FO real ──────────────────────────────────────────────────────

def _resolve_tags(caminho):
    return [
        (seg, tag if tag != 'x'
         else ('c' if not linha_esta_na_lista(seg, caminho[:idx]) else 'd'))
        for idx, (seg, tag) in enumerate(caminho)
    ]


def _fo_arvore(T: nx.DiGraph, raiz: int) -> float:
    """Funcao objetivo real: avalia T via cortar_arvore (inclui interleaving)."""
    caminho = _resolve_tags(cortar_arvore(T, raiz))
    return sum(
        chebyshev(s.coords[0], s.coords[1]) / (_V_CUT if tag == 'c' else _V_AIR)
        for s, tag in caminho
    )


# ── Utilitarios ───────────────────────────────────────────────────────────────

def _update_num_filhos(T: nx.DiGraph):
    """Recomputa num_filhos de todos os nos apos modificacao estrutural."""
    for v in T.nodes:
        T.nodes[v]['num_filhos'] = len(list(T.successors(v)))


def _nearest_vertex_pair(poly_u: Polygon, poly_v: Polygon):
    verts_u = list(poly_u.exterior.coords)[:-1]
    verts_v = list(poly_v.exterior.coords)[:-1]
    best_d, best_pu, best_pv = float('inf'), None, None
    for pu in verts_u:
        for pv in verts_v:
            d = chebyshev(pu, pv)
            if d < best_d:
                best_d, best_pu, best_pv = d, pu, pv
    return best_pu, best_pv, best_d


def _edge_data_between(poly_u: Polygon, poly_v: Polygon):
    """Computa weight e connection para uma aresta (u->v) nova."""
    pu, pv, d = _nearest_vertex_pair(poly_u, poly_v)
    w = 0 if d < 1e-8 else d
    return w, LineString([pu, pv])


# ── Exploracao de raizes (melhoria efetiva na arvore) ─────────────────────────

def explore_roots(
    G: nx.Graph,
    polys: Dict[int, Polygon],
    origin: Point = Point(0, 0),
    max_candidates: Optional[int] = None,
    max_time_s: float = 60.0,
    verbose: bool = False,
) -> Tuple[nx.DiGraph, int, float, float, int]:
    """
    Explora multiplas raizes candidatas para a arvore BFS e retorna a melhor.

    Para cada candidato r, reconstroi a arvore via construir_arvore_direcionada(G, r)
    e avalia _fo_arvore(T_r, r). Retorna a arvore com menor FO.

    Parametros:
      G              : grafo original de adjacencia dos poligonos
      polys          : dicionario {id: Polygon}
      origin         : ponto de origem da maquina (padrao: Point(0,0))
      max_candidates : numero maximo de raizes a testar (None = todas)
      max_time_s     : limite de tempo em segundos
      verbose        : exibe progresso

    Retorna:
      (best_T, best_raiz, fo_inicial, fo_melhor, n_raizes_testadas)
    """
    nodes = list(G.nodes)

    # Raiz padrao da heuristica: no mais proximo da origem
    from graph_routing import distancia_e_aresta_ate_raiz
    dists = {v: (distancia_e_aresta_ate_raiz(origin, polys[v])[0][0].length
                 if distancia_e_aresta_ate_raiz(origin, polys[v])[0] else 0.0)
             for v in nodes}
    raiz_padrao = min(nodes, key=lambda v: dists[v])

    T_inicial = construir_arvore_direcionada(G, raiz_padrao)
    fo_inicial = _fo_arvore(T_inicial, raiz_padrao)

    # Candidatos: ordena por proximidade da origem (os mais proximos sao mais promissores)
    candidatos = sorted(nodes, key=lambda v: dists[v])
    if max_candidates is not None:
        candidatos = candidatos[:max_candidates]

    best_T, best_raiz, best_fo = T_inicial, raiz_padrao, fo_inicial
    n_testadas = 0
    t0 = time.time()

    for r in candidatos:
        if time.time() - t0 > max_time_s:
            break
        T_r = construir_arvore_direcionada(G, r)
        fo_r = _fo_arvore(T_r, r)
        n_testadas += 1
        if verbose:
            marker = ' *' if fo_r < best_fo - 1e-9 else ''
            print(f"      raiz={r:>3}  FO={fo_r:.6f}{marker}")
        if fo_r < best_fo - 1e-9:
            best_T, best_raiz, best_fo = T_r, r, fo_r

    return best_T, best_raiz, fo_inicial, best_fo, n_testadas


# ── LS estrutural na arvore (mantido para referencia) ─────────────────────────

def _child_priority_swap_ls(
    T: nx.DiGraph, raiz: int, fo_init: float, max_time_s: float
) -> Tuple[nx.DiGraph, float, int]:
    """
    Troca num_filhos entre pares de irmaos para alterar ordem de interleaving.
    Resultado empirico: 0 melhorias em todas as instancias testadas.
    Razao: o disparo geometrico e independente da prioridade exceto quando
    dois filhos compartilham o mesmo vertice de conexao no pai (raro).
    """
    best_T = T.copy(); best_fo = fo_init; ni = 0
    t0 = time.time()
    improved = True
    while improved:
        if time.time() - t0 > max_time_s: break
        improved = False
        for v in list(nx.dfs_preorder_nodes(best_T, raiz)):
            if time.time() - t0 > max_time_s: break
            children = list(best_T.successors(v))
            if len(children) < 2: continue
            for i in range(len(children)):
                if time.time() - t0 > max_time_s: break
                for j in range(i + 1, len(children)):
                    ci, cj = children[i], children[j]
                    nf_i = best_T.nodes[ci]['num_filhos']
                    nf_j = best_T.nodes[cj]['num_filhos']
                    if nf_i == nf_j: continue
                    T_c = best_T.copy()
                    T_c.nodes[ci]['num_filhos'] = nf_j
                    T_c.nodes[cj]['num_filhos'] = nf_i
                    fo_c = _fo_arvore(T_c, raiz)
                    if fo_c < best_fo - 1e-9:
                        best_T, best_fo, improved, ni = T_c, fo_c, True, ni + 1
                        break
                if improved: break
            if improved: break
    return best_T, best_fo, ni


def _subtree_relocation_ls(
    T: nx.DiGraph, raiz: int, polys: Dict[int, Polygon],
    fo_init: float, max_time_s: float
) -> Tuple[nx.DiGraph, float, int]:
    """
    Move subarvores de pai para pai alternativo.
    Resultado empirico: 0 melhorias em todas as instancias testadas.
    Razao: BFS ja produz conexoes de distancia minima; qualquer realocacao
    aumenta o custo de conexao e portanto a FO.
    """
    best_T = T.copy(); best_fo = fo_init; ni = 0
    t0 = time.time()
    improved = True
    while improved:
        if time.time() - t0 > max_time_s: break
        improved = False
        for v in list(nx.dfs_preorder_nodes(best_T, raiz)):
            if v == raiz or time.time() - t0 > max_time_s: break
            old_p = list(best_T.predecessors(v))[0]
            subtree = {v} | nx.descendants(best_T, v)
            for new_p in [u for u in best_T.nodes if u not in subtree and u != old_p]:
                if time.time() - t0 > max_time_s: break
                T_c = best_T.copy()
                T_c.remove_edge(old_p, v)
                w, conn = _edge_data_between(polys[new_p], polys[v])
                T_c.add_edge(new_p, v, weight=w, connection=conn)
                _update_num_filhos(T_c)
                fo_c = _fo_arvore(T_c, raiz)
                if fo_c < best_fo - 1e-9:
                    best_T, best_fo, improved, ni = T_c, fo_c, True, ni + 1
                    break
            if improved: break
    return best_T, best_fo, ni


# ── API publica ───────────────────────────────────────────────────────────────

def run_tree_ls(
    arvore: nx.DiGraph,
    raiz: int,
    polys: Dict[int, Polygon],
    max_time_s: float = 60.0,
    verbose: bool = False,
) -> Tuple[nx.DiGraph, int, float, float, int, int]:
    """
    Aplica os operadores de LS estrutural na arvore.
    Ver docstring do modulo para explicacao de por que esses operadores
    nao encontram melhorias na pratica.

    Retorna: (best_T, best_raiz, fo_init, fo_best, ni_swap, ni_reloc)
    """
    fo_init = _fo_arvore(arvore, raiz)
    slot = max_time_s / 3.0
    t0 = time.time()

    T1, fo1, ni_swap1 = _child_priority_swap_ls(arvore, raiz, fo_init, slot)
    if verbose:
        print(f"    [ChildSwap] {fo_init:.6f} -> {fo1:.6f}  ni={ni_swap1}")

    rem = max(1.0, max_time_s - (time.time() - t0))
    T2, fo2, ni_reloc = _subtree_relocation_ls(T1, raiz, polys, fo1, rem * 0.75)
    if verbose:
        print(f"    [SubtreeReloc] {fo1:.6f} -> {fo2:.6f}  ni={ni_reloc}")

    rem = max(1.0, max_time_s - (time.time() - t0))
    T3, fo3, ni_swap2 = _child_priority_swap_ls(T2, raiz, fo2, rem)
    if verbose:
        print(f"    [ChildSwap-2] {fo2:.6f} -> {fo3:.6f}  ni={ni_swap2}")

    return T3, raiz, fo_init, fo3, ni_swap1 + ni_swap2, ni_reloc
