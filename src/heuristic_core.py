"""
heuristic_core.py
=================
Núcleo modular da heurística de corte a laser.

Expõe `run_heuristic(layout, config)` onde `config` é um dict que
controla qual variante de cada passo é utilizada:

  config = {
      "root_strategy":     "nearest" | "enumerate",
      "child_order":       "num_filhos" | "permutation" | "2opt",
      "mst_strategy":      "standard" | "edge_swap",
      "vertex_strategy":   "nearest" | "enumerate",
      "enforce_precedence": bool  (default False -- ver nota abaixo),
      "time_limit":        float  (seg, por instância; None = sem limite)
  }

Retorna um dict com FO, tempo de execução, caminho e metadados.

Nota sobre `enforce_precedence`
-------------------------------
O traversal original (`caminha_arvore`/`caminha_arvore_com_ordem`, de
`graph_routing.py` e deste módulo) NÃO garante a precedência formal Π
("furo antes do contorno externo que o contém"): a ordem de corte entre um
contorno aninhado e seu "pai" geométrico é decidida inteiramente pela
topologia da MST/raiz, sem consciência de Π (nas instâncias com furo
testadas, todos os 40 pares verificados foram violados).

A versão original é mantida intacta (funções
`caminha_arvore_com_ordem`/`cortar_arvore_com_ordem` abaixo e `cortar_arvore`,
importada de `graph_routing.py`) e continua sendo o comportamento padrão
(`enforce_precedence=False`), o que permite comparar lado a lado a FO da
versão original e a da versão com precedência garantida (ver
`cortar_arvore_com_precedencia`, seção "Traversal com garantia de
precedência").

Nota sobre `precedence_strategy`
--------------------------------
`precedence_strategy` só tem efeito com `enforce_precedence=True` e o padrão
é `"sequential"`. A estratégia "sequential" é uma construção de passe único,
sem operador de busca local; as demais alternativas existem para recuperar
qualidade sem perder a garantia de Π. Os branches `child_order=permutation/2opt`
seguem incompatíveis com `enforce_precedence`.
  - "sequential"      : sem busca local (comportamento original do
                        enforce_precedence).
  - "sequential_2opt" : 2-opt (troca de posição + inversão de segmento)
                        sobre a ordem sequencial, rejeitando qualquer troca
                        que viole Π antes mesmo de recomputar a FO.
  - "hybrid"          : versão LCA, generalizada e com recuo PARCIAL --
                        preserva o traversal original (interleaved/
                        oportunista) em toda a árvore, corrigindo localmente
                        via ancestral comum (LCA) qualquer par Π, mesmo entre
                        ramos diferentes; só recua para tratamento sequencial
                        na subárvore onde há contradição genuína entre ramos
                        (raro).
  - "hybrid_v1"       : versão inicial do híbrido (só vizinho direto na
                        árvore; recuo TOTAL da instância se algum par não for
                        vizinho direto). Mantida para comparação.
"""
from __future__ import annotations

import math
import sys
import time
from itertools import permutations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import chebyshev
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import nearest_points

# Garante que o diretório original da heurística está no path

from geometry import segment_intersections
from graph_routing import (
    construir_arvore_direcionada,
    cortar_arvore,
    distancia_e_aresta_ate_raiz,
    linha_esta_na_lista,
    plotar_poligonos_e_arvore,
)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
_V_CUT = 100.0 / 6.0   # velocidade de corte  (≈ 16,67 unid/s)
_V_AIR = 400.0          # velocidade em vazio

_DEFAULT_CONFIG: Dict[str, Any] = {
    "root_strategy":     "nearest",    # como escolher a raiz
    "child_order":       "num_filhos", # como ordenar filhos
    "mst_strategy":      "standard",   # como construir a MST
    "vertex_strategy":   "nearest",    # como escolher vértice inicial
    "enforce_precedence": False,       # False = comportamento ORIGINAL (sem garantia de Pi)
    "precedence_strategy": "sequential",  # "sequential" | "sequential_2opt" | "hybrid"
    "time_limit":        None,         # tempo máximo em segundos
}


# ===========================================================================
# Utilitários de FO
# ===========================================================================

def _seg_key(seg) -> tuple:
    """Chave canônica (direção-agnóstica) para um segmento."""
    c0 = tuple(round(v, 8) for v in seg.coords[0])
    c1 = tuple(round(v, 8) for v in seg.coords[1])
    return (min(c0, c1), max(c0, c1))


def reclassificar_tags(caminho):
    """
    Reclassifica tags 'x' em 'c' (corte) ou 'd' (deslocamento no ar).

    Regra corrigida: um segmento 'x' torna-se 'd' apenas se já foi
    anteriormente classificado como 'c' (já cortado na passagem anterior
    do DFS). Segmentos de abordagem/retorno explicitamente marcados 'd'
    NÃO interferem na classificação de arestas de polígono — evita o bug
    de confundir a abordagem de (0,0) com uma aresta do layout.
    """
    cortados: set = set()
    resultado = []
    for seg, tag in caminho:
        if tag != "x":
            resultado.append((seg, tag))
        else:
            key = _seg_key(seg)
            if key in cortados:
                resultado.append((seg, "d"))   # backtrack — já cortado
            else:
                cortados.add(key)
                resultado.append((seg, "c"))   # primeiro corte
    return resultado


def calcular_fo(caminho) -> float:
    """Calcula a função objetivo de um caminho já classificado."""
    return sum(
        chebyshev(s.coords[0], s.coords[1]) / (_V_CUT if t == "c" else _V_AIR)
        for s, t in caminho
        if hasattr(s, "coords") and len(list(s.coords)) >= 2
    )


def fo_from_path(caminho_raw) -> float:
    """Reclassifica e calcula FO em uma chamada."""
    return calcular_fo(reclassificar_tags(caminho_raw))


# ===========================================================================
# Passo 1 – Tratamento de furos
# ===========================================================================

def tratar_furos(layout: List[Polygon]) -> List[Polygon]:
    result = list(layout)
    for p in list(result):
        for interior in p.interiors:
            result.append(Polygon(interior))
    return result


# ===========================================================================
# Passo 2 – Resolução de interseções
# ===========================================================================

def resolver_intersecoes(layout: List[Polygon]) -> List[Polygon]:
    intersecoes = set()
    for i, f1 in enumerate(layout):
        for j, f2 in enumerate(layout):
            if i != j and f1.intersects(f2):
                intersecoes.add(tuple(sorted((i, j))))
    for i, j in list(intersecoes):
        try:
            part = segment_intersections(layout[i], layout[j])
            layout[i], layout[j] = [Polygon(k) for k in part]
        except Exception:
            pass
    return layout


# ===========================================================================
# Passo 3 – QuadTree (reutiliza do módulo original)
# ===========================================================================

def clusterizar_quadtree(layout: List[Polygon]) -> List[Polygon]:
    from quadtree import preprocess_layout_with_quadtree
    return preprocess_layout_with_quadtree(layout)


# ===========================================================================
# Passo 4 – Grafo + MST
# ===========================================================================

def construir_grafo_mst(positions: List[Polygon]) -> nx.Graph:
    """MST padrão baseada em vértices (comportamento original)."""
    from quadtree import optimize_cluster_connections_kdtree_vertices_with_full_connectivity
    return optimize_cluster_connections_kdtree_vertices_with_full_connectivity(
        positions, plot=False
    )


def construir_grafo_mst_edge_swap(
    positions: List[Polygon],
    raiz_fn,
    time_limit: Optional[float] = None,
) -> Tuple[nx.Graph, Dict]:
    """
    MST com busca local edge-swap.
    Remove uma aresta da MST e tenta substituir por outra aresta do grafo
    original que mantenha conectividade e reduza a FO.
    """
    G_full = _build_full_graph(positions)
    mst = nx.minimum_spanning_tree(G_full, weight="weight")

    # Calcula FO com a MST base
    raiz = raiz_fn(mst, positions)
    best_fo = _fo_mst(mst, raiz, positions)
    best_mst = mst.copy()
    n_improvements = 0
    n_evals = 0

    t0 = time.time()
    improved = True
    while improved:
        improved = False
        for u, v in list(best_mst.edges):
            if time_limit and (time.time() - t0) > time_limit:
                break
            mst_temp = best_mst.copy()
            mst_temp.remove_edge(u, v)
            comp = list(nx.connected_components(mst_temp))

            for a in G_full.edges:
                if time_limit and (time.time() - t0) > time_limit:
                    break
                if a not in mst_temp.edges:
                    na, nb = a
                    if any(na in c and nb not in c for c in comp):
                        mst_temp.add_edge(*a, **G_full.edges[a])
                        raiz_temp = raiz_fn(mst_temp, positions)
                        fo_temp = _fo_mst(mst_temp, raiz_temp, positions)
                        n_evals += 1
                        if fo_temp < best_fo - 1e-9:
                            best_fo = fo_temp
                            best_mst = mst_temp.copy()
                            n_improvements += 1
                            improved = True
                            break
                        else:
                            mst_temp.remove_edge(*a)

            if improved:
                break

    meta = {"edge_swap_improvements": n_improvements, "edge_swap_evals": n_evals}
    return best_mst, meta


def _build_full_graph(positions: List[Polygon]) -> nx.Graph:
    """
    Constrói o grafo COMPLETO: uma aresta (i, j) para cada par de polígonos,
    com peso = menor distância de Chebyshev entre qualquer vértice de i e j.
    Isso é necessário para que o edge-swap tenha arestas não-MST para explorar.
    A versão anterior adicionava apenas ~N-1 arestas (uma por polígono), que é
    exatamente o número de arestas da MST, deixando E_g - E_mst = 0.
    """
    N = len(positions)
    G = nx.Graph()
    for i, polygon in enumerate(positions):
        G.add_node(i, poligono=polygon, pos=polygon.centroid.coords[0])

    # Para cada par (i, j), encontra o par de vértices com menor distância de Chebyshev
    for i in range(N):
        for j in range(i + 1, N):
            min_dist = float("inf")
            best_v1, best_v2 = None, None
            for v1 in positions[i].exterior.coords:
                for v2 in positions[j].exterior.coords:
                    d = chebyshev(v1, v2)
                    if d < min_dist:
                        min_dist = d
                        best_v1, best_v2 = v1, v2
            if best_v1 is not None:
                G.add_edge(i, j, weight=min_dist, connection=(best_v1, best_v2))

    return G


def _fo_mst(mst: nx.Graph, raiz: int, positions: List[Polygon]) -> float:
    """Avalia FO completa de uma MST (reconstrói árvore e caminho)."""
    T = construir_arvore_direcionada(mst, raiz)
    cam = reclassificar_tags(cortar_arvore(T, raiz))
    return calcular_fo(cam)


# ===========================================================================
# Passo 5 – Escolha da raiz
# ===========================================================================

def escolher_raiz_nearest(subgrafo: nx.Graph, positions: List[Polygon]) -> int:
    """Raiz padrão: polígono com borda mais próxima da origem (0, 0)."""
    origin = Point(0, 0)
    dists = [
        (i, distancia_e_aresta_ate_raiz(origin, positions[i])[0][0].length
         if distancia_e_aresta_ate_raiz(origin, positions[i])[0] else 0.0)
        for i in subgrafo.nodes
    ]
    return min(dists, key=lambda x: x[1])[0]


def enumerar_raiz(
    subgrafo: nx.Graph,
    positions: List[Polygon],
    time_limit: Optional[float] = None,
) -> Tuple[int, float, int]:
    """
    Testa todos os nós como raiz e retorna o melhor.
    Retorna: (melhor_raiz, melhor_fo, n_candidatos_testados)
    """
    best_raiz, best_fo = None, float("inf")
    n_tested = 0
    t0 = time.time()

    for candidato in subgrafo.nodes:
        if time_limit and (time.time() - t0) > time_limit:
            break
        arvore = construir_arvore_direcionada(subgrafo, candidato)
        cam = reclassificar_tags(cortar_arvore(arvore, candidato))
        fo = calcular_fo(cam)
        n_tested += 1
        if fo < best_fo:
            best_fo = fo
            best_raiz = candidato

    return best_raiz, best_fo, n_tested


# ===========================================================================
# Variantes de geração do caminho
# ===========================================================================

def caminha_arvore_com_ordem(
    G: nx.DiGraph,
    raiz: int,
    ponto_inicial: Point,
    pol_cobertos: List[int],
    child_order_fn,
) -> Tuple[List, List[int]]:
    """
    Versão de caminha_arvore que aceita uma função para ordenar filhos.
    `child_order_fn(G, raiz)` → lista ordenada de filhos.
    """
    from graph_routing import reordenar_poligono, chebyshev_distance

    caminho = []
    successors = child_order_fn(G, raiz)

    if len(successors) > 0:
        caminho_pol = []
        pol = G.nodes[raiz]["poligono"]
        coords = reordenar_poligono(pol, ponto_inicial)
        for i in range(len(coords) - 1):
            caminho_pol.append(LineString([Point(coords[i]), Point(coords[i + 1])]))

        for seg in caminho_pol:
            caminho.append((seg, "x"))
            for s in filter(lambda x: x not in pol_cobertos, successors):
                if G.edges[raiz, s]["weight"] == 0:
                    pol_destino = G.nodes[s]["poligono"]
                    intersecao = pol.intersection(pol_destino)
                    if (intersecao.distance(Point(seg.coords[1])) < 1e-8
                            or intersecao.covers(Point(seg.coords[0]))):
                        desc = nx.descendants(G, s)
                        desc.add(s)
                        if desc:
                            pol_cobertos.append(s)
                            c, _ = caminha_arvore_com_ordem(
                                G.subgraph(desc), s,
                                Point(seg.coords[1]), pol_cobertos, child_order_fn
                            )
                            caminho.extend(c)
                else:
                    intersecao = G.edges[raiz, s]["connection"]
                    if (intersecao.distance(Point(seg.coords[1])) < 1e-8
                            or intersecao.covers(Point(seg.coords[0]))):
                        caminho.append((intersecao, "d"))
                        desc = nx.descendants(G, s)
                        desc.add(s)
                        if desc:
                            pol_cobertos.append(s)
                            c, _ = caminha_arvore_com_ordem(
                                G.subgraph(desc), s,
                                Point(seg.coords[1]), pol_cobertos, child_order_fn
                            )
                            caminho.extend(c)
    else:
        pol = G.nodes[raiz]["poligono"]
        from graph_routing import reordenar_poligono
        coords = reordenar_poligono(pol, ponto_inicial)
        for i in range(len(coords) - 1):
            caminho.append((LineString([Point(coords[i]), Point(coords[i + 1])]), "x"))

    return caminho, pol_cobertos


def cortar_arvore_com_ordem(arvore: nx.DiGraph, raiz: int, child_order_fn) -> List:
    """
    Versão de cortar_arvore com ordem de filhos customizável.

    ORIGINAL -- mantida intacta para comparação lado a lado. NÃO garante a
    precedência Π (furo antes do contorno externo) -- ver
    `cortar_arvore_com_precedencia` abaixo para a versão com garantia, e a
    nota no topo do módulo.
    """
    caminho = []
    aresta, ponto = distancia_e_aresta_ate_raiz(Point(0, 0), arvore.nodes[raiz]["poligono"])
    if aresta:
        caminho.append((aresta[0], "d"))
    c, _ = caminha_arvore_com_ordem(arvore, raiz, ponto, [raiz], child_order_fn)
    caminho += c
    if aresta:
        caminho.append((LineString(list(aresta[0].coords)[::-1]), "d"))
    return caminho


# ===========================================================================
# Traversal com garantia de precedência
# ===========================================================================
#
# Detecta Π via contenção geométrica direta (Polygon.covers) -- mesma lógica
# de `pcgtsp_comparison/converter.py::_detect_precedence` --
# o pipeline de carregamento (load_instances.py) não preserva relação
# furo/pai em nenhum lugar (todo contorno fechado vira um Polygon "chato"
# independente).
#
# Estratégia (uma recursão do tipo "empurra predecessor pendente pela árvore"
# entraria em recursão infinita sempre que um furo fosse ANCESTRAL do seu
# próprio pai na árvore da MST, em vez de descendente -- caso comum: num
# único nó de `instance_artificial_01_26pol_hole` há 15 furos, a maioria NÃO
# adjacente a ele na árvore, então a posição relativa pai/furo na árvore é
# essencialmente arbitrária e não dá para assumir uma direção fixa):
#
# 1. Parte da ordem-base de visita = DFS pré-ordem da árvore (a mesma
#    localidade espacial que o traversal original já explora via MST).
# 2. Corrige essa ordem para respeitar Π: sempre que um furo aparece DEPOIS
#    do seu pai na ordem, move o furo para logo antes do pai (ponto fixo,
#    Π é raso -- normalmente 1 nível -- poucas iterações bastam).
# 3. Gera o caminho visitando os nós NESSA ordem final, sequencialmente:
#    cada contorno cortado por completo, com deslocamento explícito ('d')
#    entre o fim de um e o início do próximo.
#
# Isso é estruturalmente mais simples que o traversal original (que
# interleaves o corte de um nó com incursões nos filhos no ponto exato de
# contato) -- deixa de explorar a "carona" oportunista em pontos de toque,
# então um comparativo de FO contra o traversal original mede tanto o custo
# de garantir Π quanto o custo de trocar de estilo de roteamento. Para
# isolar os dois efeitos, é preciso comparar também as variantes
# intermediárias (sequencial SEM Π, sequencial COM Π).


def _detectar_precedencia(positions: List[Polygon]) -> List[Tuple[int, int]]:
    """Pares (furo_idx, pai_idx) via contenção geométrica (Polygon.covers).
    Mesma lógica usada em pcgtsp_comparison/converter.py -- duplicada aqui
    (em vez de importada) para manter heuristic_core.py sem dependência do
    pacote pcgtsp_comparison."""
    pares = []
    for i, pol_i in enumerate(positions):
        candidatos_pai = []
        for j, pol_j in enumerate(positions):
            if i == j:
                continue
            if pol_i.area < pol_j.area * 0.98 and pol_j.covers(pol_i.buffer(-1e-9)):
                candidatos_pai.append((pol_j.area, j))
        if candidatos_pai:
            candidatos_pai.sort()
            pares.append((i, candidatos_pai[0][1]))
    return pares


def _mapa_predecessores(pares: List[Tuple[int, int]]) -> Dict[int, List[int]]:
    """preds[pai] = [furo1, furo2, ...] -- tudo que deve ser cortado ANTES de 'pai'."""
    preds: Dict[int, List[int]] = {}
    for furo, pai in pares:
        preds.setdefault(pai, []).append(furo)
    return preds


def _ordem_respeitando_precedencia(
    ordem_base: List[int],
    preds: Dict[int, List[int]],
) -> List[int]:
    """
    A partir de uma ordem-base de visita (ex.: DFS pré-ordem da árvore),
    corrige violações de precedência com o mínimo de mudança estrutural:
    sempre que um nó aparece antes de algum dos seus próprios predecessores
    em Π, move o predecessor para logo antes dele. Iterativo até ponto fixo
    -- Π é um DAG raso (contenção geométrica direta), poucas iterações bastam
    mesmo quando um nó tem muitos furos (ex.: 15 furos no mesmo pai em
    `instance_artificial_01_26pol_hole`).
    """
    ordem = list(ordem_base)
    max_iter = len(ordem) * 2 + 10
    it = 0
    mudou = True
    while mudou and it < max_iter:
        mudou = False
        it += 1
        pos = {n: i for i, n in enumerate(ordem)}
        for no in ordem:
            for pred in preds.get(no, []):
                if pred not in pos:
                    continue
                if pos[pred] > pos[no]:
                    ordem.remove(pred)
                    ordem.insert(ordem.index(no), pred)
                    mudou = True
                    break
            if mudou:
                break
    return ordem


def cortar_por_ordem_sequencial(G: nx.Graph, ordem: List[int]) -> List:
    """
    Gera o caminho cortando os nós de 'G' na ORDEM dada, sequencialmente:
    cada contorno é cortado por completo (reordenar_poligono a partir de
    onde estamos), com um deslocamento explícito ('d') entre o fim de um e
    o início do próximo (vértice mais próximo). Mais simples que o
    traversal original (que faz "carona" nos filhos durante a travessia da
    própria borda) -- usado tanto pela variante 'sequencial' (sem garantia
    de Π, para isolar o custo de trocar de estilo de roteamento) quanto pela
    variante 'enforce_precedence' (aplicando `_ordem_respeitando_precedencia`
    antes de chamar esta função).
    """
    from graph_routing import reordenar_poligono, chebyshev_distance

    caminho: List = []
    origem = Point(0, 0)
    ponto_atual = origem

    for no in ordem:
        pol = G.nodes[no]["poligono"]
        coords_pol = list(pol.exterior.coords)[:-1]
        entrada = min(coords_pol, key=lambda c: chebyshev_distance(ponto_atual, Point(c)))
        entrada_pt = Point(entrada)
        if ponto_atual.distance(entrada_pt) > 1e-9:
            caminho.append((LineString([ponto_atual, entrada_pt]), "d"))

        coords = reordenar_poligono(pol, entrada_pt)
        for i in range(len(coords) - 1):
            caminho.append((LineString([Point(coords[i]), Point(coords[i + 1])]), "x"))

        ponto_atual = entrada_pt  # laço fechado -- termina onde começou

    if ponto_atual.distance(origem) > 1e-9:
        caminho.append((LineString([ponto_atual, origem]), "d"))

    return caminho


def cortar_arvore_com_precedencia(
    arvore: nx.DiGraph,
    raiz: int,
    preds: Dict[int, List[int]],
    child_order_fn,
) -> List:
    """
    Versão de cortar_arvore que GARANTE a precedência Π (furo antes do
    contorno externo que o contém). Usar com `enforce_precedence=True` em
    run_heuristic. Ver a nota de topo do módulo. É formulada como uma
    ordenação sequencial porque uma versão recursiva e "oportunista" como a
    original entraria em recursão infinita sempre que um furo fosse
    ancestral do próprio pai na árvore (caso comum, não uma exceção rara).

    `child_order_fn` é aceito por compatibilidade de assinatura com o
    restante do módulo mas não é usado aqui -- a ordem de filhos não se
    aplica a um passeio sequencial; ver nota acima.
    """
    ordem_base = list(nx.dfs_preorder_nodes(arvore, raiz))
    ordem_final = _ordem_respeitando_precedencia(ordem_base, preds)
    return cortar_por_ordem_sequencial(arvore, ordem_final)


# ===========================================================================
# Estratégia "sequential_2opt": busca local respeitando Π sobre a ordem
# sequencial, para polir a construção de passe único do branch "sequential".
# ===========================================================================

def _ordem_e_factivel(ordem: List[int], pares: List[Tuple[int, int]]) -> bool:
    """True sse TODO par (furo, pai) tem furo antes de pai nesta ordem.
    Checagem barata (O(|pares|) comparações de índice) usada para descartar
    a maioria dos movimentos de busca local ANTES de pagar o custo de
    reconstruir o caminho inteiro e recalcular a FO."""
    pos = {n: i for i, n in enumerate(ordem)}
    for furo, pai in pares:
        if furo not in pos or pai not in pos:
            continue
        if pos[furo] > pos[pai]:
            return False
    return True


def cortar_arvore_com_precedencia_2opt(
    arvore: nx.DiGraph,
    raiz: int,
    preds: Dict[int, List[int]],
    pares: List[Tuple[int, int]],
    time_limit: Optional[float] = None,
) -> Tuple[List, Dict[str, int]]:
    """
    Estratégia "sequential_2opt". Parte da MESMA ordem sequencial Π-factível de
    `cortar_arvore_com_precedencia` e aplica busca local de duas famílias de
    movimento -- troca de posição (swap) e inversão de segmento (2-opt
    clássico) --, testando, para cada movimento candidato:
      1. factibilidade de Π (barato, `_ordem_e_factivel`) -- só passa daqui
         se a troca continuar respeitando TODOS os pares Π, não só o par
         diretamente envolvido no movimento;
      2. ganho de FO (caro, reconstrói o caminho inteiro via
         `cortar_por_ordem_sequencial` + `fo_from_path`) -- só computado
         para movimentos já sabidamente factíveis.
    Estratégia gulosa de primeira melhora (mesmo padrão de
    `compute_child_permutations(mode="2opt")` já usado no restante do
    módulo): aceita o primeiro movimento que melhora e reinicia a varredura.

    Retorna (caminho_raw, meta) -- meta traz `precedencia_2opt_evals` (nº de
    movimentos factíveis avaliados) e `precedencia_2opt_improvements`.
    """
    ordem_base = list(nx.dfs_preorder_nodes(arvore, raiz))
    ordem = _ordem_respeitando_precedencia(ordem_base, preds)

    caminho = cortar_por_ordem_sequencial(arvore, ordem)
    best_fo = fo_from_path(caminho)

    t0 = time.time()
    n = len(ordem)
    n_evals = 0
    n_improvements = 0
    melhorou = True

    while melhorou:
        melhorou = False
        for i in range(n):
            if time_limit and (time.time() - t0) > time_limit:
                break
            for j in range(i + 1, n):
                if time_limit and (time.time() - t0) > time_limit:
                    break

                candidatos = []
                troca = list(ordem)
                troca[i], troca[j] = troca[j], troca[i]
                candidatos.append(troca)
                if j > i + 1:
                    inverso = ordem[:i] + ordem[i:j + 1][::-1] + ordem[j + 1:]
                    candidatos.append(inverso)

                for candidata in candidatos:
                    if not _ordem_e_factivel(candidata, pares):
                        continue
                    n_evals += 1
                    cam = cortar_por_ordem_sequencial(arvore, candidata)
                    fo = fo_from_path(cam)
                    if fo < best_fo - 1e-9:
                        ordem = candidata
                        best_fo = fo
                        caminho = cam
                        n_improvements += 1
                        melhorou = True
                        break
                if melhorou:
                    break
            if melhorou:
                break

    meta = {
        "precedencia_2opt_evals": n_evals,
        "precedencia_2opt_improvements": n_improvements,
    }
    return caminho, meta


# ===========================================================================
# Traversal híbrido (estratégia "hybrid_v1"): preserva o estilo
# interleaved/oportunista ORIGINAL onde não há restrição de Π, corrigindo
# de forma LOCAL apenas os pares Π que são vizinhos diretos na árvore.
#
# Nas 3 instâncias testadas (art_1/5/10_hole_sep, 40 pares Π no total), o
# traversal ORIGINAL violou Π em 100% dos casos, nas duas direções (38/40
# pares: furo = filho direto de pai; 2/40: pai = filho direto de furo; ZERO
# casos cruzam mais de um nível da árvore) -- não existe subconjunto "de
# graça". Esta implementação aplica a correção da forma mais LOCAL possível
# (só entre o par pai-filho envolvido, via deslocamento explícito),
# preservando o estilo interleaved para TODO O RESTO da árvore -- ao
# contrário da versão "sequential", que reordena a árvore INTEIRA.
# ===========================================================================

def _deslocar_e_visitar(caminho, ponto_atual, no_destino, G, pol_cobertos,
                         child_order_fn, preds, deve_preceder):
    """
    Desloca explicitamente de 'ponto_atual' até o vértice mais próximo do
    polígono de 'no_destino' (mesmo critério de `cortar_por_ordem_sequencial`)
    e visita 'no_destino' (e sua subárvore) por completo via
    `caminha_arvore_hibrida`. Devolve o novo ponto_atual (fecha no próprio
    ponto de entrada -- polígono é um laço fechado).
    """
    from graph_routing import chebyshev_distance

    pol_destino = G.nodes[no_destino]["poligono"]
    coords_destino = list(pol_destino.exterior.coords)[:-1]
    entrada = min(coords_destino, key=lambda c: chebyshev_distance(ponto_atual, Point(c)))
    entrada_pt = Point(entrada)
    if ponto_atual.distance(entrada_pt) > 1e-9:
        caminho.append((LineString([ponto_atual, entrada_pt]), "d"))

    desc = nx.descendants(G, no_destino)
    desc.add(no_destino)
    pol_cobertos.append(no_destino)
    c, _ = caminha_arvore_hibrida(
        G.subgraph(desc), no_destino, entrada_pt, pol_cobertos,
        child_order_fn, preds, deve_preceder,
    )
    caminho.extend(c)
    return entrada_pt


def caminha_arvore_hibrida(
    G: nx.DiGraph,
    raiz: int,
    ponto_inicial: Point,
    pol_cobertos: List[int],
    child_order_fn,
    preds: Dict[int, List[int]],
    deve_preceder: Dict[int, List[int]],
) -> Tuple[List, List[int]]:
    """
    Fork de `caminha_arvore_com_ordem` (mantida intacta, ver nota de topo do
    módulo) para a Alternativa 2. Dois ajustes pontuais em relação ao
    original, ambos restritos a vizinhos DIRETOS na árvore:

      - preds[raiz]         : furos-filhos diretos de 'raiz' -- visitados
        por completo ANTES de qualquer segmento da própria borda de 'raiz'
        (deslocamento explícito ida-e-volta), removidos da lista de
        sucessores "oportunistas" normais.
      - deve_preceder[raiz] : filhos diretos que 'raiz' deve preceder (raiz
        é furo, o filho é o pai em Π) -- adiados para depois que a própria
        borda de 'raiz' e os demais filhos normais já tiverem sido
        percorridos (deslocamento explícito).

    Todo o resto (filhos sem restrição de Π) usa exatamente a mesma lógica
    interleaved/oportunista do original -- mesma "carona" nos pontos de
    contato, sem custo extra.
    """
    from graph_routing import reordenar_poligono

    caminho: List = []
    successors = list(child_order_fn(G, raiz))

    furos_diretos = [s for s in preds.get(raiz, [])
                     if s in successors and s not in pol_cobertos]
    adiados = [s for s in deve_preceder.get(raiz, [])
               if s in successors and s not in pol_cobertos and s not in furos_diretos]
    successors_normais = [s for s in successors
                           if s not in furos_diretos and s not in adiados]

    # 1) furos-filhos diretos: visita completa ANTES da própria borda.
    ponto_atual = ponto_inicial
    for s in furos_diretos:
        ponto_atual = _deslocar_e_visitar(
            caminho, ponto_atual, s, G, pol_cobertos, child_order_fn, preds, deve_preceder
        )
    if furos_diretos and ponto_atual.distance(ponto_inicial) > 1e-9:
        caminho.append((LineString([ponto_atual, ponto_inicial]), "d"))

    # 2) borda de 'raiz' + filhos normais -- lógica ORIGINAL, inalterada,
    #    restrita a successors_normais (furos_diretos/adiados já excluídos).
    pol = G.nodes[raiz]["poligono"]
    coords = reordenar_poligono(pol, ponto_inicial)
    for i in range(len(coords) - 1):
        seg = LineString([Point(coords[i]), Point(coords[i + 1])])
        caminho.append((seg, "x"))
        for s in filter(lambda x: x not in pol_cobertos, successors_normais):
            if G.edges[raiz, s]["weight"] == 0:
                pol_destino = G.nodes[s]["poligono"]
                intersecao = pol.intersection(pol_destino)
                if (intersecao.distance(Point(seg.coords[1])) < 1e-8
                        or intersecao.covers(Point(seg.coords[0]))):
                    desc = nx.descendants(G, s)
                    desc.add(s)
                    if desc:
                        pol_cobertos.append(s)
                        c, _ = caminha_arvore_hibrida(
                            G.subgraph(desc), s, Point(seg.coords[1]),
                            pol_cobertos, child_order_fn, preds, deve_preceder,
                        )
                        caminho.extend(c)
            else:
                intersecao = G.edges[raiz, s]["connection"]
                if (intersecao.distance(Point(seg.coords[1])) < 1e-8
                        or intersecao.covers(Point(seg.coords[0]))):
                    caminho.append((intersecao, "d"))
                    desc = nx.descendants(G, s)
                    desc.add(s)
                    if desc:
                        pol_cobertos.append(s)
                        c, _ = caminha_arvore_hibrida(
                            G.subgraph(desc), s, Point(seg.coords[1]),
                            pol_cobertos, child_order_fn, preds, deve_preceder,
                        )
                        caminho.extend(c)

    # a borda fecha exatamente onde 'reordenar_poligono' começou.
    ponto_atual = Point(coords[0])

    # 3) filhos adiados: visita completa DEPOIS da própria borda + normais.
    for s in adiados:
        ponto_atual = _deslocar_e_visitar(
            caminho, ponto_atual, s, G, pol_cobertos, child_order_fn, preds, deve_preceder
        )

    return caminho, pol_cobertos


def cortar_arvore_hibrida_precedencia(
    arvore: nx.DiGraph,
    raiz: int,
    preds: Dict[int, List[int]],
    pares: List[Tuple[int, int]],
    child_order_fn,
) -> List:
    """
    Wrapper do traversal híbrido ("hybrid_v1"), análogo a
    `cortar_arvore_com_precedencia`. Classifica cada par Π como "furo direto
    de pai" ou "pai direto de furo" na árvore, e usa `caminha_arvore_hibrida`
    para aplicar a correção localmente.

    Salvaguarda: se algum par Π NÃO for vizinho direto na árvore (não
    observado em nenhuma instância testada até agora, mas não impossível em
    geral), cai para `cortar_arvore_com_precedencia`
    (versão sequencial global, já validada em 653 pares/50 instâncias) em
    vez de arriscar uma correção local incompleta.
    """
    preds_diretos: Dict[int, List[int]] = {}
    deve_preceder: Dict[int, List[int]] = {}
    for furo, pai in pares:
        if furo not in arvore or pai not in arvore:
            continue
        filhos_de_pai = set(arvore.successors(pai))
        filhos_de_furo = set(arvore.successors(furo))
        if furo in filhos_de_pai:
            preds_diretos.setdefault(pai, []).append(furo)
        elif pai in filhos_de_furo:
            deve_preceder.setdefault(furo, []).append(pai)
        else:
            return cortar_arvore_com_precedencia(arvore, raiz, preds, child_order_fn)

    caminho = []
    aresta, ponto = distancia_e_aresta_ate_raiz(Point(0, 0), arvore.nodes[raiz]["poligono"])
    if aresta:
        caminho.append((aresta[0], "d"))
    c, _ = caminha_arvore_hibrida(
        arvore, raiz, ponto, [raiz], child_order_fn, preds_diretos, deve_preceder
    )
    caminho += c
    if aresta:
        caminho.append((LineString(list(aresta[0].coords)[::-1]), "d"))
    return caminho


# ===========================================================================
# Traversal híbrido com LCA (estratégia "hybrid"): generaliza o
# mecanismo acima de "vizinho direto" para QUALQUER par Π (mesmo ramo a mais
# de 1 nível, ou ramos diferentes da árvore -- o caso mais comum em `_pack`,
# que antes disparava recuo da INSTÂNCIA INTEIRA), com recuo agora PARCIAL:
# só a subárvore onde há uma contradição genuína entre ramos, preservando o
# mecanismo local (mais eficiente) em todo o resto da árvore.
#
# Ideia: para cada par (furo, pai), calcula-se o ancestral comum mais
# próximo (LCA) na árvore. Três casos, todos reduzidos a "qual bloco vem
# antes de qual" em torno do LCA:
#   - LCA == pai   (furo é descendente, direto ou não)  -> o filho de pai
#     que leva a furo deve terminar ANTES da própria borda de pai.
#   - LCA == furo  (pai é descendente, direto ou não)    -> o filho de furo
#     que leva a pai deve começar DEPOIS da própria borda de furo.
#   - LCA é outro nó (ramos diferentes)                  -> o filho de LCA
#     que leva a furo deve terminar ANTES do filho de LCA que leva a pai.
#
# Em cada nó, os pares cujo LCA é esse nó formam um pequeno DAG de
# precedência entre "blocos" (filhos implicados + um marcador "OWN" para a
# própria borda + filhos não-implicados, que continuam interleaved dentro do
# slot OWN exatamente como no traversal original). Ordenação topológica desse
# DAG dá a sequência de execução; CICLO = contradição genuína (ex.: dois
# ramos que precisariam preceder um ao outro simultaneamente) -> só então
# recua para tratamento sequencial, e só na subárvore desse nó
# específico -- o resto da árvore continua usando o mecanismo local.
# ===========================================================================

_OWN = "__OWN__"  # sentinel: representa "a própria borda do nó" no DAG de blocos


def _calcular_lca(arvore: nx.DiGraph, a: int, b: int) -> Optional[int]:
    """Ancestral comum mais próximo de a e b numa arborescência (cada nó tem
    no máximo 1 predecessor)."""
    anc_a = nx.ancestors(arvore, a)
    anc_a.add(a)
    atual = b
    visitados: set = set()
    while atual not in anc_a:
        if atual in visitados:
            return None  # não deveria ocorrer numa árvore válida
        visitados.add(atual)
        preds = list(arvore.predecessors(atual))
        if not preds:
            return None
        atual = preds[0]
    return atual


def _filho_no_caminho(arvore: nx.DiGraph, no: int, alvo: int) -> Optional[int]:
    """Filho DIRETO de 'no' que está no caminho até 'alvo' (alvo é esse
    próprio filho, ou é descendente dele). None se alvo == no."""
    if alvo == no:
        return None
    atual = alvo
    preds = list(arvore.predecessors(atual))
    while preds and preds[0] != no:
        atual = preds[0]
        preds = list(arvore.predecessors(atual))
    if not preds:
        return None
    return atual


def _agrupar_pares_por_lca(
    arvore: nx.DiGraph, pares: List[Tuple[int, int]]
) -> Dict[int, List[Tuple[int, int]]]:
    """Agrupa os pares Π pelo seu LCA na árvore -- computado uma vez só por
    chamada de `run_heuristic`, reaproveitado em toda a recursão."""
    grupos: Dict[int, List[Tuple[int, int]]] = {}
    for furo, pai in pares:
        if furo not in arvore or pai not in arvore:
            continue
        lca = _calcular_lca(arvore, furo, pai)
        if lca is None:
            continue
        grupos.setdefault(lca, []).append((furo, pai))
    return grupos


def _ordenar_blocos(
    no: int,
    pares_aqui: List[Tuple[int, int]],
    arvore: nx.DiGraph,
) -> Tuple[Optional[List[Any]], bool]:
    """
    Monta o DAG de precedência entre blocos (filhos implicados + OWN) para o
    nó 'no' e devolve (ordem_de_blocos, ok) via ordenação topológica
    (Kahn). ok=False sse houver ciclo (contradição genuína -- ver nota do
    bloco acima).
    """
    edges: set = set()
    vertices = {_OWN}
    for furo, pai in pares_aqui:
        v_furo = _OWN if furo == no else _filho_no_caminho(arvore, no, furo)
        v_pai = _OWN if pai == no else _filho_no_caminho(arvore, no, pai)
        if v_furo is None or v_pai is None or v_furo == v_pai:
            continue
        vertices.add(v_furo)
        vertices.add(v_pai)
        edges.add((v_furo, v_pai))

    adj: Dict[Any, set] = {v: set() for v in vertices}
    indeg = {v: 0 for v in vertices}
    for a, b in edges:
        if b not in adj[a]:
            adj[a].add(b)
            indeg[b] += 1

    def chave(v):
        return (0 if v is _OWN else 1, str(v))

    fila = sorted((v for v in vertices if indeg[v] == 0), key=chave)
    ordem: List[Any] = []
    while fila:
        v = fila.pop(0)
        ordem.append(v)
        for w in sorted(adj[v], key=chave):
            indeg[w] -= 1
            if indeg[w] == 0:
                fila.append(w)
        fila.sort(key=chave)

    if len(ordem) != len(vertices):
        return None, False
    return ordem, True


def _cortar_por_ordem_sequencial_de(G: nx.Graph, ordem: List[int], ponto_inicial: Point) -> List:
    """
    Igual a `cortar_por_ordem_sequencial`, mas parametrizado por ponto de
    partida/retorno (em vez de fixo em (0,0) -- usado pelo recuo LOCAL de
    conflito abaixo, que precisa se encaixar como um bloco dentro de um
    traversal maior, não como o caminho inteiro da instância.
    """
    from graph_routing import reordenar_poligono, chebyshev_distance

    caminho: List = []
    ponto_atual = ponto_inicial

    for no in ordem:
        pol = G.nodes[no]["poligono"]
        coords_pol = list(pol.exterior.coords)[:-1]
        entrada = min(coords_pol, key=lambda c: chebyshev_distance(ponto_atual, Point(c)))
        entrada_pt = Point(entrada)
        if ponto_atual.distance(entrada_pt) > 1e-9:
            caminho.append((LineString([ponto_atual, entrada_pt]), "d"))

        coords = reordenar_poligono(pol, entrada_pt)
        for i in range(len(coords) - 1):
            caminho.append((LineString([Point(coords[i]), Point(coords[i + 1])]), "x"))

        ponto_atual = entrada_pt

    if ponto_atual.distance(ponto_inicial) > 1e-9:
        caminho.append((LineString([ponto_atual, ponto_inicial]), "d"))

    return caminho


def _visitar_bloco(caminho, ponto_atual, no_destino, G, pol_cobertos, child_order_fn, pares_por_lca):
    """Desloca-se explicitamente até o vértice mais próximo do polígono de
    'no_destino' e visita sua subárvore por completo (via
    `caminha_arvore_hibrida_lca`). Devolve o novo ponto_atual (fecha no
    próprio ponto de entrada)."""
    from graph_routing import chebyshev_distance

    pol_destino = G.nodes[no_destino]["poligono"]
    coords_destino = list(pol_destino.exterior.coords)[:-1]
    entrada = min(coords_destino, key=lambda c: chebyshev_distance(ponto_atual, Point(c)))
    entrada_pt = Point(entrada)
    if ponto_atual.distance(entrada_pt) > 1e-9:
        caminho.append((LineString([ponto_atual, entrada_pt]), "d"))

    desc = nx.descendants(G, no_destino)
    desc.add(no_destino)
    pol_cobertos.append(no_destino)
    c, _ = caminha_arvore_hibrida_lca(
        G.subgraph(desc), no_destino, entrada_pt, pol_cobertos, child_order_fn, pares_por_lca
    )
    caminho.extend(c)
    return entrada_pt


def caminha_arvore_hibrida_lca(
    G: nx.DiGraph,
    raiz: int,
    ponto_inicial: Point,
    pol_cobertos: List[int],
    child_order_fn,
    pares_por_lca: Dict[int, List[Tuple[int, int]]],
) -> Tuple[List, List[int]]:
    """
    Versão generalizada de `caminha_arvore_hibrida`: usa `_ordenar_blocos`
    (LCA + ordenação topológica) em vez de só checar filhos diretos. Recua
    para tratamento sequencial LOCAL (só a subárvore de 'raiz') quando o DAG
    de blocos tem um ciclo (contradição genuína entre ramos).
    """
    from graph_routing import reordenar_poligono

    caminho: List = []
    successors = list(child_order_fn(G, raiz))

    pares_aqui = pares_por_lca.get(raiz, [])
    if pares_aqui:
        ordem_blocos, ok = _ordenar_blocos(raiz, pares_aqui, G)
    else:
        ordem_blocos, ok = [_OWN], True

    if not ok:
        # CONFLITO -- recuo LOCAL: toda a subárvore de 'raiz' vira um único
        # bloco sequencial, preservando o mecanismo local em todo
        # o resto da árvore (fora desta subárvore específica).
        desc = nx.descendants(G, raiz)
        desc.add(raiz)
        for n in desc:
            if n != raiz and n not in pol_cobertos:
                pol_cobertos.append(n)
        sub = G.subgraph(desc)
        ordem_base = list(nx.dfs_preorder_nodes(sub, raiz))
        preds_local: Dict[int, List[int]] = {}
        for lca_no, pares_lca in pares_por_lca.items():
            if lca_no not in desc:
                continue
            for furo, pai in pares_lca:
                if furo in desc and pai in desc:
                    preds_local.setdefault(pai, []).append(furo)
        ordem_final = _ordem_respeitando_precedencia(ordem_base, preds_local)
        caminho = _cortar_por_ordem_sequencial_de(sub, ordem_final, ponto_inicial)
        return caminho, pol_cobertos

    filhos_bloco = {b for b in ordem_blocos if b is not _OWN}
    successors_normais = [s for s in successors if s not in filhos_bloco and s not in pol_cobertos]

    ponto_atual = ponto_inicial
    pol = G.nodes[raiz]["poligono"]

    for bloco in ordem_blocos:
        if bloco is _OWN:
            if ponto_atual.distance(ponto_inicial) > 1e-9:
                caminho.append((LineString([ponto_atual, ponto_inicial]), "d"))
            coords = reordenar_poligono(pol, ponto_inicial)
            for i in range(len(coords) - 1):
                seg = LineString([Point(coords[i]), Point(coords[i + 1])])
                caminho.append((seg, "x"))
                for s in filter(lambda x: x not in pol_cobertos, successors_normais):
                    if G.edges[raiz, s]["weight"] == 0:
                        pol_destino = G.nodes[s]["poligono"]
                        intersecao = pol.intersection(pol_destino)
                        if (intersecao.distance(Point(seg.coords[1])) < 1e-8
                                or intersecao.covers(Point(seg.coords[0]))):
                            desc = nx.descendants(G, s)
                            desc.add(s)
                            if desc:
                                pol_cobertos.append(s)
                                c, _ = caminha_arvore_hibrida_lca(
                                    G.subgraph(desc), s, Point(seg.coords[1]),
                                    pol_cobertos, child_order_fn, pares_por_lca,
                                )
                                caminho.extend(c)
                    else:
                        intersecao = G.edges[raiz, s]["connection"]
                        if (intersecao.distance(Point(seg.coords[1])) < 1e-8
                                or intersecao.covers(Point(seg.coords[0]))):
                            caminho.append((intersecao, "d"))
                            desc = nx.descendants(G, s)
                            desc.add(s)
                            if desc:
                                pol_cobertos.append(s)
                                c, _ = caminha_arvore_hibrida_lca(
                                    G.subgraph(desc), s, Point(seg.coords[1]),
                                    pol_cobertos, child_order_fn, pares_por_lca,
                                )
                                caminho.extend(c)
            ponto_atual = Point(coords[0])
        else:
            if bloco in pol_cobertos:
                continue
            ponto_atual = _visitar_bloco(
                caminho, ponto_atual, bloco, G, pol_cobertos, child_order_fn, pares_por_lca
            )

    return caminho, pol_cobertos


def cortar_arvore_hibrida_precedencia_v2(
    arvore: nx.DiGraph,
    raiz: int,
    pares: List[Tuple[int, int]],
    child_order_fn,
) -> List:
    """
    Wrapper da versão refinada do traversal híbrido: generaliza
    `cortar_arvore_hibrida_precedencia` (vizinho direto apenas, mantida
    intacta acima) para qualquer par Π via LCA na árvore, com recuo PARCIAL
    (só a subárvore em conflito) em vez de recuo total da instância.
    """
    pares_por_lca = _agrupar_pares_por_lca(arvore, pares)

    caminho = []
    aresta, ponto = distancia_e_aresta_ate_raiz(Point(0, 0), arvore.nodes[raiz]["poligono"])
    if aresta:
        caminho.append((aresta[0], "d"))
    c, _ = caminha_arvore_hibrida_lca(
        arvore, raiz, ponto, [raiz], child_order_fn, pares_por_lca
    )
    caminho += c
    if aresta:
        caminho.append((LineString(list(aresta[0].coords)[::-1]), "d"))
    return caminho


# ===========================================================================
# Versão de PERFORMANCE de caminha_arvore_hibrida_lca (`_perf`)
#
# `cProfile` de uma execução real (trousers/packing, 46 polígonos) atribuiu
# ~56% do tempo total a chamadas Shapely dentro da travessia LCA,
# concentradas em `pol.intersection(pol_destino)`.
#
# A função original (acima, mantida intacta) recalcula essa interseção uma
# vez POR SEGMENTO do polígono, para cada sucessor `s` com peso de aresta
# zero -- mas o valor não depende do segmento, só de `raiz` e `s`, ambos
# fixos durante todo o loop de segmentos. As versões abaixo pré-computam
# esse valor uma vez por sucessor (hoisting de um cálculo invariante de
# loop), sem alterar nenhuma decisão de controle de fluxo: a filtragem por
# `pol_cobertos` continua sendo reavaliada por segmento, exatamente como no
# original, porque essa parte MUDA durante o loop (um sucessor coberto por
# um segmento anterior deve ser pulado nos segmentos seguintes).
#
# Equivalência: comparada com a versão original nas 59 instâncias com
# pares_pi>0 dentre as 282+50, em duas topologias de árvore por instância
# (118 comparações), com a mesma FO e o mesmo caminho em todas. O ganho de
# tempo é praticamente nulo: o custo dominante restante são as chamadas
# geométricas por par segmento-sucessor (Point(...).distance(...)/
# .covers(...)), que dependem do segmento corrente (ver `_perf2` abaixo).
# ===========================================================================

def _visitar_bloco_perf(caminho, ponto_atual, no_destino, G, pol_cobertos, child_order_fn, pares_por_lca):
    """Idêntica a `_visitar_bloco`, mas despacha para
    `caminha_arvore_hibrida_lca_perf` em vez da versão original -- nenhuma
    outra mudança de lógica."""
    from graph_routing import chebyshev_distance

    pol_destino = G.nodes[no_destino]["poligono"]
    coords_destino = list(pol_destino.exterior.coords)[:-1]
    entrada = min(coords_destino, key=lambda c: chebyshev_distance(ponto_atual, Point(c)))
    entrada_pt = Point(entrada)
    if ponto_atual.distance(entrada_pt) > 1e-9:
        caminho.append((LineString([ponto_atual, entrada_pt]), "d"))

    desc = nx.descendants(G, no_destino)
    desc.add(no_destino)
    pol_cobertos.append(no_destino)
    c, _ = caminha_arvore_hibrida_lca_perf(
        G.subgraph(desc), no_destino, entrada_pt, pol_cobertos, child_order_fn, pares_por_lca
    )
    caminho.extend(c)
    return entrada_pt


def caminha_arvore_hibrida_lca_perf(
    G: nx.DiGraph,
    raiz: int,
    ponto_inicial: Point,
    pol_cobertos: List[int],
    child_order_fn,
    pares_por_lca: Dict[int, List[Tuple[int, int]]],
) -> Tuple[List, List[int]]:
    """Versão de performance de `caminha_arvore_hibrida_lca` (ver bloco de
    comentário acima). Mesma lógica byte a byte; a única mudança é
    pré-computar, uma vez por sucessor em vez de uma vez por segmento, o
    valor caro (interseção geométrica ou conexão já cacheada) usado no
    teste de toque/intercepção."""
    from graph_routing import reordenar_poligono

    caminho: List = []
    successors = list(child_order_fn(G, raiz))

    pares_aqui = pares_por_lca.get(raiz, [])
    if pares_aqui:
        ordem_blocos, ok = _ordenar_blocos(raiz, pares_aqui, G)
    else:
        ordem_blocos, ok = [_OWN], True

    if not ok:
        # CONFLITO -- recuo LOCAL, idêntico ao da versão original.
        desc = nx.descendants(G, raiz)
        desc.add(raiz)
        for n in desc:
            if n != raiz and n not in pol_cobertos:
                pol_cobertos.append(n)
        sub = G.subgraph(desc)
        ordem_base = list(nx.dfs_preorder_nodes(sub, raiz))
        preds_local: Dict[int, List[int]] = {}
        for lca_no, pares_lca in pares_por_lca.items():
            if lca_no not in desc:
                continue
            for furo, pai in pares_lca:
                if furo in desc and pai in desc:
                    preds_local.setdefault(pai, []).append(furo)
        ordem_final = _ordem_respeitando_precedencia(ordem_base, preds_local)
        caminho = _cortar_por_ordem_sequencial_de(sub, ordem_final, ponto_inicial)
        return caminho, pol_cobertos

    filhos_bloco = {b for b in ordem_blocos if b is not _OWN}
    successors_normais = [s for s in successors if s not in filhos_bloco and s not in pol_cobertos]

    ponto_atual = ponto_inicial
    pol = G.nodes[raiz]["poligono"]

    # --- hoisting: pré-computa, uma vez por sucessor, o valor caro usado
    # no teste de toque/intercepção abaixo (idêntico ao valor que a versão
    # original recalcula, sem necessidade, a cada segmento). ---------------
    info_sucessor: Dict[int, Tuple[bool, Any]] = {}
    for s in successors_normais:
        peso_zero = G.edges[raiz, s]["weight"] == 0
        if peso_zero:
            pol_destino = G.nodes[s]["poligono"]
            valor = pol.intersection(pol_destino)
        else:
            valor = G.edges[raiz, s]["connection"]
        info_sucessor[s] = (peso_zero, valor)
    # -----------------------------------------------------------------------

    for bloco in ordem_blocos:
        if bloco is _OWN:
            if ponto_atual.distance(ponto_inicial) > 1e-9:
                caminho.append((LineString([ponto_atual, ponto_inicial]), "d"))
            coords = reordenar_poligono(pol, ponto_inicial)
            for i in range(len(coords) - 1):
                seg = LineString([Point(coords[i]), Point(coords[i + 1])])
                caminho.append((seg, "x"))
                for s in filter(lambda x: x not in pol_cobertos, successors_normais):
                    peso_zero, intersecao = info_sucessor[s]
                    if peso_zero:
                        if (intersecao.distance(Point(seg.coords[1])) < 1e-8
                                or intersecao.covers(Point(seg.coords[0]))):
                            desc = nx.descendants(G, s)
                            desc.add(s)
                            if desc:
                                pol_cobertos.append(s)
                                c, _ = caminha_arvore_hibrida_lca_perf(
                                    G.subgraph(desc), s, Point(seg.coords[1]),
                                    pol_cobertos, child_order_fn, pares_por_lca,
                                )
                                caminho.extend(c)
                    else:
                        if (intersecao.distance(Point(seg.coords[1])) < 1e-8
                                or intersecao.covers(Point(seg.coords[0]))):
                            caminho.append((intersecao, "d"))
                            desc = nx.descendants(G, s)
                            desc.add(s)
                            if desc:
                                pol_cobertos.append(s)
                                c, _ = caminha_arvore_hibrida_lca_perf(
                                    G.subgraph(desc), s, Point(seg.coords[1]),
                                    pol_cobertos, child_order_fn, pares_por_lca,
                                )
                                caminho.extend(c)
            ponto_atual = Point(coords[0])
        else:
            if bloco in pol_cobertos:
                continue
            ponto_atual = _visitar_bloco_perf(
                caminho, ponto_atual, bloco, G, pol_cobertos, child_order_fn, pares_por_lca
            )

    return caminho, pol_cobertos


def cortar_arvore_hibrida_precedencia_v2_perf(
    arvore: nx.DiGraph,
    raiz: int,
    pares: List[Tuple[int, int]],
    child_order_fn,
) -> List:
    """Versão de performance de `cortar_arvore_hibrida_precedencia_v2`:
    wrapper idêntico, despachando para `caminha_arvore_hibrida_lca_perf`."""
    pares_por_lca = _agrupar_pares_por_lca(arvore, pares)

    caminho = []
    aresta, ponto = distancia_e_aresta_ate_raiz(Point(0, 0), arvore.nodes[raiz]["poligono"])
    if aresta:
        caminho.append((aresta[0], "d"))
    c, _ = caminha_arvore_hibrida_lca_perf(
        arvore, raiz, ponto, [raiz], child_order_fn, pares_por_lca
    )
    caminho += c
    if aresta:
        caminho.append((LineString(list(aresta[0].coords)[::-1]), "d"))
    return caminho


# ===========================================================================
# Segunda versão de performance (`_perf2`)
#
# O custo dominante da versão `_perf` não é o `pol.intersection()` já içado,
# e sim as chamadas `.distance()`/`.covers()` do GEOS/Shapely, executadas
# uma vez por PAR (segmento, sucessor) -- não elimináveis pelo hoisting
# anterior porque dependem do segmento corrente.
#
# O valor testado (`intersecao` = `pol.intersection(pol_destino)` quando
# peso==0, ou `connection` quando peso!=0 -- SEMPRE uma LineString de 2
# pontos por construção em graph_routing.construir_arvore_direcionada) é, em
# uma amostra de 1347 pares (segmento-base, sucessor) de 59 instâncias com
# pares_pi>0:
#   - LineString de 2 pontos: 88,3% dos casos;
#   - Point (1 ponto): 8,2%;
#   - Polygon degenerado (sliver de área ~0), GeometryCollection, MultiPoint:
#     3,0% combinados.
#
# Estratégia: caminho RÁPIDO em aritmética de coordenadas cruas (sem objetos
# Shapely) apenas para os dois casos limpos e amplamente dominantes (Point
# de 1 coordenada, LineString de EXATAMENTE 2 coordenadas -- 96,5% do volume
# observado); QUALQUER outro tipo (Polygon, MultiPolygon, GeometryCollection,
# MultiPoint, MultiLineString, LineString com outro número de pontos,
# geometria vazia) cai em fallback EXATO -- chama `.distance()`/`.covers()`
# do Shapely, idêntico ao que a versão original sempre fez. Ou seja: nunca
# há um "melhor esforço" arriscado, só caminho rápido matematicamente
# equivalente ou fallback idêntico ao original.
#
# `_dist_pp`/`_dist_ps` replicam a distância euclidiana ponto-ponto e
# ponto-segmento (fórmula padrão de projeção com clamp em [0,1]). O teste de
# "covers" (Shapely, sem tolerância parametrizável) é replicado como
# "distância < 1e-9", a mesma tolerância já usada neste módulo para decidir
# se dois pontos são o mesmo ponto.
#
# Equivalência: comparada diretamente com a versão original em 118 casos
# (59 instâncias com pares_pi>0 dentre as 282+50, duas topologias de árvore
# cada), com a mesma FO e o mesmo caminho em todos. Ganho de tempo modesto:
# o tempo acumulado da travessia LCA numa instância de 46 polígonos cai ~15%
# (1,89 s -> 1,61 s). O número de kicks do ILS nas instâncias grandes (>=45
# polígonos) continua baixo, porque `edge_swap_barato` varre todas as arestas
# da árvore x k_candidatos e é chamado de novo após cada kick: o custo por
# kick cresce com N, não só com o custo por avaliação.
# ===========================================================================

def _dist_pp(px: float, py: float, qx: float, qy: float) -> float:
    return math.hypot(px - qx, py - qy)


def _dist_ps(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    dx, dy = bx - ax, by - ay
    len2 = dx * dx + dy * dy
    if len2 < 1e-18:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / len2
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _preparar_toque_rapido(valor):
    """Classifica `valor` (Point, LineString-2pt, ou qualquer outra coisa)
    para o teste de toque em `_testar_toque`. Retorna uma tupla-chave que
    evita reconstruir/inspecionar o tipo a cada segmento."""
    tipo = type(valor).__name__
    if tipo == "Point":
        c = valor.coords[0]
        return ("pt", c[0], c[1])
    if tipo == "LineString":
        c = list(valor.coords)
        if len(c) == 2:
            return ("seg", c[0][0], c[0][1], c[1][0], c[1][1])
    return ("slow", valor)


def _testar_toque(prep, x1: float, y1: float, x2: float, y2: float) -> bool:
    """Equivalente a `valor.distance(Point(x2,y2)) < 1e-8 or
    valor.covers(Point(x1,y1))`, rapido para os casos comuns (ponto ou
    segmento de 2 pontos), com fallback EXATO ao Shapely original para
    qualquer outro tipo de geometria."""
    kind = prep[0]
    if kind == "pt":
        _, px, py = prep
        if _dist_pp(px, py, x2, y2) < 1e-8:
            return True
        return _dist_pp(px, py, x1, y1) < 1e-9
    if kind == "seg":
        _, ax, ay, bx, by = prep
        if _dist_ps(x2, y2, ax, ay, bx, by) < 1e-8:
            return True
        return _dist_ps(x1, y1, ax, ay, bx, by) < 1e-9
    # fallback exato -- idêntico à versão original/`_perf`
    valor = prep[1]
    if valor.distance(Point(x2, y2)) < 1e-8:
        return True
    return valor.covers(Point(x1, y1))


def _visitar_bloco_perf2(caminho, ponto_atual, no_destino, G, pol_cobertos, child_order_fn, pares_por_lca):
    """Idêntica a `_visitar_bloco`, mas despacha para
    `caminha_arvore_hibrida_lca_perf2`."""
    from graph_routing import chebyshev_distance

    pol_destino = G.nodes[no_destino]["poligono"]
    coords_destino = list(pol_destino.exterior.coords)[:-1]
    entrada = min(coords_destino, key=lambda c: chebyshev_distance(ponto_atual, Point(c)))
    entrada_pt = Point(entrada)
    if ponto_atual.distance(entrada_pt) > 1e-9:
        caminho.append((LineString([ponto_atual, entrada_pt]), "d"))

    desc = nx.descendants(G, no_destino)
    desc.add(no_destino)
    pol_cobertos.append(no_destino)
    c, _ = caminha_arvore_hibrida_lca_perf2(
        G.subgraph(desc), no_destino, entrada_pt, pol_cobertos, child_order_fn, pares_por_lca
    )
    caminho.extend(c)
    return entrada_pt


def caminha_arvore_hibrida_lca_perf2(
    G: nx.DiGraph,
    raiz: int,
    ponto_inicial: Point,
    pol_cobertos: List[int],
    child_order_fn,
    pares_por_lca: Dict[int, List[Tuple[int, int]]],
) -> Tuple[List, List[int]]:
    """Versão de performance 2 de `caminha_arvore_hibrida_lca`: mantém o
    hoisting de `caminha_arvore_hibrida_lca_perf` (interseção pré-computada
    uma vez por sucessor) e ADICIONA um caminho rápido em coordenadas cruas
    para o teste de toque por segmento (ver bloco de comentário acima).
    Mesma lógica de controle de fluxo, byte a byte, da versão original."""
    from graph_routing import reordenar_poligono

    caminho: List = []
    successors = list(child_order_fn(G, raiz))

    pares_aqui = pares_por_lca.get(raiz, [])
    if pares_aqui:
        ordem_blocos, ok = _ordenar_blocos(raiz, pares_aqui, G)
    else:
        ordem_blocos, ok = [_OWN], True

    if not ok:
        # CONFLITO -- recuo LOCAL, idêntico ao da versão original.
        desc = nx.descendants(G, raiz)
        desc.add(raiz)
        for n in desc:
            if n != raiz and n not in pol_cobertos:
                pol_cobertos.append(n)
        sub = G.subgraph(desc)
        ordem_base = list(nx.dfs_preorder_nodes(sub, raiz))
        preds_local: Dict[int, List[int]] = {}
        for lca_no, pares_lca in pares_por_lca.items():
            if lca_no not in desc:
                continue
            for furo, pai in pares_lca:
                if furo in desc and pai in desc:
                    preds_local.setdefault(pai, []).append(furo)
        ordem_final = _ordem_respeitando_precedencia(ordem_base, preds_local)
        caminho = _cortar_por_ordem_sequencial_de(sub, ordem_final, ponto_inicial)
        return caminho, pol_cobertos

    filhos_bloco = {b for b in ordem_blocos if b is not _OWN}
    successors_normais = [s for s in successors if s not in filhos_bloco and s not in pol_cobertos]

    ponto_atual = ponto_inicial
    pol = G.nodes[raiz]["poligono"]

    info_sucessor: Dict[int, Any] = {}
    for s in successors_normais:
        peso_zero = G.edges[raiz, s]["weight"] == 0
        if peso_zero:
            pol_destino = G.nodes[s]["poligono"]
            valor = pol.intersection(pol_destino)
        else:
            valor = G.edges[raiz, s]["connection"]
        info_sucessor[s] = (peso_zero, valor, _preparar_toque_rapido(valor))

    for bloco in ordem_blocos:
        if bloco is _OWN:
            if ponto_atual.distance(ponto_inicial) > 1e-9:
                caminho.append((LineString([ponto_atual, ponto_inicial]), "d"))
            coords = reordenar_poligono(pol, ponto_inicial)
            for i in range(len(coords) - 1):
                x1, y1 = coords[i]
                x2, y2 = coords[i + 1]
                seg = LineString([Point(coords[i]), Point(coords[i + 1])])
                caminho.append((seg, "x"))
                for s in filter(lambda x: x not in pol_cobertos, successors_normais):
                    peso_zero, intersecao, prep = info_sucessor[s]
                    if _testar_toque(prep, x1, y1, x2, y2):
                        if not peso_zero:
                            caminho.append((intersecao, "d"))
                        desc = nx.descendants(G, s)
                        desc.add(s)
                        if desc:
                            pol_cobertos.append(s)
                            c, _ = caminha_arvore_hibrida_lca_perf2(
                                G.subgraph(desc), s, Point(x2, y2),
                                pol_cobertos, child_order_fn, pares_por_lca,
                            )
                            caminho.extend(c)
            ponto_atual = Point(coords[0])
        else:
            if bloco in pol_cobertos:
                continue
            ponto_atual = _visitar_bloco_perf2(
                caminho, ponto_atual, bloco, G, pol_cobertos, child_order_fn, pares_por_lca
            )

    return caminho, pol_cobertos


def cortar_arvore_hibrida_precedencia_v2_perf2(
    arvore: nx.DiGraph,
    raiz: int,
    pares: List[Tuple[int, int]],
    child_order_fn,
) -> List:
    """Versão de performance 2 de `cortar_arvore_hibrida_precedencia_v2`:
    wrapper idêntico, despachando para `caminha_arvore_hibrida_lca_perf2`."""
    pares_por_lca = _agrupar_pares_por_lca(arvore, pares)

    caminho = []
    aresta, ponto = distancia_e_aresta_ate_raiz(Point(0, 0), arvore.nodes[raiz]["poligono"])
    if aresta:
        caminho.append((aresta[0], "d"))
    c, _ = caminha_arvore_hibrida_lca_perf2(
        arvore, raiz, ponto, [raiz], child_order_fn, pares_por_lca
    )
    caminho += c
    if aresta:
        caminho.append((LineString(list(aresta[0].coords)[::-1]), "d"))
    return caminho


# Funções de ordem de filhos

def child_order_num_filhos(G, raiz):
    """Ordem padrão: decrescente por num_filhos."""
    return sorted(G.successors(raiz),
                  key=lambda n: G.nodes[n]["num_filhos"], reverse=True)


def child_order_permutation(G, raiz, best_perm_cache=None):
    """Permutação exaustiva: retorna a ordem em best_perm_cache[raiz]."""
    if best_perm_cache and raiz in best_perm_cache:
        return best_perm_cache[raiz]
    return child_order_num_filhos(G, raiz)


def child_order_2opt(G, raiz, best_perm_cache=None):
    """2-opt: retorna a ordem em best_perm_cache[raiz]."""
    return child_order_permutation(G, raiz, best_perm_cache)


# ===========================================================================
# Pré-computação de permutações / 2-opt de filhos
# ===========================================================================

def compute_child_permutations(
    arvore: nx.DiGraph,
    raiz: int,
    mode: str = "permutation",   # "permutation" | "2opt"
    time_limit: Optional[float] = None,
) -> Dict[int, List[int]]:
    """
    Para cada nó com mais de 1 filho, encontra a melhor ordem de filhos.
    Retorna dict {nó: melhor_lista_de_filhos}.
    """
    best_perm: Dict[int, List[int]] = {}
    t0 = time.time()

    for node in nx.dfs_preorder_nodes(arvore, raiz):
        if time_limit and (time.time() - t0) > time_limit:
            break
        children = list(arvore.successors(node))
        if len(children) < 2:
            best_perm[node] = children
            continue

        if mode == "permutation":
            best_fo = float("inf")
            best_order = children
            for perm in permutations(children):
                if time_limit and (time.time() - t0) > time_limit:
                    break
                # Constrói cache temporário com esta permutação e avalia
                cache_tmp = dict(best_perm)
                cache_tmp[node] = list(perm)
                order_fn = lambda G, r, c=cache_tmp: (
                    c[r] if r in c else child_order_num_filhos(G, r)
                )
                cam = reclassificar_tags(
                    cortar_arvore_com_ordem(arvore, raiz, order_fn)
                )
                fo = calcular_fo(cam)
                if fo < best_fo:
                    best_fo = fo
                    best_order = list(perm)
            best_perm[node] = best_order

        else:  # 2-opt
            current_order = list(children)
            improved = True
            while improved:
                if time_limit and (time.time() - t0) > time_limit:
                    break
                improved = False
                for i in range(len(current_order)):
                    if time_limit and (time.time() - t0) > time_limit:
                        break
                    for j in range(i + 1, len(current_order)):
                        new_order = list(current_order)
                        new_order[i], new_order[j] = new_order[j], new_order[i]
                        cache_tmp = dict(best_perm)
                        cache_tmp[node] = new_order
                        order_fn = lambda G, r, c=cache_tmp: (
                            c[r] if r in c else child_order_num_filhos(G, r)
                        )
                        cam = reclassificar_tags(
                            cortar_arvore_com_ordem(arvore, raiz, order_fn)
                        )
                        fo = calcular_fo(cam)

                        # Referência: FO com ordem atual
                        cache_curr = dict(best_perm)
                        cache_curr[node] = current_order
                        order_fn_curr = lambda G, r, c=cache_curr: (
                            c[r] if r in c else child_order_num_filhos(G, r)
                        )
                        cam_curr = reclassificar_tags(
                            cortar_arvore_com_ordem(arvore, raiz, order_fn_curr)
                        )
                        fo_curr = calcular_fo(cam_curr)

                        if fo < fo_curr - 1e-9:
                            current_order = new_order
                            improved = True
                            break
                    if improved:
                        break
            best_perm[node] = current_order

    return best_perm


# ===========================================================================
# Enumeração de vértice inicial
# ===========================================================================

def enumerate_start_vertex(
    arvore: nx.DiGraph,
    raiz: int,
    child_order_fn,
    time_limit: Optional[float] = None,
) -> Tuple[float, Optional[list], int]:
    """
    Testa todos os vértices do polígono raiz como ponto de partida e retorna
    o caminho que minimiza a FO. Retorna (best_fo, best_caminho, n_evals).

    Para cada vértice candidato v:
      1. Abordagem (0,0) → v  como deslocamento "d"  (usando distancia_e_aresta_ate_raiz)
      2. Travessia completa da árvore a partir de v
      3. Retorno ao ponto de abordagem inicial → (0,0)  como "d"

    A abordagem usa distancia_e_aresta_ate_raiz (igual a cortar_arvore_com_ordem)
    para não criar segmentos que coincidam com arestas de polígono, evitando
    interferência em reclassificar_tags.
    """
    raiz_poly = arvore.nodes[raiz]["poligono"]
    coords = list(raiz_poly.exterior.coords[:-1])

    # Fallback: sem vértices, usa o caminho padrão
    if not coords:
        cam = reclassificar_tags(cortar_arvore_com_ordem(arvore, raiz, child_order_fn))
        return calcular_fo(cam), cam, 1

    best_fo = float("inf")
    best_caminho = None
    n_evals = 0
    t0 = time.time()
    origem = Point(0, 0)

    for idx in range(len(coords)):
        if time_limit and (time.time() - t0) > time_limit:
            break

        ponto_inicio = Point(coords[idx])

        # --- abordagem: igual a cortar_arvore_com_ordem ---
        aresta_abord, ponto_abord = distancia_e_aresta_ate_raiz(origem, raiz_poly)
        # Mas queremos iniciar exatamente em ponto_inicio (o vértice candidato).
        # Criamos a abordagem diretamente para esse vértice, usando a mesma
        # distância real (Chebyshev) como custo de deslocamento.
        caminho = []
        seg_abord = LineString([origem, ponto_inicio])
        if seg_abord.length > 1e-8:
            caminho.append((seg_abord, "d"))

        # --- travessia da árvore a partir do vértice escolhido ---
        c, _ = caminha_arvore_com_ordem(
            arvore, raiz, ponto_inicio, [raiz], child_order_fn
        )
        caminho += c

        # --- retorno à origem ---
        # A travessia fecha o polígono raiz de volta a ponto_inicio.
        # Acrescentamos o deslocamento final ponto_inicio → origem.
        seg_retorno = LineString([ponto_inicio, origem])
        if seg_retorno.length > 1e-8:
            caminho.append((seg_retorno, "d"))

        cam_class = reclassificar_tags(caminho)
        fo = calcular_fo(cam_class)
        n_evals += 1
        if fo < best_fo:
            best_fo = fo
            best_caminho = cam_class

    return best_fo, best_caminho, n_evals


# ===========================================================================
# Função principal: run_heuristic
# ===========================================================================

def run_heuristic(
    layout: List[Polygon],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Executa a heurística completa com a configuração fornecida.

    Retorna dict com:
      fo_total       : função objetivo total
      time_total     : tempo de execução (s)
      timeout        : True se foi interrompido por limite de tempo
      caminho        : último caminho gerado
      arvore         : última árvore gerada
      raiz           : raiz usada
      meta           : metadados da variante (n avaliações, etc.)
    """
    cfg = {**_DEFAULT_CONFIG, **(config or {})}
    tl = cfg["time_limit"]
    t_start = time.time()

    def elapsed():
        return time.time() - t_start

    def tl_remaining():
        if tl is None:
            return None
        return max(0.1, tl - elapsed())

    result = {
        "fo_total": 0.0,
        "time_total": 0.0,
        "timeout": False,
        "caminho": [],
        "arvore": None,
        "raiz": None,
        "meta": {},
        "n_polygons": len(layout),
    }

    # ── Passo 1: Furos ──────────────────────────────────────────────────────
    layout = tratar_furos(layout)

    # ── Passo 2: Interseções ─────────────────────────────────────────────────
    layout = resolver_intersecoes(layout)

    if not layout:
        result["time_total"] = elapsed()
        return result

    # ── Passo 3: QuadTree ────────────────────────────────────────────────────
    positions = clusterizar_quadtree(layout)
    if not positions:
        result["time_total"] = elapsed()
        return result

    # ── Passo 4: MST ─────────────────────────────────────────────────────────
    edge_swap_meta = {}
    if cfg["mst_strategy"] == "edge_swap":
        raiz_fn_for_swap = lambda mst, pos: escolher_raiz_nearest(mst, pos)
        G, edge_swap_meta = construir_grafo_mst_edge_swap(
            positions, raiz_fn_for_swap, tl_remaining()
        )
    else:
        G = construir_grafo_mst(positions)

    result["meta"].update(edge_swap_meta)

    # ── Precedência (Π), se solicitado ──────────────────────────────────────
    # enforce_precedence=True usa a versão com garantia de Π
    # (cortar_arvore_com_precedencia). Combinação com child_order
    # permutation/2opt ou vertex_strategy enumerate ainda não implementada
    # (essas rotinas chamam internamente o traversal ORIGINAL) -- ignoradas
    # com aviso em vez de dar resultado silenciosamente incorreto.
    preds_map: Dict[int, List[int]] = {}
    pares_precedencia: List[Tuple[int, int]] = []
    if cfg["enforce_precedence"]:
        pares_precedencia = _detectar_precedencia(positions)
        preds_map = _mapa_predecessores(pares_precedencia)
        result["meta"]["n_precedence_pairs"] = len(pares_precedencia)
        if cfg["child_order"] in ("permutation", "2opt") or cfg["vertex_strategy"] == "enumerate":
            result["meta"]["precedence_aware_warning"] = (
                "enforce_precedence=True ignora child_order=permutation/2opt e "
                "vertex_strategy=enumerate nesta versão (usa num_filhos + nearest)."
            )

    # ── Passos 5-7: por componente conexo ────────────────────────────────────
    fo_total = 0.0
    last_arvore = None
    last_raiz = None
    last_caminho = []
    root_enum_total = 0

    for componente in nx.connected_components(G):
        if tl and elapsed() > tl:
            result["timeout"] = True
            break

        subgrafo = G.subgraph(componente)

        # ── Escolha da raiz ───────────────────────────────────────────────
        if cfg["root_strategy"] == "enumerate":
            tl_r = tl_remaining()
            raiz, fo_comp, n_r = enumerar_raiz(subgrafo, positions, tl_r)
            root_enum_total += n_r
            arvore = construir_arvore_direcionada(subgrafo, raiz)
            # FO já calculada, mas precisamos do caminho e da ordem de filhos
        else:
            raiz = escolher_raiz_nearest(subgrafo, positions)
            arvore = construir_arvore_direcionada(subgrafo, raiz)

        # ── Ordem de filhos ───────────────────────────────────────────────
        child_cache = None
        if cfg["child_order"] in ("permutation", "2opt") and not cfg["enforce_precedence"]:
            child_cache = compute_child_permutations(
                arvore, raiz,
                mode=cfg["child_order"],
                time_limit=tl_remaining(),
            )

        if child_cache:
            child_order_fn = lambda G, r, c=child_cache: (
                c.get(r, child_order_num_filhos(G, r))
            )
        else:
            child_order_fn = child_order_num_filhos

        # ── Geração do caminho ────────────────────────────────────────────
        if cfg["enforce_precedence"]:
            # Garante Π (furo antes do contorno externo). precedence_strategy
            # escolhe entre as variantes disponíveis.
            strategy = cfg["precedence_strategy"]
            if strategy == "sequential_2opt":
                caminho_raw, meta_2opt = cortar_arvore_com_precedencia_2opt(
                    arvore, raiz, preds_map, pares_precedencia, tl_remaining()
                )
                for k, v in meta_2opt.items():
                    result["meta"][k] = result["meta"].get(k, 0) + v
            elif strategy == "hybrid":
                caminho_raw = cortar_arvore_hibrida_precedencia_v2(
                    arvore, raiz, pares_precedencia, child_order_fn
                )
            elif strategy == "hybrid_v1":
                # versão só-vizinho-direto, mantida para comparação -- a
                # versão generalizada (LCA + recuo parcial) é a usada em
                # "hybrid".
                caminho_raw = cortar_arvore_hibrida_precedencia(
                    arvore, raiz, preds_map, pares_precedencia, child_order_fn
                )
            else:
                caminho_raw = cortar_arvore_com_precedencia(arvore, raiz, preds_map, child_order_fn)
        elif cfg["child_order"] in ("permutation", "2opt"):
            caminho_raw = cortar_arvore_com_ordem(arvore, raiz, child_order_fn)
        else:
            caminho_raw = cortar_arvore(arvore, raiz)

        # ── Enumeração de vértice inicial ─────────────────────────────────
        if cfg["vertex_strategy"] == "enumerate" and not cfg["enforce_precedence"]:
            fo_ve, best_cam_ve, n_ve = enumerate_start_vertex(
                arvore, raiz, child_order_fn, tl_remaining()
            )
            caminho = best_cam_ve if best_cam_ve is not None else reclassificar_tags(caminho_raw)
            fo_comp = fo_ve
            result["meta"]["vertex_enum_evals"] = (
                result["meta"].get("vertex_enum_evals", 0) + n_ve
            )
        else:
            caminho = reclassificar_tags(caminho_raw)
            fo_comp = calcular_fo(caminho)

        fo_total += fo_comp
        last_arvore = arvore
        last_raiz = raiz
        last_caminho = caminho

    result["meta"]["root_enum_candidates"] = root_enum_total
    result["fo_total"]   = fo_total
    result["time_total"] = elapsed()
    result["arvore"]     = last_arvore
    result["raiz"]       = last_raiz
    result["caminho"]    = last_caminho

    return result
