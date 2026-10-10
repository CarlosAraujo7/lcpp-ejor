"""
converter.py
============
Converte uma instância LCPP (lista de segmentos, formato do projeto) para o
formato .pcglns (entrada do solver PCGLNS, ver external_tools/pcglns/), fazendo
a redução real LCPP -> PCGTSP (GTSP + precedência entre grupos).

Redução usada
-------------
- Mesmo pré-processamento da heurística (tratar_furos + resolver_intersecoes +
  clusterizar_quadtree, de src/heuristic_core.py) é aplicado
  primeiro, para que o layout geométrico convertido seja IDÊNTICO ao que a
  nossa heurística de fato usa -- garante que a comparação de FO é apples-to-
  apples (mesmos polígonos, mesma ordem).
- Cada contorno resultante = um grupo/set GTSP.
- Cidades candidatas de um grupo = os vértices desse polígono (um candidato
  de entrada/saída por vértice -- é exatamente o espaço de escolha que a
  heurística já usa dinamicamente via `reordenar_poligono`, que sempre entra
  pelo vértice mais próximo do ponto de chegada).
- Custo cidade-a-cidade entre grupos DIFERENTES = distância de Chebyshev entre
  os pontos, escalada por SCALE=10 e arredondada a inteiro (o parser Julia só
  aceita Int64; 10x preserva 1 casa decimal de precisão gometrica).
- Custo dentro do mesmo grupo = 0 (nunca é usado -- GTSP visita no máximo 1
  cidade por grupo; é a convenção observada nos arquivos .pcglns originais).
- Precedência Pi: contorno A precede contorno B sse A está geometricamente
  contido em B (via `Polygon.covers`). Isso é calculado do zero aqui, via
  geometria pura -- é a Pi REAL do problema, independente de a heurística
  aplicá-la ou não (o pipeline de carregamento NÃO preserva relação
  furo/pai em nenhum lugar, então essa detecção geométrica é a única fonte
  de Pi disponível no projeto).
- Grupo inicial (START_GROUP_SECTION) = grupo do polígono mais próximo da
  origem (0,0) -- mesmo critério de raiz da heurística
  (`escolher_raiz_nearest`), para máxima comparabilidade metodológica.

Uso:
    from converter import convert_instance
    stats = convert_instance("instances/generated/art_1_regular_sep.txt",
                              "saida.pcglns")
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial.distance import chebyshev
from shapely.geometry import Point, Polygon

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
from data.load_instances import load_instance  # noqa: E402
import heuristic_core as hc  # noqa: E402

SCALE = 10  # fator de escala p/ preservar precisao ao arredondar p/ Int64


def _detect_precedence(positions: List[Polygon]) -> List[Tuple[int, int]]:
    """
    Detecta pares (furo, pai) via contencao geometrica direta. Retorna lista
    de (idx_furo, idx_pai) -- idx_furo deve ser cortado ANTES de idx_pai.
    """
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


def _preprocess(layout_original: List[Polygon]) -> List[Polygon]:
    """Aplica o MESMO pre-processamento que run_heuristic() usa internamente,
    para que 'positions' seja identico ao que a heuristica de fato roteia."""
    layout = hc.tratar_furos(list(layout_original))
    layout = hc.resolver_intersecoes(layout)
    positions = hc.clusterizar_quadtree(layout)
    return positions


def _escolher_grupo_inicial(positions: List[Polygon], precedence_pairs: List[Tuple[int, int]]) -> int:
    """
    Mesmo criterio de raiz da heuristica: poligono com borda mais proxima da
    origem (0,0). Retorna indice 0-based do poligono/grupo.

    Ressalva descoberta empiricamente (teste em art_10_hole_sep): se o grupo
    inicial escolhido tambem participa de alguma restricao de precedencia
    (como furo OU como pai), o PCGLNS pode reportar uma "violacao" que na
    verdade e um artefato de linearizacao -- o tour do GTSP e um CICLO, e
    fixar o grupo inicial "corta" esse ciclo num ponto arbitrario; se esse
    ponto coincide com um grupo que tem restricao de ordem, a checagem linear
    (posicao no vetor impresso) fica ambigua perto do ponto de corte. A
    biblioteca original (ESC07.pcglns etc.) evita isso: nos exemplos
    conferidos, o grupo inicial nunca aparece em GTSP_SET_ORDERING. Seguimos
    a mesma convencao aqui: preferimos o poligono mais proximo da origem
    QUE NAO participe de nenhum par de precedencia; só caímos no mais
    próximo irrestrito se nenhum candidato elegível existir.
    """
    origem = Point(0, 0)
    envolvidos = {i for par in precedence_pairs for i in par}
    # Quando TODOS os poligonos participam de alguma restricao (comum em
    # instancias pequenas/densas, ex. dighe2 tem so 2 grupos e os 2
    # participam do unico par), usar o mais proximo IRRESTRITO pode escolher
    # um "suc" (que tem predecessor pendente). Como o grupo inicial fica fixo
    # na posicao 0 do ciclo linearizado, nenhum grupo pode precede-lo: se ele
    # proprio exige um predecessor, a instancia fica ESTRUTURALMENTE
    # infactivel para qualquer metodo que respeite essa linearizacao (VNS e a
    # checagem _viol_ciclica/_ordem_e_factivel_ciclica compartilhada). Por
    # isso, antes de cair no irrestrito, tenta-se o mais proximo entre os
    # grupos que NUNCA aparecem como sucessor (sao fontes no DAG de Pi, ou
    # nao participam dele) -- esses sempre podem legitimamente ser
    # visitados primeiro.
    sucessores = {suc for _, suc in precedence_pairs}
    dists = [(pol.exterior.distance(origem), idx) for idx, pol in enumerate(positions)]
    dists.sort()
    for _, idx in dists:
        if idx not in envolvidos:
            return idx
    for _, idx in dists:
        if idx not in sucessores:
            return idx
    # fallback final: nenhum candidato valido (nao deveria ocorrer com Pi
    # aciclico) -- usa o mais proximo mesmo assim.
    return dists[0][1]


def convert_instance(instance_path: str | Path, output_path: str | Path) -> Dict:
    """
    Converte uma instancia LCPP para .pcglns. Retorna um dict de estatisticas
    uteis para a comparacao posterior (nome, n_grupos, n_cidades, grupo
    inicial, pares de precedencia, e o termo constante de tempo-de-corte
    fixo -- Sum(L_i / v_cut) -- necessario para reconverter o custo do tour
    do PCGLNS de volta para a mesma unidade de FO da nossa heuristica).
    """
    name, layout_original = load_instance(instance_path)
    positions = _preprocess(layout_original)
    n_groups = len(positions)

    if n_groups == 0:
        raise ValueError(f"Instancia vazia apos pre-processamento: {name}")

    # Cidades: uma por vertice de cada poligono (exclui o vertice de fechamento duplicado)
    city_coords: List[Tuple[float, float]] = []
    city_group: List[int] = []  # grupo (0-based) de cada cidade
    group_cities: List[List[int]] = [[] for _ in range(n_groups)]

    for g_idx, pol in enumerate(positions):
        coords = list(pol.exterior.coords)[:-1]
        for (x, y) in coords:
            c_idx = len(city_coords)
            city_coords.append((x, y))
            city_group.append(g_idx)
            group_cities[g_idx].append(c_idx)

    n_cities = len(city_coords)

    # Termo fixo de tempo de corte: soma das distancias de Chebyshev entre
    # vertices consecutivos de cada poligono, dividido por v_cut -- IDENTICO
    # ao que calcular_fo() computa para os segmentos 'c' (corte), soh que
    # aqui calculado UMA VEZ (independente de ordem/tour), pois o valor eh
    # uma constante do layout (T_i^cut independente da ordem global).
    #
    # Arestas compartilhadas entre poligonos que se TOCAM (layouts "_pack")
    # sao contadas uma unica vez: calcular_fo() (via reclassificar_tags) so
    # conta uma parede compartilhada como corte 'c' UMA vez (a segunda
    # ocorrencia vira 'd', backtrack), entao somar o perimetro de cada
    # poligono independentemente superestimaria cut_time_fixed (ex.:
    # jakobs1_pack.txt: 24,54 vs 21,72). Usa-se a MESMA chave canonica de
    # aresta (min/max de coordenadas arredondadas) que
    # heuristic_core.reclassificar_tags usa internamente.
    def _seg_key(a, b):
        a = tuple(round(v, 6) for v in a)
        b = tuple(round(v, 6) for v in b)
        return (min(a, b), max(a, b))

    arestas_unicas = set()
    for pol in positions:
        coords = list(pol.exterior.coords)
        for i in range(len(coords) - 1):
            arestas_unicas.add(_seg_key(coords[i], coords[i + 1]))

    cut_time_fixed = sum(
        chebyshev(k[0], k[1]) / hc._V_CUT for k in arestas_unicas
    )

    # Matriz de distancias (cidade-cidade), escalada e arredondada a inteiro
    dist = np.zeros((n_cities, n_cities), dtype=np.int64)
    for i in range(n_cities):
        gi = city_group[i]
        for j in range(i + 1, n_cities):
            gj = city_group[j]
            if gi == gj:
                d = 0
            else:
                d = int(round(chebyshev(city_coords[i], city_coords[j]) * SCALE))
            dist[i, j] = d
            dist[j, i] = d

    # Precedencia real (furo -> pai), calculada via geometria pura
    precedence_pairs = _detect_precedence(positions)  # [(furo_idx, pai_idx), ...] 0-based (grupos)

    # Grupo inicial (mesma regra de raiz da heuristica)
    start_group = _escolher_grupo_inicial(positions, precedence_pairs)

    # ------------------------------------------------------------------
    # Escreve o arquivo .pcglns
    # ------------------------------------------------------------------
    lines = []
    lines.append(f"NAME : {Path(instance_path).stem}.pcglns")
    lines.append("TYPE : PCGLNS")
    lines.append("COMMENT : Convertido do LCPP (projeto EJOR) via pcgtsp_comparison/converter.py")
    lines.append(f"DIMENSION : {n_cities}")
    lines.append(f"GTSP_SETS : {n_groups}")
    lines.append("EDGE_WEIGHT_TYPE : EXPLICIT")
    lines.append("EDGE_WEIGHT_FORMAT : FULL_MATRIX")
    lines.append("EDGE_WEIGHT_SECTION")
    for i in range(n_cities):
        lines.append(" ".join(str(int(v)) for v in dist[i]))
    lines.append("GTSP_SET_SECTION")
    for g_idx in range(n_groups):
        cidades = " ".join(str(c + 1) for c in group_cities[g_idx])  # 1-indexed
        lines.append(f"{g_idx + 1} {cidades} -1")
    lines.append("GTSP_SET_ORDERING")
    for furo_idx, pai_idx in precedence_pairs:
        lines.append(f"{furo_idx + 1} {pai_idx + 1} -1")  # 1-indexed
    lines.append("START_GROUP_SECTION")
    lines.append(str(start_group + 1))  # 1-indexed
    lines.append("EOF")

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")

    return {
        "name": name,
        "n_groups": n_groups,
        "n_cities": n_cities,
        "start_group": start_group + 1,
        "n_precedence_pairs": len(precedence_pairs),
        "precedence_pairs": precedence_pairs,
        "cut_time_fixed": cut_time_fixed,
        "scale": SCALE,
    }


if __name__ == "__main__":
    import json
    inst = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "out.pcglns"
    stats = convert_instance(inst, out)
    print(json.dumps({k: v for k, v in stats.items() if k != "precedence_pairs"}, indent=2))
    print(f"Pares de precedencia: {stats['precedence_pairs']}")
    print(f"Arquivo escrito em: {out}")
