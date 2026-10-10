"""
main.py
=======
Executa a heurística base (QuadTree + KD-tree + MST + travessia, sem busca
local) sobre todas as instâncias de instances/generated e instances/original.

Para cada instância grava, em results/heuristic/:
  paths/<instância>   relatório com a FO e o caminho (segmento, tag 'c'/'d')
  plots/              figura do caminho com os índices das etapas
  plots_clean/        figura do caminho sem os índices

Uso: python src/main.py
"""
import sys
import os
from pathlib import Path
from datetime import datetime

# Raiz do projeto (um nível acima de src/)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))  # garante que data/load_instances resolve mesmo se importado de fora

from os import makedirs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial.distance import chebyshev
from shapely import Polygon, Point

from geometry import segment_intersections
from timer import timeit
from graph_routing import construir_arvore_direcionada, plotar_poligonos_e_arvore, cortar_arvore, distancia_e_aresta_ate_raiz, linha_esta_na_lista
from quadtree import preprocess_layout_with_quadtree, \
    optimize_cluster_connections_kdtree_vertices_with_full_connectivity
from data.load_instances import load_all_instances


# Diretórios de saída relativos à raiz do projeto
_OUT_PATHS      = _PROJECT_ROOT / 'results' / 'heuristic' / 'paths'
_OUT_PLOTS      = _PROJECT_ROOT / 'results' / 'heuristic' / 'plots'
_OUT_PLOTS_CLEAN = _PROJECT_ROOT / 'results' / 'heuristic' / 'plots_clean'

makedirs(_OUT_PATHS,       exist_ok=True)
makedirs(_OUT_PLOTS,       exist_ok=True)
makedirs(_OUT_PLOTS_CLEAN, exist_ok=True)

# ==========================================
# LOOP PRINCIPAL DE PROCESSAMENTO
# ==========================================

# Carrega todas as 282 instâncias diretamente da pasta instances/
_all_instances = load_all_instances()
_n_total       = len(_all_instances)
_session_start = datetime.now()

print("=" * 68)
print(f"  Heurística  |  {_n_total} instâncias  |  início: {_session_start:%H:%M:%S}")
print("=" * 68)

for _inst_idx, instance in enumerate(_all_instances, start=1):
    _inst_start = datetime.now()
    print(f"\n[{_inst_idx:>3}/{_n_total}] {instance[0]}", flush=True)

    p_out         = str(_OUT_PATHS)
    p_plot        = str(_OUT_PLOTS)
    p_plot_result = str(_OUT_PLOTS_CLEAN)

    # fo_total acumula a FO de todos os componentes da instância
    fo_total = 0.0

    # Abre (ou cria) um arquivo de texto para registrar o relatório da instância atual
    with open(f"{p_out}/{instance[0]}", 'w+', encoding='utf-8') as f:
        print(f"Processando arquivo: {instance[0]}", file=f)
        print(file=f)

        # Inicia a contagem de tempo de execução usando o gerenciador de contexto
        with timeit(file_write=f):
            layout = instance[1]

            # 2. Tratamento de Furos (Buracos nos Polígonos)
            # Se um polígono tem um anel interior (furo), ele é extraído e
            # adicionado ao layout como um polígono independente para que a máquina também o corte.
            for p in list(layout):
                if len(p.interiors) > 0:
                    for i in p.interiors:
                        layout.append(Polygon(i))

            # 3. Tratamento de Interseções e Sobreposições
            intersecoes = set()
            # Varre todas as combinações de peças buscando colisões
            for i, f1 in enumerate(layout):
                for j, f2 in enumerate(layout):
                    if i != j:
                        if f1.intersects(f2):
                            # Salva a dupla colidente ordenada para não processar duas vezes
                            intersecoes.add(tuple(sorted((i, j))))

            # Processa as interseções fatiando os segmentos compartilhados
            for i,j in list(intersecoes):
                part = segment_intersections(layout[i], layout[j])
                layout[i], layout[j] = [Polygon(k) for k in part]

            # 4. Clusterização e Otimização da Rede (Grafos)
            # Organiza o espaço usando a QuadTree
            clustered_positions = preprocess_layout_with_quadtree(layout)

            # Gera a Árvore Geradora Mínima (MST) conectando todos os vértices de forma ótima
            G = optimize_cluster_connections_kdtree_vertices_with_full_connectivity(clustered_positions, plot=False)

            # 5. Geração das Rotas de Corte
            for componente in nx.connected_components(G):
                subgrafo = G.subgraph(componente)

                # Descobre qual polígono está mais perto da origem (0,0) da máquina
                dists = [(i, distancia_e_aresta_ate_raiz(Point(0,0), clustered_positions[i])) for i in subgrafo.nodes]

                # Elege esse polígono como a 'Raiz' da árvore de corte
                raiz = min([(i, d[0][0].length if len(d[0]) > 0 else 0) for i, d in dists], key=lambda x: x[1])[0]

                # Constrói a árvore direcionada a partir da raiz eleita
                arvore = construir_arvore_direcionada(subgrafo, raiz)

                # Gera a sequência bruta de movimentos (o caminho do laser/faca)
                caminho = cortar_arvore(arvore, raiz)

                # 6. Pós-processamento do Caminho (Classificação Corte vs Deslocamento em Vazio)
                # O caminho retorna passos com a tag 'x'. Aqui é verificado se a linha já foi percorrida antes.
                # Se NÃO foi percorrida, tag 'c' (corte). Se JÁ foi, tag 'd' (deslocamento vazio / laser desligado).
                caminho = [(j[0], j[1] if j[1] != 'x' else ('c' if not linha_esta_na_lista(j[0], caminho[:i]) else 'd')) for i, j in enumerate(caminho)]

                # 7. Cálculo da Função Objetivo (FO)
                # Soma os tempos de cada trecho. Tempo = Distância de Chebyshev / Velocidade.
                # Velocidade de corte ('c') = 100/6. Velocidade em vazio = 400.
                fo = sum([chebyshev(i[0].coords[0], i[0].coords[1]) / (100/6 if i[1] == 'c' else 400) for i in caminho])
                fo_total += fo

        # ==========================================
        # SAÍDA DE DADOS E GERAÇÃO DE GRÁFICOS
        # ==========================================
        print(file=f)
        print("Função Objetivo:", fo_total, file=f)
        print(file=f)

        # Salva o arquivo de log com as coordenadas de cada linha do caminho e seu estado (c ou d)
        [print(*i[0].coords[0], *i[0].coords[1], i[1], file=f) for i in caminho]

        # Plota e salva a imagem do caminho detalhado com os índices das etapas
        plotar_poligonos_e_arvore(arvore, caminho=caminho, raiz=raiz, hierarquia=False)
        plt.savefig(f"{p_plot}/{instance[0].split('.')[0]}.png")
        plt.close()

        # Plota e salva a imagem do caminho "limpo", sem a poluição visual dos índices (números)
        plotar_poligonos_e_arvore(arvore, caminho=caminho, raiz=raiz, hierarquia=False, indices=False)
        plt.savefig(f"{p_plot_result}/{instance[0].split('.')[0]}.png")
        plt.close()

    _elapsed = datetime.now() - _inst_start
    _done_s  = (datetime.now() - _session_start).total_seconds()
    _avg_s   = _done_s / _inst_idx
    _rem     = _avg_s * (_n_total - _inst_idx)
    print(
        f"         FO: {fo_total:.4f}  "
        f"tempo: {_elapsed.total_seconds():.1f}s  "
        f"ETA: ~{int(_rem // 60)}m{int(_rem % 60)}s",
        flush=True
    )

_session_elapsed = datetime.now() - _session_start
print("\n" + "=" * 68)
print(f"  Concluído: {_n_total} instâncias em {_session_elapsed}")
print("=" * 68)
