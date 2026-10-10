from scipy.spatial import KDTree
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.neighbors import KDTree
from scipy.spatial import KDTree
from scipy.spatial.distance import chebyshev

class QT:
    """
    QuadTree (Árvore Quaternária) utilizada para particionar o espaço 2D e organizar
    a distribuição das peças. Ela facilita a divisão do cenário em blocos menores
    antes do processo de clusterização, reduzindo a complexidade das buscas de proximidade.
    """

    def __init__(self, bounds, max_items=4):
        # bounds é uma tupla definindo a região: (xmin, ymin, xmax, ymax)
        self.bounds = bounds
        # Limite máximo de polígonos que uma única célula/quadrante pode armazenar
        self.max_items = max_items
        # Lista dos itens contidos neste quadrante (armazenados como (item, centroid))
        self.items = []
        # Filhos deste nó (None indica que ele é uma "folha", senão será uma lista de 4 sub-QTs)
        self.children = None

    def insert(self, item, centroid):
        """Insere um item (polígono) na QuadTree, subdividindo o quadrante atual se ficar cheio."""
        # Se a célula já foi subdividida, delega a inserção para o nó filho correspondente
        if self.children is not None:
            for child in self.children:
                if child.contains(centroid):
                    child.insert(item, centroid)
                    return

        # Se não foi subdividida, guarda o item neste nó
        self.items.append((item, centroid))

        # Se excedeu a capacidade do nó e ainda não tem filhos, divide a célula atual em 4
        if len(self.items) > self.max_items and self.children is None:
            self.subdivide()

    def contains(self, point):
        """Verifica de forma simples se uma coordenada (x, y) está dentro da área delimitada (bounds)."""
        xmin, ymin, xmax, ymax = self.bounds
        return xmin <= point[0] <= xmax and ymin <= point[1] <= ymax

    def subdivide(self):
        """
        Calcula o ponto central do quadrante atual e cria 4 novas células/filhos (sub-quadrantes):
        Superior-Esquerdo, Superior-Direito, Inferior-Esquerdo, Inferior-Direito.
        """
        xmin, ymin, xmax, ymax = self.bounds
        xmid, ymid = (xmin + xmax) / 2, (ymin + ymax) / 2

        # Instancia os 4 novos quadrantes definindo os limites matemáticos
        self.children = [
            QT((xmin, ymin, xmid, ymid), self.max_items),
            QT((xmid, ymin, xmax, ymid), self.max_items),
            QT((xmin, ymid, xmid, ymax), self.max_items),
            QT((xmid, ymid, xmax, ymax), self.max_items)
        ]

        # Redistribui os itens que estavam neste quadrante para os seus novos filhos recém-criados
        for item, centroid in self.items:
            for child in self.children:
                if child.contains(centroid):
                    child.insert(item, centroid)
                    break

        # Esvazia a lista do nó atual, visto que agora ele virou um galho e as folhas guardam os dados
        self.items = []

    def get_clusters(self):
        """Varre a árvore de forma recursiva e retorna a lista de peças organizadas por vizinhança espacial."""
        # Se chegou num nó folha, retorna os polígonos dentro dele
        if self.children is None:
            return [item[0] for item in self.items] if self.items else []

        clusters = []
        # Concatena a resposta de todos os filhos
        for child in self.children:
            clusters.extend(child.get_clusters())
        return clusters


def preprocess_layout_with_quadtree(positions, max_items=4):
    """
    Função auxiliar que cria uma Bounding Box englobando todas as peças fornecidas,
    inicia a estrutura QuadTree sobre essa área e injeta todos os polígonos nela.
    """
    xmin = min(p.bounds[0] for p in positions)
    ymin = min(p.bounds[1] for p in positions)
    xmax = max(p.bounds[2] for p in positions)
    ymax = max(p.bounds[3] for p in positions)

    # Inicia a raiz da QuadTree abrangendo o universo completo das posições
    quadtree = QT((xmin, ymin, xmax, ymax), max_items)

    # Insere iterativamente baseando-se no centroide de cada peça
    for polygon in positions:
        quadtree.insert(polygon, polygon.centroid.coords[0])

    return quadtree.get_clusters()


def optimize_cluster_connections_kdtree(positions, plot=True):
    """
    Constrói um mapa de conexões focando nos *Centroides* das peças.
    Utiliza uma KD-Tree (Árvore k-dimensional) para busca otimizada de vizinhos
    e em seguida roda um algoritmo de MST (Minimum Spanning Tree - Árvore Geradora Mínima)
    para criar um roteiro sem ciclos garantindo que os clusters estejam minimamente ligados.
    """
    # Extrai as coordenadas centrais (x, y) de cada polígono
    centroids = np.array([polygon.centroid.coords[0] for polygon in positions])

    # Inicia a KDTree. Essa estrutura mapeia os pontos no plano facilitando consultas super rápidas de "Qual é o ponto mais próximo?"
    kdtree = KDTree(centroids)

    G = nx.Graph()

    # Adiciona todos os polígonos como nós (bolinhas) num grafo lógico do NetworkX
    for i, polygon in enumerate(positions):
        G.add_node(i, pos=polygon.centroid.coords[0])

    # Para cada polígono, usa a KDTree para localizar a quem ele deve se conectar
    for i, polygon in enumerate(positions):
        # A busca k=2 retorna os dois pontos mais próximos. O índice 0 é sempre ele mesmo (distância zero),
        # por isso descartamos o primeiro e usamos o índice 1.
        _, nearest_idx = kdtree.query(polygon.centroid.coords[0], k=2)
        nearest_idx = nearest_idx[1]

        # Calcula a distância de Chebyshev entre o polígono atual e seu vizinho eleito
        dist = chebyshev(positions[i].centroid.coords[0], positions[nearest_idx].centroid.coords[0])
        # Traça uma aresta entre eles informando a distância como 'peso' (weight)
        G.add_edge(i, nearest_idx, weight=dist)

    # Cria a Árvore Geradora Mínima: Uma versão enxuta do grafo original onde todos estão
    # interconectados usando o caminho mais curto/leve possível.
    mst = nx.minimum_spanning_tree(G, weight="weight")

    if plot:
        # Lógica padrão do Matplotlib para plotar a visualização dos dados e da rede
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = plt.colormaps.get_cmap("tab10")

        for i, polygon in enumerate(positions):
            x, y = polygon.exterior.xy
            edgecolor = colors(i % 10)
            facecolor = colors(i % 10, alpha=0.3)

            ax.fill(x, y, edgecolor=edgecolor, facecolor=facecolor, lw=2)
            ax.text(polygon.centroid.x, polygon.centroid.y, str(i), color="black", fontsize=10, ha="center", va="center")

        # Desenhar as linhas vermelhas tracejadas que compõem a MST gerada
        for edge in mst.edges(data=True):
            i, j, data = edge
            p1 = positions[i].centroid.coords[0]
            p2 = positions[j].centroid.coords[0]
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r--', lw=2, label="Conexão MST" if i == 0 else "")

        ax.set_xlim(-10, 220)
        ax.set_ylim(-10, max(y) + 10 if positions else 50)
        ax.set_aspect('equal')
        ax.set_title(f"Clusters e Conexões Otimizadas (QuadTree + KDTree + MST)")
        plt.gca().invert_yaxis()
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.legend()
        plt.show()


def optimize_cluster_connections_kdtree_vertices_with_full_connectivity(positions, plot=True):
    """
    Uma versão muito mais robusta e complexa da função anterior. Em vez de conectar a partir dos CENTRÓIDES,
    este algoritmo foca nos limites da borda (VÉRTICES) dos polígonos.
    Isso é vital em contextos CNC (como laser ou faca) onde a ponte entre duas peças deve ocorrer
    a partir das bordas mais próximas de ambas para economizar deslocamento no mundo real.
    """
    # 1. Montagem do mapa total de vértices
    all_vertices = []
    polygon_index = []  # Um registro paralelo para sabermos a que polígono cada vértice (x, y) pertence

    for i, polygon in enumerate(positions):
        for vertex in polygon.exterior.coords:
            all_vertices.append(vertex)
            polygon_index.append(i)

    all_vertices = np.array(all_vertices)

    # 2. Inicializa a KD-Tree contendo centenas ou milhares de vértices
    kdtree = KDTree(all_vertices)

    G = nx.Graph()

    # 3. Cria a base de nós do Grafo
    for i, polygon in enumerate(positions):
        G.add_node(i, pos=polygon.centroid.coords[0], poligono=polygon)

    # 4. Busca a conexão borda-com-borda ideal
    for i, polygon in enumerate(positions):
        min_dist = float("inf")
        best_v1, best_v2 = None, None
        best_target_poly = None

        # Para cada quina/vértice (v1) do polígono atual
        for v1 in polygon.exterior.coords:
            # Traz os 5 vértices mais próximos deste v1 em todo o mapa
            distances, nearest_idxs = kdtree.query(v1, k=5)
            for nearest_idx in nearest_idxs:
                # Pula qualquer vértice que também faça parte do próprio polígono atual
                if polygon_index[nearest_idx] != i:
                    v2 = all_vertices[nearest_idx]
                    dist = chebyshev(v1, v2)

                    # Se achou uma nova distância recorde (a mais curta possível)
                    if dist < min_dist:
                        min_dist = dist
                        best_v1, best_v2 = v1, v2
                        best_target_poly = polygon_index[nearest_idx]
                    break  # Para a busca para este vértice 'v1' pois o primeiro 'forasteiro' achado será o mais próximo

        # Se encontrou uma conexão válida com o mundo externo, registra a aresta guardando as coordenadas
        # exatas da ponte (best_v1, best_v2)
        if best_v1 is not None and best_v2 is not None and best_target_poly is not None:
            G.add_edge(i, best_target_poly, weight=min_dist, connection=(best_v1, best_v2))


    # Agrupa quais polígonos já se tornaram ilhas conectadas (componentes conexos)
    connected_components = list(nx.connected_components(G))
    cluster_colors = {node: i for i, component in enumerate(connected_components) for node in component}

    # Passa o pente fino limpando ligações redundantes
    mst = nx.minimum_spanning_tree(G, weight="weight")

    # Verifica se após o MST a árvore ficou fragmentada (Várias "ilhas" de grafos sem conexão unificadora)
    components = list(nx.connected_components(mst))
    num_clusters = len(components)

    # 5. Modo de Força Bruta de Resgate: Se o mapa tiver mais de 1 ilha isolada
    if num_clusters > 1:
        potential_bridges = []
        # Analisa a ilha C1 contra a ilha C2 (Uma contra todas)
        for c1 in range(num_clusters):
            for c2 in range(c1 + 1, num_clusters):
                # Extrai cada polígono da ilha 1 contra o polígono da ilha 2
                for poly1 in components[c1]:
                    for poly2 in components[c2]:
                        # Testa as distâncias de cada quina e guarda uma lista com TODAS as pontes possíveis
                        for v1 in positions[poly1].exterior.coords:
                            for v2 in positions[poly2].exterior.coords:
                                dist = chebyshev(v1, v2)
                                potential_bridges.append((dist, poly1, poly2, v1, v2))

        # Ordena as pontes da menor (melhor) para a maior
        potential_bridges.sort()

        # Vai adicionando as melhores pontes uma a uma ao Grafo Principal
        for dist, i, j, v1, v2 in potential_bridges:
            G.add_edge(i, j, weight=dist, connection=(v1, v2))

            # Recalcula a MST
            mst = nx.minimum_spanning_tree(G, weight="weight")
            # Se agora todos estão conectados numa única árvore gigante, interrompe o processo para não gastar CPU atoa
            if nx.is_connected(mst):
                break

    # Reestrutura as paletas de cores baseadas na nova arquitetura unificada
    connected_components = list(nx.connected_components(mst))

    # 6. Plotagem da Interface final
    if plot:
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = plt.colormaps.get_cmap("tab10")

        for i, polygon in enumerate(positions):
            x, y = polygon.exterior.xy
            cluster_id = cluster_colors[i]
            edgecolor = colors(cluster_id % 10)
            facecolor = colors(cluster_id % 10, alpha=0.3)

            ax.fill(x, y, edgecolor=edgecolor, facecolor=facecolor, lw=2)
            ax.text(polygon.centroid.x, polygon.centroid.y, str(i), color="black", fontsize=10, ha="center", va="center")

        # Diferente da função anterior (onde a linha vermelha saía do meio da peça),
        # aqui o plot das linhas contínuas vermelhas ilustra a exata transição de uma quina
        # para a outra quina baseada na chave "connection".
        for edge in mst.edges(data=True):
            i, j, data = edge
            v1, v2 = data["connection"]
            plt.plot([v1[0], v2[0]], [v1[1], v2[1]], 'r-', lw=2, label="Conexão MST" if i == 0 else "")

        ax.set_aspect('equal')
        plt.gca().invert_yaxis()
        plt.grid(True, linestyle="--", linewidth=0.5)

    # Retorna a Árvore de Otimização Lógica Definitiva (MST Baseada nas Quinas)
    return mst
