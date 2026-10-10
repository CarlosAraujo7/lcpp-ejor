"""
load_instances.py
=================
Carrega todas as instâncias do problema de corte a laser a partir dos
arquivos .txt nas pastas instances/generated e instances/original.

Formato dos arquivos:
  Linha 1: N  (número total de segmentos de linha)
  Linhas 2..N+1: x1 y1 x2 y2  (endpoints de cada segmento)

Os segmentos são agrupados em polígonos pela conectividade de pontos:
o ponto final de um segmento deve coincidir (dentro de tolerância)
com o ponto inicial do próximo segmento do mesmo polígono.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from shapely.geometry import Polygon, LinearRing

# Raiz do projeto (src/data/ -> 2 niveis acima = raiz do projeto)
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[2]

_INST_GERADAS   = _PROJECT_ROOT / "instances" / "generated"
_INST_ORIGINAIS = _PROJECT_ROOT / "instances" / "original"

# Tolerância para identificar pontos "iguais" (fechamento de polígono)
_TOL = 1e-6


# ---------------------------------------------------------------------------
# Parser de arquivo
# ---------------------------------------------------------------------------

def _parse_segments(filepath: str | Path) -> List[Tuple[float, float, float, float]]:
    """Lê um arquivo e retorna lista de (x1, y1, x2, y2)."""
    segs: List[Tuple[float, float, float, float]] = []
    with open(filepath, "r", encoding="utf-8") as fh:
        lines = [l.strip() for l in fh if l.strip()]
    if not lines:
        return segs
    # primeira linha: número de segmentos (pode ou não estar presente)
    try:
        n = int(lines[0])
        data_lines = lines[1:]
    except ValueError:
        data_lines = lines

    for line in data_lines:
        parts = line.split()
        if len(parts) >= 4:
            x1, y1, x2, y2 = map(float, parts[:4])
            segs.append((x1, y1, x2, y2))
    return segs


def _segments_to_polygons(segs: List[Tuple[float, float, float, float]]) -> List[Polygon]:
    """
    Reconstrói polígonos a partir de segmentos conectados.
    Usa union-find / cadeia de adjacência para agrupar arestas que
    compartilham endpoints (dentro de tolerância _TOL).
    """
    if not segs:
        return []

    # Representação: para cada segmento guardar (p_inicio, p_fim)
    pts_start = [(s[0], s[1]) for s in segs]
    pts_end   = [(s[2], s[3]) for s in segs]

    used = [False] * len(segs)
    polygons: List[Polygon] = []

    def near(a, b):
        return abs(a[0] - b[0]) < _TOL and abs(a[1] - b[1]) < _TOL

    def find_next(current_end, exclude_idx):
        """Encontra segmento não usado cujo início está perto de current_end."""
        for k in range(len(segs)):
            if not used[k] and k != exclude_idx:
                if near(pts_start[k], current_end):
                    return k
                # tenta sentido inverso
                if near(pts_end[k], current_end):
                    return -(k + 1)  # índice negativo sinaliza inversão
        return None

    for i in range(len(segs)):
        if used[i]:
            continue
        # Inicia nova cadeia a partir do segmento i
        chain = [(pts_start[i], pts_end[i])]
        used[i] = True
        current_end = pts_end[i]
        start_pt    = pts_start[i]

        # Estende a cadeia enquanto houver continuidade
        while True:
            nxt = find_next(current_end, -1)
            if nxt is None:
                break
            if nxt >= 0:
                k = nxt
                if used[k]:
                    break
                chain.append((pts_start[k], pts_end[k]))
                used[k] = True
                current_end = pts_end[k]
            else:
                k = -(nxt + 1)
                if used[k]:
                    break
                chain.append((pts_end[k], pts_start[k]))
                used[k] = True
                current_end = pts_start[k]

            # Fechou o polígono?
            if near(current_end, start_pt):
                break

        # Extrai coordenadas únicas
        coords = [c[0] for c in chain]
        if near(current_end, start_pt):
            coords.append(start_pt)   # fecha explicitamente
        if len(coords) >= 3:
            try:
                poly = Polygon(coords)
                if poly.is_valid and not poly.is_empty and poly.area > _TOL:
                    polygons.append(poly)
                elif not poly.is_valid:
                    poly = poly.buffer(0)
                    if not poly.is_empty and poly.area > _TOL:
                        # buffer(0) pode retornar MultiPolygon
                        from shapely.geometry import MultiPolygon
                        if isinstance(poly, MultiPolygon):
                            polygons.extend(list(poly.geoms))
                        else:
                            polygons.append(poly)
            except Exception:
                pass

    # Garante que tudo na lista é Polygon simples (explode MultiPolygons)
    from shapely.geometry import MultiPolygon
    flat = []
    for p in polygons:
        if isinstance(p, MultiPolygon):
            flat.extend([g for g in p.geoms if g.area > _TOL])
        elif p.area > _TOL:
            flat.append(p)
    return flat


# ---------------------------------------------------------------------------
# Carregamento de instâncias
# ---------------------------------------------------------------------------

def load_instance(filepath: str | Path) -> Tuple[str, List[Polygon]]:
    """
    Carrega uma instância e retorna (nome, lista_de_polygons).
    """
    fp = Path(filepath)
    segs = _parse_segments(fp)
    polys = _segments_to_polygons(segs)
    return fp.name, polys


def load_all_instances(
    include_generated: bool = True,
    include_original: bool = True,
) -> List[Tuple[str, List[Polygon]]]:
    """
    Carrega todas as instâncias disponíveis e retorna lista de
    (nome_arquivo, lista_de_polygons).

    Parâmetros
    ----------
    include_generated : incluir instâncias da pasta 'instancias geradas'
    include_original  : incluir instâncias das pastas 'instancias originais'
    """
    instances: List[Tuple[str, List[Polygon]]] = []

    if include_generated and _INST_GERADAS.exists():
        for fp in sorted(_INST_GERADAS.glob("*.txt")):
            name, polys = load_instance(fp)
            if polys:
                instances.append((name, polys))

    if include_original and _INST_ORIGINAIS.exists():
        for subdir in sorted(_INST_ORIGINAIS.iterdir()):
            if subdir.is_dir():
                for fp in sorted(subdir.glob("*.txt")):
                    # prefixo para evitar colisões de nome
                    name, polys = load_instance(fp)
                    tagged = f"{subdir.name}__{name}"
                    if polys:
                        instances.append((tagged, polys))

    return instances


# ---------------------------------------------------------------------------
# Utilidade: informações rápidas sobre as instâncias
# ---------------------------------------------------------------------------

def instance_stats(instances):
    """Imprime estatísticas básicas."""
    print(f"Total de instâncias: {len(instances)}")
    poly_counts = [len(p) for _, p in instances]
    print(f"Polígonos por instância: min={min(poly_counts)}, "
          f"max={max(poly_counts)}, média={sum(poly_counts)/len(poly_counts):.1f}")


if __name__ == "__main__":
    insts = load_all_instances()
    instance_stats(insts)
