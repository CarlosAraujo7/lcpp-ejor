"""
convert_zhang_pack.py
======================
Converte instancias .pack do capsule Code Ocean do Zhang et al. 2024
("Heuristic approaches for the cutting path problem") para o formato de
segmentos usado neste projeto (mesmo formato de instances/original/*.txt):

    linha 1: N (numero total de segmentos)
    linhas 2..N+1: x1 y1 x2 y2  (endpoints de cada segmento)

Formato .pack (uma peca por bloco):
    nome
    n_pecas
    largura_folha
    altura_folha
    param_escala
    <bloco por peca, repetido n_pecas vezes>:
        idx_peca flag1
        n_vertices flag2
        x0 x1 x2 ... (n_vertices valores)
        y0 y1 y2 ... (n_vertices valores)
    <metadados finais, ignorados>

Cada peca e um poligono fechado unico (sem furos, confirmado por inspecao
manual dos 6 arquivos disponiveis). Cada peca vira sua propria cadeia de
segmentos fechada (convencao "separated": pecas tratadas como entidades
distintas mesmo se as coordenadas colidem, exatamente como no modelo NRM/
GTSP do proprio Zhang2024 -- "corner points shared by different pieces...
will be regarded as different vertices").
"""
from __future__ import annotations
import sys
from pathlib import Path


def parse_pack(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        lines = [l.strip() for l in fh if l.strip()]

    name = lines[0]
    n_pieces = int(lines[1])
    # lines[2], lines[3], lines[4] = largura, altura, param -- nao usados

    cursor = 5
    pieces = []
    for _ in range(n_pieces):
        header1 = lines[cursor].split(); cursor += 1
        header2 = lines[cursor].split(); cursor += 1
        n_verts = int(header2[0])
        xs = list(map(float, lines[cursor].split())); cursor += 1
        ys = list(map(float, lines[cursor].split())); cursor += 1
        if len(xs) != n_verts or len(ys) != n_verts:
            raise ValueError(
                f"{path.name}: peca {header1[0]} esperava {n_verts} vertices, "
                f"leu {len(xs)}x/{len(ys)}y"
            )
        pieces.append(list(zip(xs, ys)))

    return name, pieces


def pieces_to_segments(pieces):
    segs = []
    for verts in pieces:
        k = len(verts)
        for i in range(k):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % k]
            segs.append((x1, y1, x2, y2))
    return segs


def convert(pack_path: Path, out_path: Path):
    name, pieces = parse_pack(pack_path)
    segs = pieces_to_segments(pieces)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(f"{len(segs)}\n")
        for x1, y1, x2, y2 in segs:
            fh.write(f"{x1} {y1} {x2} {y2}\n")
    n_verts_total = sum(len(p) for p in pieces)
    return name, len(pieces), n_verts_total, len(segs)


if __name__ == "__main__":
    src_dir = Path(sys.argv[1])
    dst_dir = Path(sys.argv[2])
    dst_dir.mkdir(parents=True, exist_ok=True)

    # |P|, |V| esperados, extraidos manualmente das Tabelas 3/6 do paper
    # Zhang et al. 2024, para validacao cruzada.
    esperado = {
        "blaz2_s":  (4, 30),
        "dighe1":   (16, 62),
        "dighe1_s": (4, 16),
        "dighe2":   (10, 47),
        "fu":       (12, 43),
        "poly1a":   (15, 69),
    }

    print(f"{'instancia':<12} {'|P|':>5} {'esperado':>10} {'|V|':>5} {'esperado':>10} {'segs':>6}  status")
    for pack_file in sorted(src_dir.glob("*.pack")):
        stem = pack_file.stem
        out_path = dst_dir / f"{stem}.txt"
        name, n_p, n_v, n_segs = convert(pack_file, out_path)
        exp_p, exp_v = esperado.get(stem, (None, None))
        ok = (n_p == exp_p and n_v == exp_v)
        status = "OK" if ok else "DIVERGIU"
        print(f"{stem:<12} {n_p:>5} {exp_p!s:>10} {n_v:>5} {exp_v!s:>10} {n_segs:>6}  {status}")
