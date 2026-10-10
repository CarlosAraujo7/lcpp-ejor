import sys
import os
import re
from pathlib import Path


def get_line_contains_idx(substr, lines):
    idx = -1
    for line in lines:
        if line.startswith(substr):
            idx = lines.index(line)
    
    return idx


def convert_wight_section(text, isSop):
    orderings = []
    lines = text.split("\n")
    idx = get_line_contains_idx("EDGE_WEIGHT_SECTION", lines)
    if isSop:
        del lines[idx + 1]
    
    dims_idx = get_line_contains_idx("DIMENSION", lines)
    sets_idx = get_line_contains_idx("GTSP_SETS", lines)

    if idx == -1 or dims_idx == -1:
        return text, orderings

    dims = int(lines[dims_idx].split(" : ")[1])
    for i in range(idx + 1, idx + dims + 1):
        strip_str = lines[i].strip()
        strip_str = re.sub('\s+', ' ', strip_str)
        float_lst = list(map(float, strip_str.split(' ')))
        tmplst = []
        for vert_idx, fl in enumerate(float_lst):
            if fl == -1:
                tmplst.append(vert_idx + 1)
        
        orderings.append(tmplst)
        lines[i] = strip_str
    
    return "\n".join(lines), orderings


def read_mat(text):
    lines = text.split("\n")
    idx = get_line_contains_idx("EDGE_WEIGHT_SECTION", lines)
    dims_idx = get_line_contains_idx("DIMENSION", lines)\

    if idx == -1 or dims_idx == -1:
        return []
    
    dist_mat = []
    dims = int(lines[dims_idx].split(" : ")[1])
    for i in range(idx + 1, idx + dims + 1):
        strip_str = lines[i].strip()
        strip_str = re.sub('\s+', ' ', strip_str)
        float_lst = list(map(float, strip_str.split(' ')))
        dist_arr = []
        for vert_idx, fl in enumerate(float_lst):
            dist_arr.append(round(fl))
        dist_mat.append(dist_arr)
    
    return dist_mat


def check_tour(filename, tour):
    if not filename.endswith(".pcglns"):
        print("Wrong file format")
        return
    
    inst_file = open(filename, "r")
    text = inst_file.read()
    inst_file.close()
    dist_mat = read_mat(text)
    sum = 0
    for i in range(len(tour) - 1):
        # print(str(dist_mat[tour[i] - 1][tour[i + 1] - 1]) + "\n")
        if (dist_mat[tour[i] - 1][tour[i + 1] - 1] == -1):
            print("Found -1 dist")
            exit(0)
        
        sum += dist_mat[tour[i] - 1][tour[i + 1] - 1]

    if (dist_mat[tour[len(tour) - 1] - 1][tour[0] - 1] != -1):
        sum += dist_mat[tour[len(tour) - 1] - 1][tour[0] - 1]

    return sum



if __name__ == "__main__":
    argc = len(sys.argv)
    print(str(argc))
    if argc != 3:
        print("Wrong arguments number")
        exit(0)
    
    input_file = sys.argv[1]
    tour = eval(sys.argv[2])

    sum = check_tour(input_file, tour)
    print("Length:", sum, "tour:", tour)