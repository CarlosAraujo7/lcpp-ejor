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
        int_lst = [round(x) for x in float_lst]
        tmplst = []
        for vert_idx, fl in enumerate(int_lst):
            if fl == -1:
                tmplst.append(vert_idx + 1)
        
        orderings.append(tmplst)
        lines[i] = ' '.join(str(x) for x in int_lst)
    
    return "\n".join(lines), orderings


def add_sets_ordering_section(lines, orderings, isSop):
    sets = []
    dims = -1
    if not isSop:
        idx = get_line_contains_idx("GTSP_SETS", lines)
        if idx == -1:
            return lines
        sets_num = int(lines[idx].split(" : ")[1])
        
        idx = get_line_contains_idx("GTSP_SET_SECTION", lines)
        if idx == -1:
            return lines
        
        for set_line_idx in range(idx + 1, idx + sets_num + 1):
            set_vals = []
            splitted = lines[set_line_idx].split(" ")[:-1]
            for set_val in splitted[1:]:
                set_vals.append(int(set_val))
            sets.append(set_vals)
        # print(sets)
        # print(orderings)
    else:
        dims_idx = get_line_contains_idx("DIMENSION", lines)
        if dims_idx == -1:
            return lines
        dims = int(lines[dims_idx].split(" : ")[1])
        for i in range(dims):
            set_vals = []
            set_vals.append(i + 1)
            sets.append(set_vals)

    # inverted_orderings = [None] * sets_num
    inverted_orderings = []
    for ord_idx, ordering in enumerate(orderings):
        if ordering:
            src_idx = -1
            for inner_idx, set_to_precede in enumerate(sets):
                if ord_idx + 1 in set_to_precede:
                    src_idx = inner_idx + 1
                    break
            
            if src_idx == -1:
                print("Failed to find src set")
                break
            
            # inverted_orderings
            ordering_tmp = []
            set_to_insert = set()
            for set_from in ordering:
                dst_idx = -1
                for inner_idx, set_to_precede in enumerate(sets):
                    if set_from in set_to_precede:
                        dst_idx = inner_idx
                        break
                if dst_idx == -1:
                    print("Failed to find dst set")
                    break
                set_to_insert.add(dst_idx + 1)
            
            ordering_tmp.append(src_idx)
            ordering_tmp.append(list(set_to_insert))
            if ordering_tmp not in inverted_orderings:
                inverted_orderings.append(ordering_tmp)
    
    # print(sets, "\n")
    # print(orderings, "\n")
    # print(inverted_orderings, "\n")

    final_ordering = []
    for ord_idx, ordering in enumerate(inverted_orderings):
        if not ordering:
            continue
        
        for set_idx in ordering[1]:
            sets_set = set()
            for inner_ordering in inverted_orderings:
                if set_idx in inner_ordering[1]:
                    sets_set.add(inner_ordering[0])
            
            new_lst = [set_idx, list(sets_set)]
            if new_lst not in final_ordering:
                final_ordering.append(new_lst)
    
    pre_str = "START_GROUP_SECTION"

    idx = get_line_contains_idx(pre_str, lines)
    if idx == -1:
        return lines
    
    if isSop:
        dims_idx = get_line_contains_idx("DIMENSION", lines)
        if dims_idx == -1:
            return lines
        lines.insert(dims_idx + 1, "GTSP_SETS : " + str(dims))
        idx = idx + 1
        lines.insert(idx, "GTSP_SET_SECTION")
        idx = idx + 1
        for id, curr_set in enumerate(sets):
            lines.insert(idx, str(id + 1) + " " +  str(id + 1) + " -1")
            idx = idx + 1
    
    lines.insert(idx, "GTSP_SET_ORDERING")
    next_line_idx = 1
    for ordering in final_ordering:
        if ordering:
            ordering_str = str(ordering[0])
            for set_idx in ordering[1]:
                ordering_str = ordering_str + str(" " + str(set_idx))
            ordering_str = ordering_str + str(" -1")
            # print(idx, " ", next_line_idx)
            lines.insert(idx + next_line_idx, ordering_str)
            next_line_idx = next_line_idx + 1

    return lines


def remove_pc_specific_param(param_name, text_lines):
    idx = get_line_contains_idx(param_name, text_lines)
    if idx != -1:
        del text_lines[idx + 1]
        del text_lines[idx]

    return text_lines


def remove_params(text, isSop):
    lines = text.split("\n")

    if not isSop:
        lines = remove_pc_specific_param("NODE_WEIGHT_SECTION", lines)
        lines = remove_pc_specific_param("NODE_AGENT_SECTION", lines)
    
    return "\n".join(lines)


def rename_params(text, isSop):
    if not isSop:
        text = text.replace("GROUPS :", "GTSP_SETS :")
        text = text.replace("NODE_GROUP_SECTION", "GTSP_SET_SECTION")
    
    return text


def set_params(text, isSop):
    text, orderings = convert_wight_section(text, isSop)
    lines = text.split("\n")
    idx = get_line_contains_idx("TYPE", lines)
    if idx == -1:
        return
    
    lines[idx] = lines[idx].split(" : ")[0] + " : PCGLNS"

    idx = get_line_contains_idx("NAME", lines)
    if idx == -1:
        return

    paramName = lines[idx].split(" : ")[0]
    fullName = lines[idx].split(" : ")[1]
    lines[idx] = paramName + " : " + fullName.split(".")[0] + ".pcglns"

    if isSop:
        eof_idx = get_line_contains_idx("EOF", lines)
        if eof_idx == -1:
            return "\n".join(lines)
        
        lines.insert(eof_idx, "START_GROUP_SECTION")
        lines.insert(eof_idx + 1, "1")

    lines = add_sets_ordering_section(lines, orderings, isSop)

    return "\n".join(lines)


def convert_text(origin, isSop):
    converted = origin
    converted = converted.replace(":", " :")
    converted = converted.replace("  ", " ")

    converted = remove_params(converted, isSop)
    converted = rename_params(converted, isSop)
    converted = set_params(converted, isSop)

    return converted


def convert_file(input_dir, filename, output_dir):
    if not filename.endswith(".pcgtsp") and not filename.endswith(".sop"):
        return
    
    isSop = filename.endswith(".sop")

    inst_file = open(input_dir + filename, "r")
    text = inst_file.read()
    converted = convert_text(text, isSop)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    gtsp_file_name = os.path.splitext(filename)[0] + ".pcglns"
    gtsp_file = open(output_dir + gtsp_file_name, "w+")
    gtsp_file.write(converted)
    gtsp_file.close()


def convert_dir(input, output_dir):
    for filename in os.listdir(input):
        print("Processing " + filename + "...")
        convert_file(input, filename, output_dir)


if __name__ == "__main__":
    argc = len(sys.argv)
    if argc < 2 or argc > 3:
        print(f"Wrong arguments number")
        exit(0)
    
    input = sys.argv[1]
    is_dir = os.path.isdir(input)
    if is_dir and not input.endswith('/'):
        input = input + '/'
    
    output_dir = ""
    if argc == 3:
        output_dir = sys.argv[2]
        if not output_dir.endswith('/'):
            output_dir = output_dir + "/"
    
    if is_dir:
        convert_dir(input, output_dir)
    else:
        convert_file("", input, output_dir)