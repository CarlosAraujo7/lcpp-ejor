# Copyright 2017 Stephen L. Smith and Frank Imeson
# Modified copyright 2020 A. Kudryavtsev and M. Khachay
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Sequentially moves each vertex to its best point on the tour.
Repeats until no more moves can be found
"""
function moveopt!(
    tour::Array{Int64, 1},
    dist::Array{Int64, 2},
    sets::Array{Any, 1},
    order_constraints::Array{Constraints, 1},
    member::Array{Int64, 1},
    setdist::Distsv,
    start_set::Int64,
)
    improvement_found = true
    number_of_moves = 0
    start_position = 2

    while improvement_found && number_of_moves < 10
        improvement_found = false
        for i in start_position:length(tour)
            select_vertex = tour[i]
            delete_cost = removal_cost(tour, dist, i)
            set_ind = member[select_vertex]
            splice!(tour, i)  # remove vertex from tour

            min_insert_idx, max_insert_idx =
                calc_bounds(tour, set_ind, order_constraints, member)

            if set_ind != start_set
                min_insert_idx = max(min_insert_idx, 2)
                if max_insert_idx < min_insert_idx
                    max_insert_idx = min_insert_idx
                end
            else
                println("inserting start set!")
            end

            # find the best place to insert the vertex
            v, pos, cost = insert_cost_lb(
                tour,
                dist,
                sets[set_ind],
                set_ind,
                setdist,
                min_insert_idx,
                max_insert_idx,
                select_vertex,
                i,
                delete_cost,
            )

            insert!(tour, pos, v)
            # check if we found a better position for vertex i
            if cost < delete_cost
                improvement_found = true
                number_of_moves += 1
                start_position = min(pos, i) # start looking for swaps where tour change began
                break
            end
        end
    end
end


function moveopt_rand!(
    RNGs::Array{MersenneTwister, 1},
    tour::Array{Int64, 1},
    dist::Array{Int64, 2},
    sets::Array{Any, 1},
    order_constraints::Array{Constraints, 1},
    member::Array{Int64, 1},
    iters::Int,
    setdist::Distsv,
    start_set::Int64,
)
    tour_inds = collect(2:length(tour))
    for i in 1:iters # i = rand(1:length(tour), iters)
        i = incremental_shuffle!(RNGs, tour_inds, i)
        select_vertex = tour[i]

        delete_cost = removal_cost(tour, dist, i)
        set_ind = member[select_vertex]
        splice!(tour, i)  # remove vertex from tour

        min_insert_idx, max_insert_idx =
            calc_bounds(tour, set_ind, order_constraints, member)

        if set_ind != start_set
            min_insert_idx = max(min_insert_idx, 2)
            if max_insert_idx < min_insert_idx
                max_insert_idx = min_insert_idx
            end
        else
            println("inserting start set!")
        end

        v, pos, cost = insert_cost_lb(
            tour,
            dist,
            sets[set_ind],
            set_ind,
            setdist,
            min_insert_idx,
            max_insert_idx,
            select_vertex,
            i,
            delete_cost,
        )
        insert!(tour, pos, v)
    end
end


"""
compute the cost of inserting vertex v into position i of tour
"""
@inline function insert_cost_lb(
    tour::Array{Int64, 1},
    dist::Array{Int64, 2},
    set::Array{Int64, 1},
    setind::Int,
    setdist::Distsv,
    min_insert_idx::Int,
    max_insert_idx::Int,
    bestv::Int,
    bestpos::Int,
    best_cost::Int,
)
    if min_insert_idx == max_insert_idx && max_insert_idx == length(tour) + 1
        v1 = tour[length(tour)]
        for v in set
            insert_cost = dist[v1, v]
            if insert_cost < best_cost
                best_cost = insert_cost
                bestv = v
            end
        end
        bestpos = max_insert_idx
        return bestv, bestpos, best_cost
    end

    for i in min_insert_idx:max_insert_idx
        v1 = prev_tour(tour, i) # first check lower bound
        lb =
            setdist.vert_set[v1, setind] + setdist.set_vert[setind, tour[i]] -
            dist[v1, tour[i]]
        lb > best_cost && continue

        for v in set
            insert_cost = dist[v1, v] + dist[v, tour[i]] - dist[v1, tour[i]]
            if insert_cost < best_cost
                best_cost = insert_cost
                bestv = v
                bestpos = i
            end
        end
    end
    return bestv, bestpos, best_cost
end


"""
determine the cost of removing the vertex at position i in the tour
"""
@inline function removal_cost(tour::Array{Int64, 1}, dist::Array{Int64, 2}, i::Int64)
    if i == 1
        return dist[tour[end], tour[i]] + dist[tour[i], tour[i + 1]] -
               dist[tour[end], tour[i + 1]]
    elseif i == length(tour)
        return dist[tour[i - 1], tour[i]] + dist[tour[i], tour[1]] -
               dist[tour[i - 1], tour[1]]
    else
        return dist[tour[i - 1], tour[i]] + dist[tour[i], tour[i + 1]] -
               dist[tour[i - 1], tour[i + 1]]
    end
end


""" repeatedly perform moveopt and reopt_tour until there is no improvement """
function opt_cycle!(
    RNGs::Array{MersenneTwister, 1},
    current::Tour,
    dist::Array{Int64, 2},
    sets::Array{Any, 1},
    order_constraints::Array{Constraints, 1},
    member::Array{Int64, 1},
    param::Dict{Symbol, Any},
    setdist::Distsv,
    use,
    start_set::Int64,
)
    current.cost = tour_cost(current.tour, dist)
    prev_cost = current.cost
    for i in 1:5
        if i % 2 == 1
            current.tour = reopt_tour(current.tour, dist, sets, member, param, start_set)
        elseif param[:mode] == "fast" || use == "partial"
            moveopt_rand!(
                RNGs,
                current.tour,
                dist,
                sets,
                order_constraints,
                member,
                param[:max_removals],
                setdist,
                start_set,
            )
        else
            moveopt!(
                current.tour,
                dist,
                sets,
                order_constraints,
                member,
                setdist,
                start_set,
            )
        end
        current.cost = tour_cost(current.tour, dist)
        if i > 1 && (current.cost >= prev_cost || use == "partial")
            return
        end
        prev_cost = current.cost
    end
end


"""
Given an ordering of the sets, this alg performs BFS to find the 
optimal vertex in each set
"""
function reopt_tour(
    tour::Array{Int64, 1},
    dist::Array{Int64, 2},
    sets::Array{Any, 1},
    member::Array{Int64, 1},
    param::Dict{Symbol, Any},
    start_set::Int64,
)
    best_tour_cost = tour_cost(tour, dist)
    new_tour = copy(tour)
    prev = zeros(Int64, param[:num_vertices])
    cost_to_come = zeros(Int64, param[:num_vertices])
    for start_vertex in sets[member[tour[1]]]
        relax_in!(cost_to_come, dist, prev, Int64[start_vertex], sets[member[tour[2]]])

        # cost to get to ith set on path through (i-1)th set
        for i in 3:length(tour)
            relax_in!(
                cost_to_come,
                dist,
                prev,
                sets[member[tour[i - 1]]],
                sets[member[tour[i]]],
            )
        end

        # find the cost back to the start vertex.
        tour_cost, start_prev =
            relax(cost_to_come, dist, sets[member[tour[end]]], start_vertex)
        if tour_cost < best_tour_cost   # reconstruct the path
            best_tour_cost = tour_cost
            new_tour = extract_tour(prev, start_vertex, start_prev)
        end
    end

    return new_tour
end


""" Find the set with the smallest number of vertices """
function min_setv(
    tour::Array{Int64, 1},
    sets::Array{Any, 1},
    member::Array{Int64, 1},
    param::Dict{Symbol, Any},
)
    min_set = param[:min_set]
    for i in 1:length(tour)
        member[tour[i]] == min_set && return i
    end
    return 1
end


"""
extracting a tour from the prev pointers.
"""
function extract_tour(prev::Array{Int64, 1}, start_vertex::Int64, start_prev::Int64)
    tour = []
    vertex_step = start_prev
    while prev[vertex_step] != 0
        push!(tour, vertex_step)
        vertex_step = prev[vertex_step]
    end
    push!(tour, start_vertex)

    return reverse(tour)
end


"""
outputs the new cost and prev for vertex v2 after relaxing
does not actually update the cost
"""
@inline function relax(
    cost::Array{Int64, 1},
    dist::Array{Int64, 2},
    set1::Array{Int64, 1},
    v2::Int64,
)
    v1 = set1[1]
    min_cost = cost[v1] + dist[v1, v2]
    min_prev = v1
    for i in 2:length(set1)
        v1 = set1[i]
        newcost = cost[v1] + dist[v1, v2]
        if min_cost > newcost
            min_cost, min_prev = newcost, v1
        end
    end
    return min_cost, min_prev
end


"""
relaxes the cost of each vertex in the set set2 in-place.
"""
@inline function relax_in!(
    cost::Array{Int64, 1},
    dist::Array{Int64, 2},
    prev::Array{Int64, 1},
    set1::Array{Int64, 1},
    set2::Array{Int64, 1},
)
    for v2 in set2
        v1 = set1[1]
        cost[v2] = cost[v1] + dist[v1, v2]
        prev[v2] = v1
        for i in 2:length(set1)
            v1 = set1[i]
            newcost = cost[v1] + dist[v1, v2]
            if cost[v2] > newcost
                cost[v2], prev[v2] = newcost, v1
            end
        end
    end
end
