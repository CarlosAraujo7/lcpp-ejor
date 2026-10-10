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


mutable struct TrialResult
    tour::Tour
    insertion::Power
    removal::Power
    noise::Power
    score::Float64
end


"""
Select a removal and an insertion method using powers, and then perform
removal followed by insertion on tour.  Operation done in place.
"""
function remove_insert(
    RNGs::Array{MersenneTwister, 1},
    current::Tour,
    best::Tour,
    dist::Array{Int64, 2},
    member::Array{Int64, 1},
    setdist::Distsv,
    sets::Array{Any, 1},
    order_constraints::Array{Constraints, 1},
    powers,
    param::Dict{Symbol, Any},
    phase::Symbol,
    start_set::Int64,
)
    # make a new tour to perform the insertion and deletion on
    trial = Tour(copy(current.tour), current.cost)
    pivot_tour!(RNGs, trial.tour)
    num_removals = rand(RNGs[Threads.threadid()], param[:min_removals]:param[:max_removals])

    removal = power_select(RNGs, powers["removals"], powers["removal_total"], phase)
    if removal.name == "distance"
        sets_to_insert = distance_removal!(
            RNGs,
            trial.tour,
            dist,
            num_removals,
            member,
            removal.value,
            start_set,
        )
    elseif removal.name == "worst"
        sets_to_insert = worst_removal!(
            RNGs,
            trial.tour,
            dist,
            num_removals,
            member,
            removal.value,
            start_set,
        )
    else
        sets_to_insert = segment_removal!(RNGs, trial.tour, num_removals, member, start_set)
    end

    randomize_sets!(RNGs, sets, sets_to_insert)
    # then perform insertion
    insertion = power_select(RNGs, powers["insertions"], powers["insertion_total"], phase)
    noise = power_select(RNGs, powers["noise"], powers["noise_total"], phase)

    if insertion.name == "cheapest"
        cheapest_insertion!(
            trial.tour,
            sets_to_insert,
            dist,
            setdist,
            sets,
            order_constraints,
            member,
            start_set,
        )
    else
        randpdf_insertion!(
            RNGs,
            trial.tour,
            sets_to_insert,
            dist,
            setdist,
            sets,
            order_constraints,
            member,
            insertion.value,
            noise,
            start_set,
        )
    end
    rand(RNGs[Threads.threadid()]) < param[:prob_reopt] && opt_cycle!(
        RNGs,
        trial,
        dist,
        sets,
        order_constraints,
        member,
        param,
        setdist,
        "partial",
        start_set,
    )

    score = 100 * max(current.cost - trial.cost, 0) / current.cost

    return trial, insertion, removal, noise, score
end


"""
Select an integer between 1 and num according to
and exponential distribution with lambda = power
# goes from left of array if power is positive
# and right of array if it is negative
"""
function select_k(RNGs::Array{MersenneTwister, 1}, num::Int64, power::Float64)
    base = (1 / 2)^abs(power)
    # (1 - base^num)/(1 - base) is sum of geometric series
    rand_select = (1 - base^num) / (1 - base) * rand(RNGs[Threads.threadid()])
    bin = 1
    for k in 1:num
        if rand_select < bin
            return (power >= 0 ? (num - k + 1) : k)
        end
        rand_select -= bin
        bin *= base
    end
    return (power >= 0 ? num : 1)
end


"""
selecting a random k in 1 to length(weights) according to power
and then selecting the kth smallest element in weights
"""
function pdf_select(
    RNGs::Array{MersenneTwister, 1},
    weights::Array{Int64, 1},
    power::Float64,
)
    power == 0.0 && return rand(RNGs[Threads.threadid()], 1:length(weights))
    power > 9.0 && return rand_select(RNGs, weights, maximum(weights))
    power < -9.0 && return rand_select(RNGs, weights, minimum(weights))

    # select kth smallest.  If 1 or length(weights), simply return
    k = select_k(RNGs, length(weights), power)
    k == 1 && return rand_select(RNGs, weights, minimum(weights))
    k == length(weights) && return rand_select(RNGs, weights, maximum(weights))
    val = partialsort(weights, k)

    return rand_select(RNGs, weights, val)
end


"""
choose set with pdf_select, and then insert in best place with noise
"""
function randpdf_insertion!(
    RNGs::Array{MersenneTwister, 1},
    tour::Array{Int64, 1},
    sets_to_insert::Array{Int64, 1},
    dist::Array{Int64, 2},
    setdist::Distsv,
    sets::Array{Any, 1},
    order_constraints::Array{Constraints, 1},
    member::Array{Int64, 1},
    power::Float64,
    noise::Power,
    start_set::Int64,
)
    mindist = [typemax(Int64) for i in 1:length(sets_to_insert)]
    for i in 1:length(sets_to_insert)
        set = sets_to_insert[i]
        for vertex in tour
            if setdist.min_sv[set, vertex] < mindist[i]
                mindist[i] = setdist.min_sv[set, vertex]
            end
        end
    end
    new_vertex_in_tour = 0

    while length(sets_to_insert) > 0
        if new_vertex_in_tour != 0
            for i in 1:length(sets_to_insert)
                set = sets_to_insert[i]
                if setdist.min_sv[set, new_vertex_in_tour] < mindist[i]
                    mindist[i] = setdist.min_sv[set, new_vertex_in_tour]
                end
            end
        end
        set_index = pdf_select(RNGs, mindist, power) # select set to insert from pdf
        # find the closest vertex and the best insertion in that vertex
        nearest_set = sets_to_insert[set_index]

        min_insert_idx, max_insert_idx =
            calc_bounds(tour, nearest_set, order_constraints, member)

        if nearest_set != start_set
            min_insert_idx = max(min_insert_idx, 2)
            if max_insert_idx < min_insert_idx
                max_insert_idx = min_insert_idx
            end
        end

        if noise.name == "subset"
            bestv, bestpos = insert_subset_lb(
                RNGs,
                tour,
                dist,
                sets[nearest_set],
                nearest_set,
                setdist,
                noise.value,
            )
        else
            bestv, bestpos = insert_lb(
                RNGs,
                tour,
                dist,
                sets[nearest_set],
                nearest_set,
                min_insert_idx,
                max_insert_idx,
                setdist,
                noise.value,
            )
        end
        insert!(tour, bestpos, bestv)  # perform the insertion
        new_vertex_in_tour = bestv
        # remove the inserted set from data structures
        splice!(sets_to_insert, set_index)
        splice!(mindist, set_index)
    end
end


"""
choose vertex that can be inserted most cheaply, and insert it in that position
"""
function cheapest_insertion!(
    tour::Array{Int64, 1},
    sets_to_insert::Array{Int64, 1},
    dist::Array{Int64, 2},
    setdist::Distsv,
    sets::Array{Any, 1},
    order_constraints::Array{Constraints, 1},
    member::Array{Int64, 1},
    start_set::Int64,
)
    while length(sets_to_insert) > 0
        best_cost = typemax(Int64)
        curr_v = 0
        curr_pos = 0
        best_set = 0
        best_v = 0
        best_pos = 0
        for i in 1:length(sets_to_insert)
            set_ind = sets_to_insert[i]

            min_insert_idx, max_insert_idx =
                calc_bounds(tour, set_ind, order_constraints, member)

            if set_ind != start_set
                min_insert_idx = max(min_insert_idx, 2)
                if max_insert_idx < min_insert_idx
                    max_insert_idx = min_insert_idx
                end
            end

            # find the best place to insert the vertex
            curr_v, curr_pos, cost = insert_cost_lb(
                tour,
                dist,
                sets[set_ind],
                set_ind,
                setdist,
                min_insert_idx,
                max_insert_idx,
                curr_v,
                curr_pos,
                best_cost,
            )
            if cost < best_cost
                best_set = i
                best_cost = cost
                best_v = curr_v
                best_pos = curr_pos
            end
        end

        # now, perform the insertion
        insert!(tour, best_pos, best_v)
        # remove the inserted set from data structures
        splice!(sets_to_insert, best_set)
    end
end


"""
Given a tour and a set, this function finds the vertex in the set with minimum
insertion cost, along with the position of this insertion in the tour.  If
best_position is i, then vertex should be inserted between tour[i-1] and tour[i].
"""
@inline function insert_lb(
    RNGs::Array{MersenneTwister, 1},
    tour::Array{Int64, 1},
    dist::Array{Int64, 2},
    set::Array{Int64, 1},
    setind::Int,
    min_insert_idx::Int,
    max_insert_idx::Int,
    setdist::Distsv,
    noise::Float64,
)
    best_cost = typemax(Int64)
    bestv = 0
    bestpos = 0

    if min_insert_idx == max_insert_idx && max_insert_idx == length(tour) + 1
        v1 = tour[length(tour)]
        for v in set
            insert_cost = dist[v1, v]
            noise > 0.0 && (
                insert_cost +=
                    round(Int64, noise * rand(RNGs[Threads.threadid()]) * abs(insert_cost))
            )
            if insert_cost < best_cost
                best_cost = insert_cost
                bestv = v
            end
        end
        bestpos = max_insert_idx
        return bestv, bestpos
    end

    for i in min_insert_idx:max_insert_idx
        v1 = prev_tour(tour, i)
        lb =
            setdist.vert_set[v1, setind] + setdist.set_vert[setind, tour[i]] -
            dist[v1, tour[i]]
        lb > best_cost && continue

        for v in set
            insert_cost = dist[v1, v] + dist[v, tour[i]] - dist[v1, tour[i]]
            noise > 0.0 && (
                insert_cost +=
                    round(Int64, noise * rand(RNGs[Threads.threadid()]) * abs(insert_cost))
            )
            if insert_cost < best_cost
                best_cost = insert_cost
                bestv = v
                bestpos = i
            end
        end
    end
    return bestv, bestpos
end


@inline function insert_subset_lb(
    RNGs::Array{MersenneTwister, 1},
    tour::Array{Int64, 1},
    dist::Array{Int64, 2},
    set::Array{Int64, 1},
    setind::Int,
    setdist::Distsv,
    noise::Float64,
)
    best_cost = typemax(Int64)
    bestv = 0
    bestpos = 0
    tour_inds = collect(1:length(tour))

    for i in 1:ceil(Int64, length(tour) * noise)
        i = incremental_shuffle!(RNGs, tour_inds, i)
        v1 = prev_tour(tour, i)
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
    return bestv, bestpos
end


############ Initial Tour Construction ##########################

"""build tour from scratch on a cold restart"""
function initial_tour!(
    RNGs::Array{MersenneTwister, 1},
    lowest::Tour,
    dist::Array{Int64, 2},
    sets::Array{Any, 1},
    order_constraints::Array{Constraints, 1},
    setdist::Distsv,
    trial_num::Int64,
    member::Array{Int64, 1},
    param::Dict{Symbol, Any},
    start_set::Int64,
)
    sets_to_insert = collect(1:param[:num_sets])
    best = Tour(Int64[], typemax(Int64))

    # compute random initial tour only if past first trial
    # in this case, randomly choose between random and insertion tour.
    if param[:init_tour] == "rand" &&
       (trial_num > 1) &&
       (rand(RNGs[Threads.threadid()]) < 0.5)
        random_initial_tour!(
            RNGs,
            best.tour,
            sets_to_insert,
            dist,
            sets,
            order_constraints,
            member,
            start_set,
        )
    else
        random_insertion!(
            RNGs,
            best.tour,
            sets_to_insert,
            dist,
            sets,
            order_constraints,
            setdist,
            member,
            start_set,
        )
    end
    best.cost = tour_cost(best.tour, dist)
    lowest.cost > best.cost && (lowest = best)
    return best
end


"""
Randomly shuffle the sets, and then insert the best vertex from each set back into
the tour where sets are considered in shuffled order.
"""
function random_insertion!(
    RNGs::Array{MersenneTwister, 1},
    tour::Array{Int64, 1},
    sets_to_insert::Array{Int64, 1},
    dist::Array{Int64, 2},
    sets::Array{Any, 1},
    order_constraints::Array{Constraints, 1},
    setdist::Distsv,
    member::Array{Int64, 1},
    start_set::Int64,
)
    # Randomly permute the sets.
    shuffle!(RNGs[Threads.threadid()], sets_to_insert)
    pushed_cnt = 0
    to_insert = copy(sets_to_insert)
    while !isempty(to_insert)
        pushed_cnt += 1

        if pushed_cnt == 1
            rnd_set_idx = findfirst(x -> x == start_set, to_insert)
            rnd_set = to_insert[rnd_set_idx]
        else
            rnd_set_idx = rand(RNGs[Threads.threadid()], 1:length(to_insert))
            rnd_set = to_insert[rnd_set_idx]
        end

        min_insert_idx, max_insert_idx =
            calc_bounds(tour, rnd_set, order_constraints, member)

        if pushed_cnt > 1
            min_insert_idx = max(min_insert_idx, 2)
            if max_insert_idx < min_insert_idx
                max_insert_idx = min_insert_idx
            end
        end

        # Only have to compute the insert cost for the changed portion of the tour.
        if min_insert_idx > length(tour)
            best_vertex = rand(RNGs[Threads.threadid()], sets[rnd_set])
            best_position = length(tour) + 1
        elseif isempty(tour)
            best_vertex = rand(RNGs[Threads.threadid()], sets[rnd_set])
            best_position = 1
        else
            best_vertex, best_position = insert_lb(
                RNGs,
                tour,
                dist,
                sets[rnd_set],
                rnd_set,
                min_insert_idx,
                max_insert_idx,
                setdist,
                0.75,
            )
        end

        # Now, perform the insertion.
        insert!(tour, best_position, best_vertex)
        deleteat!(to_insert, rnd_set_idx)
    end
end


"""
Randomly shuffle the sets, and then insert the best vertex from each set back into
the tour where sets are considered in shuffled order.
"""
function random_initial_tour!(
    RNGs::Array{MersenneTwister, 1},
    tour::Array{Int64, 1},
    sets_to_insert::Array{Int64, 1},
    dist::Array{Int64, 2},
    sets::Array{Any, 1},
    order_constraints::Array{Constraints, 1},
    member::Array{Int64, 1},
    start_set::Int64,
)
    shuffle!(RNGs[Threads.threadid()], sets_to_insert)
    pushed_cnt = 0
    to_insert = copy(sets_to_insert)
    while !isempty(to_insert)
        pushed_cnt += 1

        if pushed_cnt == 1
            rnd_set_idx = findfirst(x -> x == start_set, to_insert)
            rnd_set = to_insert[rnd_set_idx]
        else
            rnd_set_idx = rand(RNGs[Threads.threadid()], 1:length(to_insert))
            rnd_set = to_insert[rnd_set_idx]
        end

        min_insert_idx, max_insert_idx =
            calc_bounds(tour, rnd_set, order_constraints, member)

        if pushed_cnt > 1
            min_insert_idx = max(min_insert_idx, 2)
            if max_insert_idx < min_insert_idx
                max_insert_idx = min_insert_idx
            end
        end

        idx_to_insert = rand(RNGs[Threads.threadid()], min_insert_idx:max_insert_idx)
        vert_to_insert = rand(RNGs[Threads.threadid()], sets[rnd_set])
        insert!(tour, idx_to_insert, vert_to_insert)
        deleteat!(to_insert, rnd_set_idx)
    end
end


######################### Removals ################################
"""
Remove the vertices randomly, but biased towards those that add the most length to the
tour.  Bias is based on the power input.  Vertices are then selected via pdf select.
"""
function worst_removal!(
    RNGs::Array{MersenneTwister, 1},
    tour::Array{Int64, 1},
    dist::Array{Int64, 2},
    num_to_remove::Int64,
    member::Array{Int64, 1},
    power::Float64,
    start_set::Int64,
)
    deleted_sets = Array{Int}(undef, 0)
    while length(deleted_sets) < num_to_remove
        removal_costs = worst_vertices(tour, dist)
        ind = pdf_select(RNGs, removal_costs[2:length(removal_costs)], power) + 1
        set_to_delete = member[tour[ind]]

        # perform the deletion
        push!(deleted_sets, set_to_delete)
        splice!(tour, ind)
    end
    return deleted_sets
end


""" removing a single continuos segment of the tour of size num_remove """
function segment_removal!(
    RNGs::Array{MersenneTwister, 1},
    tour::Array{Int64, 1},
    num_to_remove::Int64,
    member::Array{Int64, 1},
    start_set::Int64,
)
    i = rand(RNGs[Threads.threadid()], 2:length(tour))
    deleted_sets = Array{Int}(undef, 0)
    while length(deleted_sets) < num_to_remove
        i > length(tour) && (i = 2)
        set_to_delete = member[tour[i]]
        if set_to_delete == start_set
            continue
        end
        push!(deleted_sets, set_to_delete)
        splice!(tour, i)
    end
    return deleted_sets
end


"""  pick a random vertex, and delete its closest neighbors  """
function distance_removal!(
    RNGs::Array{MersenneTwister, 1},
    tour::Array{Int64, 1},
    dist::Array{Int64, 2},
    num_to_remove::Int64,
    member::Array{Int64, 1},
    power::Float64,
    start_set::Int64,
)
    deleted_sets = Array{Int}(undef, 0)
    deleted_vertices = Array{Int}(undef, 0)

    seed_index = rand(RNGs[Threads.threadid()], 2:length(tour))
    push!(deleted_sets, member[tour[seed_index]])
    push!(deleted_vertices, tour[seed_index])
    splice!(tour, seed_index)

    while length(deleted_sets) < num_to_remove
        # pick a random vertex from the set of deleted vertices
        seed_vertex = rand(RNGs[Threads.threadid()], deleted_vertices)
        # find closest vertex to the seed vertex that's still in the tour
        mindist = zeros(Int64, length(tour) - 1)
        for i in 2:length(tour)
            mindist[i - 1] = min(dist[seed_vertex, tour[i]], dist[tour[i], seed_vertex])
        end
        del_index = pdf_select(RNGs, mindist, power) + 1
        set_to_delete = member[tour[del_index]]
        if set_to_delete == start_set
            continue
        end
        push!(deleted_sets, set_to_delete)
        push!(deleted_vertices, tour[del_index])
        splice!(tour, del_index)
    end

    return deleted_sets
end


"""
determine the cost of removing each vertex from the tour, given that all others remain.
"""
function worst_vertices(tour::Array{Int64, 1}, dist::Array{Int64, 2})
    removal_cost = zeros(Int64, length(tour))
    for i in 2:length(tour)
        if i == length(tour)
            removal_cost[i] =
                dist[tour[i - 1], tour[i]] + dist[tour[i], tour[1]] -
                dist[tour[i - 1], tour[1]]
        else
            removal_cost[i] =
                dist[tour[i - 1], tour[i]] + dist[tour[i], tour[i + 1]] -
                dist[tour[i - 1], tour[i + 1]]
        end
    end
    return removal_cost
end
