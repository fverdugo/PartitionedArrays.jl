mutable struct Mixed_base_counter
    length::Int
    max_counts::Vector{Int}
    cur_counts::Vector{Int}
end

function mixedbasecounter(counts, l)
    max_counts = zeros(33)
    cur_counts = zeros(33)

    for i in 1:l
        max_counts[i] = counts[i]
    end

    max_counts[l+1] = 0
    return l, max_counts, cur_counts
end

function mixedbasecounter1(left, right)
    max_counts = zeros(33)
    cur_counts = zeros(33)
    length = left.length
    for i in 1:left.length
        max_counts[i] = left.max_counts[i] - right.cur_counts[i]
    end
    return Mixed_base_counter(length, max_counts, cur_counts)
end

function next(counter)
    for i in 1:counter.length
        counter.cur_counts[i] += 1
        if counter.cur_counts[i] > counter.max_counts[i]
            counter.cur_counts[i] = 0
            continue
        end
        break
    end
    return counter
end

function is_zero(counter)
    for i in 1:counter.length
        if counter.cur_counts[i] != 0
            return true
        end
    end
    return false
end

function product(counter, multipliers)
    k = 0
    x = 1
    for i in 1:counter.length
        for j in 1:counter.cur_counts[i]

            k = 1
            x *= multipliers[i]
        end
    end
    return x * k
end
