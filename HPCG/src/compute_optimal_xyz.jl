include("mixed_base_counter.jl")

"""
    compute_optimal_shape_XYZ(np)
    
    Calculate the optimal way to partition a 3D shape over np processors.
"""
function compute_optimal_shape_XYZ(np)

    if np == 1
        return 1, 1, 1
    end

    factors = Primes.factor(DataStructures.SortedDict, np)
    primes = collect(keys(factors))
    z = 0
    x = primes[1]

    if (length(primes) > 1)
        y = primes[2]
    end

    if length(primes) == 1
        z = x^(floor(Int, factors[x] / 3))
        y = x^(floor(Int, factors[x] / 3 + ((factors[x] % 3) >= 2 ? 1 : 0)))
        x = x^(floor(Int, factors[x] / 3 + ((factors[x] % 3) >= 1 ? 1 : 0)))
    elseif length(primes) == 2 && factors[x] == 1 && factors[y] == 1 # two distinct prime factors
        z = 1
    elseif length(primes) == 2 && factors[x] + factors[y] == 3 # three prime factors one repeated
        z = factors[x] == 2 ? x : y
    elseif length(primes) == 3 && factors[x] == 1 && factors[y] == 1 && factors[primes[3]] == 1 # three distinct and single prime factors
        z = primes[3]
    else # 3 or more prime factors so try all possible 3-subsets

        powers = collect(values(factors))
        l, m, c = mixedbasecounter(powers, length(primes))
        c_main = Mixed_base_counter(l, m, c)
        c1 = Mixed_base_counter(l, m, c)

        min_area = 2.0 * np + 1.0

        c1 = next(c1)
        while is_zero(c1)
            c2 = mixedbasecounter1(c_main, c1)
            c2 = next(c2)
            while is_zero(c2)
                tf1 = product(c1, primes)
                tf2 = product(c2, primes)
                tf3 = np / tf1 / tf2 # we derive the third dimension, we don't keep track of the factors it has

                area = tf1 * tf2 + tf2 * tf3 + tf1 * tf3
                if (area < min_area)
                    min_area = area
                    x = tf1
                    y = tf2
                    z = tf3
                end
                c2 = next(c2)
            end
            c1 = next(c1)
        end
    end
    return x, y, floor(Int, z)
end
