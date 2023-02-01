
@inline current_time(s) = time_ns()
@inline current_time(s::MPIArray) = MPI.Wtime()

to_seconds(s,t) = Float64(t)/1.0e9
to_seconds(s::MPIArray,t) = Float64(t)

barrier(a) = nothing
barrier(a::MPIArray) = MPI.Barrier(a.comm)

"""
    struct PTimer{...}

Type used to benchmark distributed applications based on PartitionedArrays.

# Properties

Properties and type parameters are private

# Sub-type hierarchy

    PTimer{...} <: Any
"""
mutable struct PTimer{A,B,C}
    parts::A
    timings::B
    current::C
    verbose::Bool
end

"""
    PTimer(ranks;verbose::Bool=false)

Construct an instance of [`PTimer`](@ref) by using the same data
distribution as for `ranks`.  If `verbose==true`, then a message will
be printed each time a new section is added in the timer when calling [`toc!`](@ref).
"""
function PTimer(parts;verbose::Bool=false)
    current = current_time(parts)
    timing = map(p->current,parts)
    T = typeof(timing)
    timings = Dict{String,T}()
    PTimer(parts,timings,current,verbose)
end

function Base.getproperty(t::PTimer, sym::Symbol)
    if sym === :data
        data = map_main(t.parts) do part
            d = (min=0.0,max=0.0,avg=0.0)
            T = typeof(d)
            Dict{String}{T}()
        end
        for (name,timing) in t.timings
            timing_main = gather(timing)
            map_main(timing_main,data) do part_to_timing,data
                ns = map(i->to_seconds(t.parts,i),part_to_timing)
                d = (min=minimum(ns),max=maximum(ns),avg=sum(ns)/length(ns))
                data[name] = d
            end
        end
        return data
    else
        getfield(t, sym)
    end
end

"""
    statistics(t::PTimer)

Return a dictionary with statistics of the compute time for the sections
currently stored in the timer `t`. 
"""
function statistics(t::PTimer)
    d = (min=0.0,max=0.0,avg=0.0)
    T = typeof(d)
    dict = Dict{String}{T}()
    for (name,timing) in t.timings
        part_to_timing = collect(timing)
        ns = map(i->to_seconds(t.parts,i),part_to_timing)
        d = (min=minimum(ns),max=maximum(ns),avg=sum(ns)/length(ns))
        dict[name] = d
    end
    dict
end

function Base.propertynames(x::PTimer, private=false)
    (fieldnames(typeof(x))...,:data)
end

"""
    tic!(t::PTimer;barrier=false)

Reset the timer `t` to start measuring the time in a section.  
If `barrier==true`, all process will be synchronized before 
resetting the timer if using a distributed back-end.
For MPI, this will result in a call to `MPI.Barrier`.
"""
function tic!(t::PTimer;barrier=false)
    if barrier
        PartitionedArrays.barrier(t.parts)
    end
    t.current = current_time(t.parts)
end

"""
    toc!(t::PTimer,name::String)

Finish measuring a code section with name `name` in timer `t`.
"""
function toc!(t::PTimer,name::String)
    current = current_time(t.parts)
    dt = current-t.current
    timing = map(p->dt,t.parts)
    t.timings[name] = timing
    if t.verbose == true
        map_main(timing) do i
            println("[$(lstrip(nice_time(to_seconds(t.parts,i)))) s in MAIN] $name")
        end
    end
    t.current = current_time(t.parts)
end

function Base.show(io::IO,k::MIME"text/plain",t::PTimer)
    print_timer(io,t)
end

function print_timer(
        io::IO,
        t::PTimer;
        linechars::Symbol=:unicode)
    function print_header(io,longest_name,w,linechars)
        rule = linechars === :unicode ? "─" : "-"
        sec_header = rpad("Section",longest_name)
        max_header = lpad("max",w)
        min_header = lpad("min",w)
        avg_header = lpad("avg",w)
        header = sec_header*max_header*min_header*avg_header
        header_w = length(header)
        println(io,rule^header_w)
        println(io,header)
        println(io,rule^header_w)
    end
    function print_section(io,longest_name,name,d)
        sec_header = rpad(name,longest_name)
        max_header = nice_time(d.max)
        min_header = nice_time(d.min)
        avg_header = nice_time(d.avg)
        header = sec_header*max_header*min_header*avg_header
        println(io,header)
    end
    function print_footer(io,longest_name,w,linechars)
        rule = linechars === :unicode ? "─" : "-"
        header_w = longest_name+3*w
        println(io,rule^header_w)
    end
    function print_on_main(io,data,linechars)
        w = length(nice_time(eps()))
        longest_name = maximum(map(i->length(i[1]),data))
        longest_name = max(longest_name,length("Section"))
        print_header(io,longest_name,w,linechars)
        for (name,d) in data
            print_section(io,longest_name,name,d)
        end
        print_footer(io,longest_name,w,linechars)
    end
    map_main(t.data) do data
        kv = collect(data)
        v = map(i->i[2].max,kv)
        i = reverse(sortperm(v))
        sorteddata = kv[i]
        print_on_main(io,sorteddata,linechars)
    end
end

print_timer(t::PTimer; kwargs...) = print_timer(stdout,t;kwargs...)
nice_time(t) = @sprintf("%12.3e",t)




