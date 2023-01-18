
@inline current_time(s) = time_ns()
@inline current_time(s::MPIData) = MPI.Wtime()

to_seconds(s,t) = Float64(t)/1.0e9
to_seconds(s::MPIData,t) = Float64(t)

barrier(a) = nothing
barrier(a::MPIData) = MPI.Barrier(a.comm)

mutable struct PTimer{A,B,C}
    parts::A
    timings::B
    current::C
    verbose::Bool
end

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

function Base.propertynames(x::PTimer, private=false)
    (fieldnames(typeof(x))...,:data)
end

function tic!(t::PTimer;barrier=false)
    if barrier
        PartitionedArrays.barrier(t.parts)
    end
    t.current = current_time(t.parts)
end

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




