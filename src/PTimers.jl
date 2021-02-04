
struct Timing
  ns::UInt
end

function Base.:-(a::Timing,b::Timing)
  ns = a.ns-b.ns
  Timing(ns)
end

@inline Timing() = Timing(time_ns())

mutable struct PTimer{A,B<:Dict}
  parts::A
  timings::B
  current::Timing
end

function PTimer(parts::AbstractPData{<:Integer})
  current = Timing()
  timing = map_parts(p->current,parts)
  T = typeof(timing)
  timings = Dict{String,T}()
  PTimer(parts,timings,current)
end

function Base.getproperty(t::PTimer, sym::Symbol)
  if sym == :data
    data = map_main(t.parts) do part
      d = (min=0.0,max=0.0,avg=0.0)
      T = typeof(d)
      Dict{String}{T}()
    end
    for (name,timing) in t.timings
      timing_main = gather(timing)
      map_main(timing_main,data) do part_to_timing,data
        ns = map(i->Float64(i.ns)/1.0e9,part_to_timing)
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

function tic!(t::PTimer)
  t.current = Timing()
end

function toc!(t::PTimer,name::String)
  current = Timing()
  dt = current-t.current
  timing = map_parts(p->dt,t.parts)
  t.timings[name] = timing
  t.current = Timing()
end

function Base.show(io::IO,k::MIME"text/plain",t::PTimer)
  print_timer(io,t)
end

function print_timer(
  io::IO,
  t::PTimer;
  linechars::Symbol=:unicode)

  map_main(t.data) do data
    kv = collect(data)
    v = map(i->i[2].max,kv)
    i = reverse(sortperm(v))
    sorteddata = kv[i]
    _print_on_main(io,sorteddata,linechars)
  end
end

print_timer(t::PTimer; kwargs...) = print_timer(stdout,t;kwargs...)

function _print_on_main(io,data,linechars)
  w = length(_nice_time(eps()))
  longest_name = maximum(map(i->length(i[1]),data))
  longest_name = max(longest_name,length("Section"))
  _print_header(io,longest_name,w,linechars)
  for (name,d) in data
    _print_section(io,longest_name,name,d)
  end
  _print_footer(io,longest_name,w,linechars)
end

_nice_time(t) = @sprintf("%12.3e",t)

function _print_header(io,longest_name,w,linechars)
  rule = linechars == :unicode ? "─" : "-"
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

function _print_section(io,longest_name,name,d)
  sec_header = rpad(name,longest_name)
  max_header = _nice_time(d.max)
  min_header = _nice_time(d.min)
  avg_header = _nice_time(d.avg)
  header = sec_header*max_header*min_header*avg_header
  println(io,header)
end

function _print_footer(io,longest_name,w,linechars)
  rule = linechars == :unicode ? "─" : "-"
  header_w = longest_name+3*w
  println(io,rule^header_w)
end

