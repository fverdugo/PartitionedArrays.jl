
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
  t::PTimer,
  filename::String,
  args...;
  linechars::Symbol=:unicode,
  format::Symbol=:csv,
  kwargs...)

  data = _setup_data(t)
  map_parts(t.parts,data) do part,data
    if part == MAIN
      open(filename,args...;kwargs...) do io
        _print_on_main(io,data,linechars,format)
      end
    end
  end
end

function print_timer(
  filename::String,
  t::PTimer;
  linechars::Symbol=:unicode,
  format::Symbol=:csv)

  print_timer(t,filename,"w";linechars=linechars,format=format)
end

function print_timer(
  io::IO,
  t::PTimer;
  linechars::Symbol=:unicode,
  format::Symbol=:REPL)

  data = _setup_data(t)
  map_parts(t.parts,data) do part,data
    if part == MAIN
      _print_on_main(io,data,linechars,format)
    end
  end
end

print_timer(t::PTimer; kwargs...) = print_timer(stdout,t;kwargs...)

function _print_on_main(io,data,linechars,format)
  if format == :REPL
    w = length(_nice_time(eps()))
    longest_name = maximum(map(i->length(i[1]),data))
    longest_name = max(longest_name,length("Section"))
    _print_header(io,longest_name,w,linechars)
    for (name,d) in data
      _print_section(io,longest_name,name,d)
    end
    _print_footer(io,longest_name,w,linechars)
  elseif format == :csv
     for (name,d) in data
       str = "\"$name\"; $(d.max); $(d.min); $(d.avg)"
       println(io,str)
     end
  else
    throw(ArgumentError("$format is an unsupported value for the format kw-argument."))
  end
end

_nice_time(t) = @sprintf("%12.3e",t)

function _setup_data(t)
  data = map_parts(t.parts) do part
    d = (min=0.0,max=0.0,avg=0.0)
    T = typeof(d)
    Dict{String}{T}()
  end
  for (name,timing) in t.timings
    timing_main = gather(timing)
    map_parts(t.parts,timing_main,data) do part, part_to_timing,data
      if part ==MAIN
        ns = map(i->Float64(i.ns)/1.0e9,part_to_timing)
        d = (min=minimum(ns),max=maximum(ns),avg=sum(ns)/length(ns))
        data[name] = d
      end
    end
  end
  map_parts(data) do data
    kv = collect(data)
    v = map(i->i[2].max,kv)
    i = reverse(sortperm(v))
    kv[i]
  end
end

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

function print_csv(
  value::AbstractPData,
  name::AbstractString,
  args...;
  kwargs...)

  parts = get_part_ids(value)
  map_parts(parts,value) do part, value
    if part == MAIN
      open(args...;kwargs...) do io
       str = "\"$name\"; $value"
       println(io,str)
      end
    end
  end
end

function print_csv(
  parts::AbstractPData{<:Integer},
  value,
  name::AbstractString,
  args...;
  kwargs...)

  pdata = map_parts(i->value,parts)
  print_csv(pdata,name,args...;kwargs...)
end

