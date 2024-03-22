module Analysis
using DataFrames
using FileIO
using JLD2

import PartitionedArraysBenchmarks as pb

parts_per_dir_all = [(1,1,1),(2,1,1),(4,1,1)]
methods_all = ["psprse","petsc_coo"]

for parts_per_dir in parts_per_dir_all
    for method in methods_all
        params = (;
                  nodes = 1,
                  ntasks_per_node = prod(parts_per_dir),
                  np = prod(parts_per_dir),
                  method,
                  cells_per_dir = (10,10,10),
                  parts_per_dir,
                  nruns = 10
                 )
        pb.runjob(:bash,:benchmark_psparse,params)
    end
end

function reduce_timing(ts)
    nworkers = length(ts)
    nruns = length(first(ts))
    t = minimum(ir->maximum(iw->ts[iw][ir],1:nworkers),1:nruns)
    t
end
resuts_dir = "results"
files = readdir(resuts_dir)
files = filter(f->occursin("benchmark_psparse",f) && f[end-4:end]==".jld2",files)
dicts = map(f->load(joinpath(resuts_dir,f)),files)
df = DataFrame(dicts)
df[!,:buildmat] = reduce_timing.(df[!,:buildmat])
df[!,:rebuildmat] = reduce_timing.(df[!,:rebuildmat])
sort!(df,[:method,:np])

display(df)


end # module
