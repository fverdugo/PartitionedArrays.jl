
include("bench.jl")

function runbench(;
  nc::Tuple=(10,10,10),
  np::Tuple=(2,2,2),
  nr::Int=1,
  title::String="seq")

  parts = PArrays.get_part_ids(PArrays.sequential,np)
  for r in 1:nr
    str_r = lpad(r,ceil(Int,log10(nr)),'0')
    title_r = "$(title)_r$(str_r)"
    bench(parts,nc,title_r)
  end
end

runbench(nc=(100,100,100),np=(2,2,2),nr=1)
#runbench(nc=(10,10,10),np=(2,2,2),nr=1)
#runbench(nc=(4,4),np=(2,2),nr=1)
