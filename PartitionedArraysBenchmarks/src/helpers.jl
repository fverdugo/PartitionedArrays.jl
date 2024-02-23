
const template = Ref{String}()

function __init__()
    template[] = read(joinpath(@__DIR__,"jobtemplate.sh"),String)
end

function runjob(cmd,args...;kwargs...)
    filename = create_jobfile(args...;kwargs...)
    run(`$cmd $filename`)
end

function  create_jobfile(fname,params;time="00:30:00",project=".")
    jobname = "$(fname)_"*String(sprint(show,hash(params))[3:end])
    jobdict = Dict([
          "time" => time,
          "nodes" => string(params.nodes),
          "ntasks_per_node" => string(params.ntasks_per_node),
          "output" => jobname*".o",
          "error" =>  jobname*".e",
          "mpiexec" => "mpiexec",
          "np" => string(params.np),
          "params" => sprint(show,params),
          "function" => fname,
          "jobname" => jobname,
          "project" => project,
         ])
    jobfile = jobname*".sh"
    open(jobfile,"w") do io
        render(io,template[],jobdict)
    end
    jobfile
end

function experiment(f,jobname,distribute,params)
    results_in_main = f(distribute,params)
    map_main(results_in_main) do results
        dict = Dict(map(p->(string(p[1])=>p[2]),collect(pairs(results))))
        save(jobname*".jld2",dict)
    end
end

