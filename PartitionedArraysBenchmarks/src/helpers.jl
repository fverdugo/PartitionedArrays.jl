
const template = Ref{String}()

function __init__()
    template[] = read(joinpath(@__DIR__,"jobtemplate.sh"),String)
end

function runjob(cmd,args...;kwargs...)
    filename = create_jobfile(args...;kwargs...)
    run(`$cmd $filename`)
end

function create_jobfile(fname,params;time="00:30:00",
        results_dir = mkpath("results"),
        project=Base.active_project()
)
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
          "resultsdir" => results_dir,
          "project" => project,
         ])
    jobfile = joinpath(results_dir,jobname*".sh")
    open(jobfile,"w") do io
        render(io,template[],jobdict)
    end
    jobfile
end

function experiment(f,jobname,distribute,params;results_dir)
    results_in_main = f(distribute,params)
    map_main(results_in_main) do results
        dict = Dict(map(p->(string(p[1])=>p[2]),collect(pairs(results))))
        jld2_file = joinpath(results_dir,jobname*".jld2")
        save(jld2_file,dict)
    end
end

