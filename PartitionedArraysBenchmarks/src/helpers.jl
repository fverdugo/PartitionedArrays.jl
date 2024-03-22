
const template = Ref{String}()

function __init__()
    template[] = read(joinpath(@__DIR__,"jobtemplate.sh"),String)
end

function runjob(cmd,args...;kwargs...)
    filename = create_jobfile(args...;kwargs...)
    run(`$cmd $filename`)
end

function jobname(params)
    String(sprint(show,hash(params))[3:end])
end

function jobparams(params::NamedTuple)
    params
end

function jobparams(params::Dict)
    (;symbol_dict(params)...)
end

function string_dict(dict::Dict{String})
    dict
end

function string_dict(results)
    Dict(map(p->(string(p[1])=>p[2]),collect(pairs(results))))
end

function symbol_dict(dict::Dict{Symbol})
    dict
end

function symbol_dict(results)
    Dict(map(p->(Symbol(string(p[1]))=>p[2]),collect(pairs(results))))
end

function create_jobfile(params;
        time="00:30:00",
        results_dir = mkpath("results"),
        project=Base.active_project()
)
    myjobname = jobname(params)
    myparams = jobparams(params)
    jobdict = Dict([
          "time" => time,
          "nodes" => string(myparams.nodes),
          "ntasks_per_node" => string(myparams.ntasks_per_node),
          "output" => myjobname*".o",
          "error" =>  myjobname*".e",
          "mpiexec" => "mpiexec",
          "np" => string(myparams.np),
          "params" => sprint(show,myparams),
          "benchmark" => myparams.benchmark,
          "jobname" => myjobname,
          "resultsdir" => results_dir,
          "project" => project,
         ])
    jobfile = joinpath(results_dir,myjobname*".sh")
    open(jobfile,"w") do io
        render(io,template[],jobdict)
    end
    jobfile
end

function experiment(f,jobname,distribute,params;results_dir)
    results_in_main = f(distribute,params)
    map_main(results_in_main) do results
        dict = string_dict(results)
        jld2_file = joinpath(results_dir,jobname*"_results.jld2")
        save(jld2_file,dict)
    end
end

