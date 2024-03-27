

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

function create_jobfile(template,params;
        filename=jobname,
        project=Base.active_project()
)
    myjobname = filename(params)
    myparams = jobparams(params)
    jobdict = Dict([
          "__PARAMS__" => sprint(show,myparams),
          "__JOBNAME__" => myjobname,
          "__PROJECT__" => project,
         ])
    params_string = Dict(map(p->(string(p[1])=>string(p[2])),collect(pairs(params))))
    jobdict = merge(jobdict,params_string)
    jobfile = myjobname*".sh"
    open(jobfile,"w") do io
        render(io,template,jobdict)
    end
    jobfile
end

function experiment(f,jobname,distribute,params)
    results_in_main = f(distribute,params)
    map_main(results_in_main) do results
        dict = string_dict(results)
        jld2_file = jobname*"_results.jld2"
        save(jld2_file,dict)
    end
end

