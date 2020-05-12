% can be run parallel to to_debug file's CubLattice_g example as a check

function cl_wo(inputArg1,inputArg2)
    f = @(x) 5*sum(x,2);
    hyperbox = [zeros(1,2);ones(1,2)];
    [q,out_param] = cubLattice_g(f,hyperbox,'uniform',1e-4,0,'transform','id');
    q
    out_param
end

