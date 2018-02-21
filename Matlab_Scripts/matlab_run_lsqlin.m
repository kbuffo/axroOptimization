function [x,resnorm,residual,exitflag] = matlab_run_lsqlin(ifs,distortion,lb,ub) 

%options = optimoptions('lsqlin','MaxFunctionEvaluations',100000);
A = [];
b = [];
Aeq = [];
beq = [];
[x,resnorm,residual,exitflag] = lsqlin(ifs,distortion,A,b,Aeq,beq,lb,ub)
end