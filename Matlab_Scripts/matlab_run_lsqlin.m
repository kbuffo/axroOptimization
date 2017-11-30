function [x,resnorm,residual,exitflag] = matlab_run_lsqlin(ifs,distortion,lb,ub) 

%options = optimoptions('lsqlin','MaxFunctionEvaluations',100000);
A = []; %eye(size(ifs,2));
b = []; %ones(size(ifs,2),1)*volt_bound;
Aeq = [];
beq = [];
lb = zeros(size(ifs,2),1);
ub = ones(size(ifs,2),1)*10;
% x0 = zeros(size(ifs,2),1);
% nonlcon = [];
[x,resnorm,residual,exitflag] = lsqlin(ifs,distortion,A,b,Aeq,beq,lb,ub)
%,Aeq,beq,lb,ub,x0,options);
end


% ifs = transpose(ifs);
% distortion = transpose(distortion);
% lb = transpose(lb);
% ub = transpose(ub);
% 
% func = @(volts)mean((volts*ifs - distortion).^2);
% options = optimoptions('fmincon','MaxFunctionEvaluations',100000);
% A = [];
% b = [];
% Aeq = [];
% beq = [];
% x0 = zeros(size(ifs,2),1);
% nonlcon = [];
% 
% [x,fval,exitflag,output] = fmincon(func,x0,A,b,Aeq,beq,lb,ub,nonlcon,options);
% %[x,fval,exitflag,output] = size(x0),size(ifs),size(distortion),size(lb)
% end