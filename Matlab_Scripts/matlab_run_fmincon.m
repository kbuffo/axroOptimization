function [x,fval,exitflag] = matlab_run_fmincon(ifs,distortion,lb,ub) 

func = @(volts)mean((ifs*volts - distortion).^2);
options = optimoptions('fmincon','MaxFunctionEvaluations',100000);
A = [];
b = [];
Aeq = [];
beq = [];
lb = zeros(size(ifs,2),1);
ub = ones(size(ifs,2),1)*10;
x0 = zeros(size(ifs,2),1);
nonlcon = [];
[x,fval,exitflag,output] = fmincon(func,x0,A,b,Aeq,beq,lb,ub,nonlcon,options);
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