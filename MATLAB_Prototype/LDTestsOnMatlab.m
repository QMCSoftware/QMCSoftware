%% Tests on MATLAB

%% Time Sobol' points
tic
n = 2^20;
d = 20;
x = net(sobolset(d),n);
toc

%% Time lattice points
tic
%n = 2^20;
%d = 1;
x = gail.lattice_gen(1,n,d);
toc
