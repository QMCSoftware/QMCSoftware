clc
clear

dim = 1;
lattice_shift = rand(1,dim);
n_2powers = (1:20)';
lattice_times = n_2powers;
sobol_times = n_2powers;
for i=1:size(n_2powers,1)
    % Lattice
    t=cputime;
    x_lat = gail.lattice_gen(1,2^i,dim);
    x_lat_shifted = mod(x_lat+lattice_shift,1);
    lattice_times(i) = cputime-t;
    % Sobol
    t=cputime;
    sob = scramble(sobolset(dim),'MatousekAffineOwen');
    x_Sobol_scrambled = net(sob,2^i);
    sobol_times(i) = cputime-t;
end

results = [n_2powers, lattice_times, sobol_times]
csvwrite('matlab_sequence_times.csv',results)
