clc
clear

dim = 1;
lattice_shift = rand(1,dim);
n_2powers = (1:20)';
lattice_times = n_2powers;
sobol_times = n_2powers;
trials = 40;
for i=1:size(n_2powers,1)
    % Lattice
    tic;
    for j=1:trials
      x_lat = gail.lattice_gen(1,2^i,dim);
      x_lat_shifted = mod(x_lat+lattice_shift,1);
    end
    lattice_times(i) = toc/trials;
    % Sobol
    tic;
    for j=1:trials
      sob = scramble(sobolset(dim),'MatousekAffineOwen');
      x_Sobol_scrambled = net(sob,2^i);
    end
    sobol_times(i) = toc/trials;
end

results = [2.^n_2powers, lattice_times, sobol_times]
csvwrite('matlab_sequence_times.csv',results)
