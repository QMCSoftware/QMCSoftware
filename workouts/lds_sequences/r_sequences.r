library(qrng)

dim <- 1;
n_2powers <- (1:20);
sobol_times <- n_2powers;
trials = 40;

for (i in n_2powers){
    t0 <- Sys.time()
    for (j in (1:trials)){
        x <- sobol(2^i, d=dim, randomize='Owen',7);}
    sobol_times[i] = (Sys.time() - t0)/trials;
    cat(sprintf('n %-10d sobol time %-10.4f\n',2^i,sobol_times[i]))}

results = cbind(2^n_2powers,sobol_times)
write.table(results,file='r_sequences.csv')
