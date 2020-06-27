library(qrng)

d <- 1;
mat <- matrix(0,20,4);
mat[,1] <- 2^(1:20);
trials <- 40;

for (i in c(1:20)){
    n = mat[i,1];
    # Sobol
    t0 <- Sys.time()
    for (j in (1:trials)){
        x <- sobol(n, d, randomize='digital.shift',seed=7);}
    mat[i,2] <- (Sys.time() - t0)/trials;
    # GHalton
    t0 <- Sys.time()
    for (j in (1:trials)){
        x <- ghalton(n, d, method='generalized');}
    mat[i,3] <- (Sys.time() - t0)/trials;
    # Korobov
    t0 <- Sys.time()
    for (j in (1:trials)){
        x <- korobov(n, d, generator=c(1),randomize='shift');}
    mat[i,4] <- (Sys.time() - t0)/trials;}
df <- as.data.frame(mat)
names(df) <- c('n','Sobol','GHalton','Korobov')
write.table(df,file='r_sequences.csv',row.names=FALSE,sep=',')
