m_max = 20;
trials = 10;

for i=1:m_max
    n = 2^i;
    
    tic
    for j=1:trials
        vdc(n,'sobolset');
    end
    vdc_sobolset_t = toc/trials;
    
    tic
    for j=1:trials
        vdc(n,'manual');
    end
    vdc_manual_t = toc/trials;
    
    tic
    for j=1:trials
        lattice_gen(1,n,1);
    end
    lattice_manual_t = toc/trials;
    
    fprintf('n: %d\n\tVDC SobolSet Time: %.4f\n\tVCD Manual Time: %.4f\n\tLattice Manual Time: %.4f\n',...
        n,vdc_sobolset_t,vdc_manual_t,lattice_manual_t)
end
