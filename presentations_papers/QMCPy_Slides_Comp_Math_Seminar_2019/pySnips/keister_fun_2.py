integrand = QuickConstruct(\ 
    lambda x: pi**(x.shape[1]/2) * \ 
              cos(norm(x, 2, axis=1)))