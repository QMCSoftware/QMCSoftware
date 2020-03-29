function [q]=vdc(n,method)
% Van der Corput sequence in base 2 where n is a power of 2. We do it this
% way because of our own VDC construction: is much faster and cubLattice
% does not need more.

if strcmp(method,'sobolset')
    q=net(sobolset(1),n);
else % We build it ourselves in cost in nlog(n)
    if n>1
        k=log2(n); % We compute the VDC seq part by part of power 2 size
        q=zeros(2^k,1);
        for l=0:k-1
           nl=2^l;
           kk=2^(k-l-1);
           ptind=repmat([false(nl,1);true(nl,1)],kk,1);
           q(ptind)=q(ptind)+1/2^(l+1);
        end
    else
    q=0;
    end 
end
end

