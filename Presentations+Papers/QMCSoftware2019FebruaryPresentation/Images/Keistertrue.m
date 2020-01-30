function I = Keistertrue(d)
%KEISTERTRUE computes the true value of the Keister integral in dimension d
%  accuracy might degrade as d increases due to round-off error
cosinteg=zeros(1,d);
cosinteg(1)=sqrt(pi)/(2*exp(1/4));
sininteg=zeros(1,d);
%sininteg(1)=integral(@(x) exp(-x.*x).*sin(x),0,inf);
sininteg(1)=4.244363835020225e-01;
cosinteg(2)=(1-sininteg(1))/2;
sininteg(2)=cosinteg(1)/2;
for j=3:d
   cosinteg(j)=((j-2)*cosinteg(j-2)-sininteg(j-1))/2;
   sininteg(j)=((j-2)*sininteg(j-2)+cosinteg(j-1))/2;
end
I=(2*(pi.^(d/2))/gamma(d/2))*cosinteg(d);
end

