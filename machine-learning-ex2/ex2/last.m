z=X*theta;
h=sigmoid(z);     %calculating the hypothesis
L1=log(h);         %taking the log
t1=-y.*L1;         %first term calculations

L2=log(1-h);      %taking logs  
t2=(1-y).*L2;     %second term calculations

J1=sum(t1-t2)/m;

L3=sum(theta.^2);
J2=(lambda/(2*m))*L3;

J=J1+J2;

%% Gradient Function
%For theta 0
N=sum(h-y);
grad(1,1)=N/m;
% For theta >0
for i=2:size(X,2)
  
  Nn=(h-y).*X(:,i);
  Mm=sum(Nn)/m;
  grad(i,1)=Mm;
  
endfor