function [J, grad,theta1] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%% General Calculations
z=X*theta;
h0=sigmoid(z);

%% COST FUNCTION

t1=-y.*log(h0);
t2=(1-y).*log(1-h0);
J1=(1/m)*(sum(t1-t2));

theta1=theta;
theta1(1,1)=0;
J2=(lambda/(2*m))*(sum(theta1.^2));

J=(J1+J2);



%% GRADIENT FUNCTION
% theta j=0

grad(1,1)=(1/m)*sum(h0-y);

%% theta >0

for i=2:size(X,2)
  
g1=(1/m)*sum((h0-y).*X(:,i));
g2=(lambda/m)*theta(i,1);
grad(i,1)=g1+g2;
  
endfor
% =============================================================

end
