function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%
%% Cost function

z=X*theta;
h=sigmoid(z);     %calculating the hypothesis
L1=log(h);         %taking the log
t1=-y.*L1;         %first term calculations

L2=log(1-h);      %taking logs  
t2=(1-y).*L2;     %second term calculations

J=sum(t1-t2)/m;

%% Gradient function
for i=1:length(theta)
N=(h-y).*X(:,i);
S=sum(N);
grad(i,1)=S/m;
% =============================================================
end
end
