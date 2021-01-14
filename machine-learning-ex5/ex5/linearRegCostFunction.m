function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% X is of the size 12 x 2
% Theta is of size 2 x 1

h0 = X * theta;   %h0 of dimensions 12 x 1

J_unreg = (0.5/m) * ( sum( ( h0 - y ).^2 ) );

J_reg = (lambda/(2*m)) * sum(theta(2:end).^2);

J = J_reg + J_unreg;

  grad(1) = (1/m) * sum( (h0 - y).*X(:,1) );

for j=2:size(theta,1)
  
  grad(j) = (1/m) * sum( (h0 - y).*X(:,j) ) + (lambda/m) * theta(j);
  
endfor



% =========================================================================

grad = grad(:);

end
