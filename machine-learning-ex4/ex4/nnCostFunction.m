function [J,grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

%%% Unregularized Cost Function
%% Activations for Layer1
a1=X; % a1 is of dimensions 5000x400
%% Activations for Layer2
a1=[ones(m,1),a1]; % Adding bias to X, Dimensions of X are 5000*401
z2=a1*Theta1'; % Dimensions of z2 are 5000x25
a2=sigmoid(z2);
%% Activations for Layer3
a2=[ones(size(a2,1),1) a2]; % Adding bias to a2, Dimensions of a2 are 5000x26
z3=a2*Theta2'; % Dimensions of z3 are 5000x10
h3=sigmoid(z3);
h0=h3; % This is Hypothesis values
%% Recode the Vector Y
Yy=zeros(m,num_labels); % Setting up a Matrix Yy of Dimensions 5000x10
for k=1:m
  
  Yy(k,y(k))=1;  % setting up matrix with linear values of Y
  
endfor
Yy;
%% Compute the cost function
Jtemp=(-Yy.*log(h0))-((1-Yy).*log(1-h0)); % Setting up Matrix J with Dimensions 5000x10
J_un=(1/m)*(sum(sum(Jtemp))); % Outputs the cost of Un-Regularized Cost 

%%% Regularized Cost Function
%% Squares of Theta1 Matrix and removing bias
t1=Theta1(:,2:end).^2; % Theta1 of Dimensions 25x401
%% Squares of Theta2 Matrix and removing bias
t2=Theta2(:,2:end).^2; % Theta2 of Dimensions 10x26
%% Regularization Term
reg=(lambda/(2*m))*((sum(sum(t1)))+(sum(sum(t2))));
%% Regularized Cost
J=reg+J_un;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
%% Backward Propogation
d3 = h0 - Yy;
d2 = (d3 * Theta2).*[ones(m,1), sigmoidGradient(z2)];

D1 = d2(:,2:end)'*a1;
D2 =  d3' * a2;

Theta1_grad = (1/m) * D1;
Theta2_grad = (1/m) * D2;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m)*Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m)*Theta2(:,2:end);
















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
