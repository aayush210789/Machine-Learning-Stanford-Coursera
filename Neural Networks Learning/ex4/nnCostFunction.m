function [J grad] = nnCostFunction(nn_params, ...
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
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
J=0;
% Setup some useful variables
m = size(X, 1);
n = size(X, 2);
%accumgrad1 =zeros(s
%accumgrad2 =0.0;
%delta2 = zeros(size(Theta1));
%delta3 = zeros(size(Theta2));

X = [ones(m, 1) X];
z2 = Theta1*X';
a2 = sigmoid(z2);
a2 = [ones(m,1) a2'];
z3 = Theta2*a2';
hx = sigmoid(z3);
% You need to return the following variables correctly 
for i = 1:m
    Y = zeros(num_labels,1);
    Y(y(i))=1; 
       J= J+ ((Y'*log(hx(:,i)) + (1-Y')*log(1-hx(:,i))));
       
% 
% delta3(:,i) = hx(:,i) - Y;
% %disp(size(Theta2));
% b=(Theta2'*delta3);
% b= b(:,2:end);
% delta2(:,i) = b.* sigmoidGradient(z2);
% accumgrad1 = (accumgrad1 + X'*delta2');
% accumgrad2 = (accumgrad2 + a2'*delta3');

end   
% disp(size(delta2));
% disp(size(delta3));

J =J*(-1/m);
J =J + (lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(: , 2:end).^2)));
% Theta2_grad = accumgrad2/m;
% Theta1_grad = accumgrad1/m;
y1=eye(num_labels);
ry = y1(y,:);
delta3 = hx' - ry;
b = (delta3*Theta2);
delta2 = b(:,2:end) .* sigmoidGradient(z2)';

Delta1 = delta2'*X;
Delta2 = delta3'*a2;

Theta1_grad = Delta1 / m + lambda*[zeros(hidden_layer_size , 1) Theta1(:,2:end)] / m;
Theta2_grad = Delta2 / m + lambda*[zeros(num_labels , 1) Theta2(:,2:end)] / m;
% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
