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

%[p,z] = predict_nn_frwd(Theta1, Theta2, X);

X_new = [ones(size(X,1),1),X];
a = X_new * Theta1';
a1 = sigmoid(a);
a1_new = [ones(size(a1,1),1),a1];
a2 = a1_new * Theta2';
z = sigmoid(a2);


%y_vec = convert_y_multi_class_vec(y,num_labels);
y_vec = zeros(size(y,1),num_labels);
for i=1:size(y,1),
	y(i);
	y_vec(i,y(i))=1;
end

hxtheta_vec = z;

J1=zeros(m,1);
%J1 = y_vec' * log(hxtheta_vec) + (1-y_vec)' * (log(1-hxtheta_vec)); 
for i=1:m,
	J1(i,1) = y_vec(i,:) * log(hxtheta_vec(i,:))' + (1-y_vec(i,:))*(log(1-hxtheta_vec(i,:)))';
end

J = (-1)*(sum(J1)/m);

% -------------------------------------------------------------------------
Reg1=0;
Reg2=0;
for i=1:size(Theta1,1),
	for j=2:size(Theta1,2),
		Reg1 = Reg1+Theta1(i,j)*Theta1(i,j);
	end
end
for i=1:size(Theta2,1),
	for j=2:size(Theta2,2),
		Reg2 = Reg2+Theta2(i,j)*Theta2(i,j);
	end
end
Reg = (lambda/(2*m))*(Reg1+Reg2);
J = J+Reg;
% =========================================================================
big_del1 = zeros(hidden_layer_size,input_layer_size+1);
big_del2 = zeros(num_labels,hidden_layer_size+1);
for i=1:m,
	X1 = X(i,:);
	a1 = [ones(1,1),X1];
	z2 = a1 * Theta1';
	a2 = sigmoid(z2);
	a2 = [ones(1,1),a2];
	z3 = a2 * Theta2';
	a3 = sigmoid(z3);
	
	small_del3 = a3 - y_vec(i,:);
	%temp_sig_grad_z2 = [ones(size(z2),1),sigmoidGradient(z2)];
	%small_del2 = (small_del3 * Theta2) .* temp_sig_grad_z2;	
	
	sig_grad_z2 = sigmoidGradient(z2);
	sig_grad_z2 = [zeros(1,1),sig_grad_z2];
	small_del2 = (small_del3 * Theta2) .* sig_grad_z2;
	small_del2 = small_del2(2:end);
	big_del2 = big_del2 + (small_del3' * a2);
	big_del1 = big_del1 + (small_del2' * a1);
end
Theta1_grad = big_del1/m;
Theta2_grad = big_del2/m;
%==========================================================================================
Reg1_matrix = (lambda/m) * Theta1;
Reg2_matrix = (lambda/m) * Theta2;

Reg1_matrix_cut = Reg1_matrix(:,2:end);
Reg2_matrix_cut = Reg2_matrix(:,2:end);
Reg1_matrix = [zeros(size(Reg1_matrix_cut,1),1),Reg1_matrix_cut];
Reg2_matrix = [zeros(size(Reg2_matrix_cut,1),1),Reg2_matrix_cut];

Theta1_grad = Theta1_grad + Reg1_matrix;
Theta2_grad = Theta2_grad + Reg2_matrix;
%==========================================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
