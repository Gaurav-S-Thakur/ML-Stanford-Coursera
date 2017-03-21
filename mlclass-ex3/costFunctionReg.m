function [J, grad] = costFunctionReg(theta, X, y, lambda)
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




hxtheta = sigmoid(X*theta);
J1 = y' * log(hxtheta) + (1-y)' * (log(1 - hxtheta));
temp_theta = theta(2:length(theta),1);
Reg_term = (lambda/(2*m))*sum(temp_theta .^ 2);
J = -J1 / m;
J = J+Reg_term;
Error = hxtheta - y;
Reg_matrix_term = lambda * theta;
%size(Reg_matrix_term)
%Reg_matrix_term(1)
Reg_matrix_term(1)=0;
grad = (1/m)*(X'*Error + Reg_matrix_term);

%for indices = 1:size(theta,1),
%	temp = X(:,indices);
%	grad(indices) = temp' * Error;
%	grad(indices) = grad(indices)/m;
%	if (indices >=2),
%		grad_reg = (lambda/m)*theta(indices,1);
%		grad(indices) = grad(indices)+grad_reg;
%	end
% =============================================================

end
