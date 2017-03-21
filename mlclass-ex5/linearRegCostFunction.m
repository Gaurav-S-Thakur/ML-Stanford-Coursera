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

hxtheta = X * theta;
Error = hxtheta - y;

J = Error .* Error ;
J = (1/m)*(sum(J))*0.5;
R1 = theta .* theta;
R1 = R1(2:end);
Reg = (lambda/(2*m))*sum(R1);
J += Reg;

for i=1:size(theta,1),
	reg_grad = (lambda/m)*theta(i,1);
	if ( i==1 ),
		reg_grad=0;
	endif
	grad(i,1) = (1/m)*sum(Error .* X(:,i));
	grad(i,1) += reg_grad;
end










% =========================================================================

grad = grad(:);

end
