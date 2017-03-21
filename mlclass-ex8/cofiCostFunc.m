function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%
%for i=1:size(R,1),
%	for j=1:size(R,2),
%		if(R(i,j)),
%			cost_ij = X(i,:)*Theta(j,:)';
%			cost_ij -= Y(i,j);
%			cost_ij = cost_ij^2;
%			J+=(cost_ij/2);
%		endif
%	end
%end

Cost_net = X*Theta';
Cost_net -= Y;
Cost_net = Cost_net .* Cost_net;
J = (sum(sum(Cost_net.*R)))/2;
Reg1 = (lambda/2)*(sum(sum(Theta .* Theta)));
Reg2 = (lambda/2)*(sum(sum(X .* X)));
J += Reg1+Reg2;

for i=1:size(X,1),
	x1=X(i,:);
	diff = x1 * Theta';
	diff -= Y(i,:);
	diff = diff .* R(i,:);
	X_grad(i,:) = diff * Theta;
	X_grad(i,:) += lambda*X(i,:);
end

for j=1:size(Theta,1),
	theta=Theta(j,:);
	diff = X*theta';
	diff -= Y(:,j);
	diff = diff .* R(:,j);
	Theta_grad(j,:) =  diff' * X;
	Theta_grad(j,:) += lambda*Theta(j,:);
end


% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
