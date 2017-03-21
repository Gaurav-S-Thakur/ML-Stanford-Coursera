data = load('ex2data2.txt');
X = data(:, [1, 2]); y = data(:, 3);
X = mapFeature(X(:,1), X(:,2));
theta = zeros(size(X,2),1);
m=length(y);
% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1
lambda = 100;


hxtheta = sigmoid(X*theta);
J1 = y' * log(hxtheta) + (1-y)' * (log(1 - hxtheta))
Reg_term = (lambda/(2*m))*sum(theta .^ 2)
J = -J1 / m;
J = J+Reg_term
