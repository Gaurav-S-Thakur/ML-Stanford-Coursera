function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_set = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_set = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
Errors = zeros(size(C_set,2),size(sigma_set,2));

x1 = [1 2 1]; x2 = [0 4 -1];
for i=1:size(Errors,1),
	for j=1:size(Errors,2),
		model= svmTrain(X, y, C_set(1,i), @(x1, x2) gaussianKernel(x1, x2, sigma_set(1,j)));
		p = svmPredict(model,Xval);
		Errors(i,j)=mean(double(p~= yval));
	end
end

min_i = 1;
min_j=1;
min_Error = min(min(Errors,[],2));
for i=1:size(Errors,1),
	for j=1:size(Errors,2),
		if(min_Error == Errors(i,j)),
			min_i=i;
			min_j=j;
		endif
	end
end

min_Error

C = C_set(1,min_i);
sigma = sigma_set(1,min_j);

% =========================================================================

end
