function [y_vec] = nnCostFunction(y,num_labels)
	y_vec = zeros(size(y,1),num_labels);
	for i=1:size(y,1),
		y_vec(i,y)=1;
	end
end