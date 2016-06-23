function [cost, grad] = logisticCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);
numCases = size(data, 2);
groundTruth = labels';
cost = 0;
thetagrad = zeros(numClasses, inputSize);
epsl = 0.01;

M = theta * data;
p = 1./(1+exp(-M));
cost = -(1 / numCases) * sum( groundTruth .* log(p + epsl) + (1 - groundTruth).* log(1 - p + epsl) )...
       + (lambda / 2.) * sum(sum(theta.^2));
thetagrad = -(1 / numCases) * (groundTruth - p) * data' + lambda * theta;
grad = [thetagrad(:)];


% jVal=-sum(log(hypothesis+0.01).*y + (1-y).*log(1-hypothesis+0.01))/m;  
% gradient(1)=sum(hypothesis-y)/m;   %reflect to theta1  
% gradient(2)=sum((hypothesis-y).*x)/m;    %reflect to theta 2  

end

