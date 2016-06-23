function [ cost, grad ] = stackedAECost_autoencoder(theta, netconfig, lambda, TrainFeature, TPPG_train)
                                        
%% Unroll softmaxTheta parameter
% Extract out the "stack"
stack = params2stack(theta, netconfig);

% You will need to compute the following gradients
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; % You need to compute this
M = size(TrainFeature, 2);
%  Forward
AEoutput = cell(length(stack)+1,1);
input = TrainFeature;
AEoutput{1}.z = input;
AEoutput{1}.a = input;

for d = 2:numel(stack)+1  
    ztmp = bsxfun(@plus, stack{d-1}.w*input, stack{d-1}.b);
    atmp = sigmoid(ztmp);
    AEoutput{d}.z = ztmp;
    AEoutput{d}.a = atmp;
    input = atmp;
end

zend = AEoutput{d}.z;
aend = AEoutput{d}.a ;

% Backpropagation
delta =  -(TPPG_train - aend).* sigmoidGrad(zend);

costterm = 0;

for d = length(stack):-1:1
    stackgrad{d}.w = (1. / M) * delta * AEoutput{d}.a' + lambda * stack{d}.w;
    stackgrad{d}.b = (1. / M) * sum(delta, 2);    
    delta = (stack{d}.w' * delta) .* sigmoidGrad(AEoutput{d}.z);   
    costterm = costterm + sum(sum(stack{d}.w .^2));
end

cost = (1. / M) * sum((1. / 2) * sum((aend - TPPG_train).^2)) + (lambda / 2.) * costterm;

grad = stack2params(stackgrad);

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end

function grad = sigmoidGrad(x)
    e_x = exp(-x);
    grad = e_x ./ ((1 + e_x).^2); 
end

function grad = softmaxGrad(x)
    e_x = exp(-x);
    grad = e_x ./ (1 + (1-e_x).*e_x ).^2;
end
