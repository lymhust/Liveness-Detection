function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
    numClasses, netconfig, ...
    lambda, data, labels)

% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.

% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example.
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example

%% Show temporary results
% global timer;
% if (timer == 1 || timer == 15 || timer == 60 || timer == 125 || timer == 250 || timer == 400)
%     finalf = stackedAEPredict_simple(theta, hiddenSize, numClasses, netconfig, data);
%     plot_features(finalf,labels);title(sprintf('Iteration = %d',timer));
% end


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; % You need to compute this

% You might find these variables useful
M = size(data, 2);
groundTruth = full(sparse(labels, 1:M, 1));


%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.

%  Forward
AEoutput = cell(length(stack)+2,1);
input = data;
AEoutput{1}.z = data;
AEoutput{1}.a = data;

for d = 2:numel(stack)+1
    ztmp = bsxfun(@plus, stack{d-1}.w*input, stack{d-1}.b);
    atmp = sigmoid(ztmp);
    AEoutput{d}.z = ztmp;
    AEoutput{d}.a = atmp;
    input = atmp;
end
zend = softmaxTheta * input;
aend = exp(zend);
aend = bsxfun(@rdivide, aend, sum(aend));
d = d+1;
AEoutput{d}.z = zend;
AEoutput{d}.a = aend;

d = d-1;
softmaxThetaGrad = -(1. / M) * (groundTruth - aend) * AEoutput{d}.a'  + lambda * softmaxTheta;

% Backpropagation
% delta4 = -(softmaxTheta' * (groundTruth - aend)) .* softmaxGrad(zend);
delta4 = -(groundTruth - aend);
delta3 = (softmaxTheta' * delta4) .* sigmoidGrad(AEoutput{d}.z);
delta = delta3;

costterm = 0;

for d = length(stack):-1:1
    stackgrad{d}.w = (1. / M) * delta * AEoutput{d}.a' + lambda * stack{d}.w;
    stackgrad{d}.b = (1. / M) * sum(delta, 2);
    delta = (stack{d}.w' * delta) .* sigmoidGrad(AEoutput{d}.z);
    costterm = costterm + sum(sum(stack{d}.w .^2));
end

cost = -(1. / M) * sum(sum(groundTruth .* log(aend))) + (lambda / 2.) * ...
    sum(sum(softmaxTheta.^2)) + (lambda / 2.) * costterm;

% stackgrad{2}.w = (1. / M) * delta3 * a2' + lambda * stack{2}.w;
% stackgrad{2}.b = (1. / M) * sum(delta3, 2);
% stackgrad{1}.w = (1. / M) * delta2 * data' + lambda * stack{1}.w;
% stackgrad{1}.b = (1. / M) * sum(delta2, 2);





% -------------------------------------------------------------------------
%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

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
