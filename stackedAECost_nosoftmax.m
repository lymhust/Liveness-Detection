function [ cost, grad ] = stackedAECost_nosoftmax(theta, netconfig, ...
    lambda, data, labels, dropout)

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

% Extract out the "stack"
stack = params2stack(theta(1:end), netconfig);

% You will need to compute the following gradients
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; % You need to compute this

% You might find these variables useful
M = size(data, 2);
groundTruth = labels;% groundTruth = full(sparse(labels, 1:M, 1));


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
AEoutput = cell(length(stack)+1,1);
input = data;
AEoutput{1}.z = data;
AEoutput{1}.a = data;
AEoutput{1}.mask = ones(size(data));

for d = 2:numel(stack)+1
    ztmp = bsxfun(@plus, stack{d-1}.w*input, stack{d-1}.b);
    atmp = sigmoid(ztmp);
    if (d < numel(stack)+1)
        mask = rand(size(ztmp))>dropout;
        atmp = atmp .* mask;
        AEoutput{d}.mask = mask;
    end
    AEoutput{d}.z = ztmp;
    AEoutput{d}.a = atmp;
    input = atmp;
end

% Backpropagation
error = -(groundTruth - atmp);
delta = error .* (atmp .* (1 - atmp)); % output delta

costterm = 0;

for d = length(stack):-1:1
    stackgrad{d}.w = (1. / M) * delta * AEoutput{d}.a' + lambda * stack{d}.w;
    stackgrad{d}.b = (1. / M) * sum(delta, 2);
    delta = (stack{d}.w' * delta) .* sigmoidGrad(AEoutput{d}.z) .* AEoutput{d}.mask;
    costterm = costterm + sum(sum(stack{d}.w .^2));
end

cost = 1/2 * sum(error(:).^2) / M + (lambda / 2.) * costterm;

%% Roll gradient vector
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

