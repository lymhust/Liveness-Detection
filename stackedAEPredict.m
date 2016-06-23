function [pred,prob,AEoutput,input] = stackedAEPredict(theta, hiddenSize, numClasses, netconfig, data)

% stackedAEPredict: Takes a trained theta and a test data set,
% and returns the predicted labels for each example.

% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example.

% Your code should produce the prediction matrix
% pred, where pred(i) is argmax_c P(y(c) | x(i)).

%% Unroll theta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start
%                from 1.
len = length(stack);
AEoutput = cell(len+2,1);
input = data;
AEoutput{1}.z = input;
AEoutput{1}.a = input;

for d = 2:numel(stack)+1
    ztmp = bsxfun(@plus, stack{d-1}.w*input, stack{d-1}.b);
    atmp = sigmoid(ztmp);
    AEoutput{d}.z = ztmp;
    AEoutput{d}.a = atmp;
    input = atmp; 
end

zend = softmaxTheta * input;
aend = exp(zend);
AEoutput{d+1}.z = zend;
AEoutput{d+1}.a = aend;

prob = bsxfun(@rdivide, aend, sum(aend));
[~,pred] = max(prob, [], 1);
end


% You might find this useful
function sigm = sigmoid(x)
sigm = 1 ./ (1 + exp(-x));
end
