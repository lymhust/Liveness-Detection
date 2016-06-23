function [pred,AEoutput,input] = stackedAEPredict_nosoftmax(theta, netconfig, data, dropout)

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

% Extract out the "stack"
stack = params2stack(theta(1:end), netconfig);

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
    if (d < numel(stack)+1)
        atmp = atmp .* (1 - dropout);
    end
    AEoutput{d}.z = ztmp;
    AEoutput{d}.a = atmp;
    input = atmp;  
end
pred = atmp;

% Display
aaa = stack{1}.w';
dim = floor(sqrt(size(aaa,1)));
dim = dim^2;
display_network(aaa(1:dim,:));


end


% You might find this useful
function sigm = sigmoid(x)
sigm = 1 ./ (1 + exp(-x));
end
