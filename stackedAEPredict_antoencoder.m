function prePPG = stackedAEPredict_antoencoder(TestFeature, encoderAEOptTheta, encodernetconfig)

enW = params2stack(encoderAEOptTheta, encodernetconfig);

len = length(enW);
AEoutput = cell(len+1,1);
input = TestFeature;
AEoutput{1}.z = input;
AEoutput{1}.a = input;

for d = 2:len+1
    ztmp = bsxfun(@plus, enW{d-1}.w*input, enW{d-1}.b);
    atmp = sigmoid(ztmp);
    AEoutput{d}.z = ztmp;
    AEoutput{d}.a = atmp;
    input = atmp;
end

prePPG = input;

end

function sigm = sigmoid(x)
sigm = 1 ./ (1 + exp(-x));
end
