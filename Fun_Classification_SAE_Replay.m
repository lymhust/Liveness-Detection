function [prob,TestLabelAll,ACC_test,ACC_dev] = Fun_Classification_SAE_Replay(TestType,classnum,Method)

switch TestType
    case 21
        % Print
        gnum = 20;
        switch Method
            case 1
                str = {'Train_DoG_Cls4','Test_DoG_Cls4','Develop_DoG_Cls4'};
                [TrainImg,TrainLabels,TestImg,TestLabels,DevImg,DevLabels] = Fun_CreateFeaturesForReplay(str,2,classnum);
            case 2
                str = {'Train_LBP_Cls4','Test_LBP_Cls4','Develop_LBP_Cls4'};
                [TrainImg,TrainLabels,TestImg,TestLabels,DevImg,DevLabels] = Fun_CreateFeaturesForReplay(str,2,classnum);
            case 3
                str = {'Train_SBFD_256by256_B64_Cls4','Test_SBFD_256by256_B64_Cls4','Develop_SBFD_256by256_B64_Cls4'};
                [TrainImg,TrainLabels,TestImg,TestLabels,DevImg,DevLabels] = Fun_CreateFeaturesForReplay(str,2,classnum);
            case 4
                str = {'Train_ESBFD_S3_D27_B64_Cls4','Test_ESBFD_S3_D27_B64_Cls4','Develop_ESBFD_S3_D27_B64_Cls4'};
                [TrainImg,TrainLabels,TestImg,TestLabels,DevImg,DevLabels] = Fun_CreateFeaturesForReplay(str,2,classnum);
                gnum = 2;
        end
    case 22
        % Mobile
        gnum = 20;
        switch Method
            case 1
                str = {'Train_DoG_Cls4','Test_DoG_Cls4','Develop_DoG_Cls4'};
                [TrainImg,TrainLabels,TestImg,TestLabels,DevImg,DevLabels] = Fun_CreateFeaturesForReplay(str,3,classnum);
            case 2
                str = {'Train_LBP_Cls4','Test_LBP_Cls4','Develop_LBP_Cls4'};
                [TrainImg,TrainLabels,TestImg,TestLabels,DevImg,DevLabels] = Fun_CreateFeaturesForReplay(str,3,classnum);
            case 3
                str = {'Train_SBFD_256by256_B64_Cls4','Test_SBFD_256by256_B64_Cls4','Develop_SBFD_256by256_B64_Cls4'};
                [TrainImg,TrainLabels,TestImg,TestLabels,DevImg,DevLabels] = Fun_CreateFeaturesForReplay(str,3,classnum);
            case 4
                str = {'Train_ESBFD_S3_D27_B64_Cls4','Test_ESBFD_S3_D27_B64_Cls4','Develop_ESBFD_S3_D27_B64_Cls4'};
                [TrainImg,TrainLabels,TestImg,TestLabels,DevImg,DevLabels] = Fun_CreateFeaturesForReplay(str,3,classnum);
                gnum = 2;
        end
    case 23
        % Video
        gnum = 20;
        switch Method
            case 1
                str = {'Train_DoG_Cls4','Test_DoG_Cls4','Develop_DoG_Cls4'};
                [TrainImg,TrainLabels,TestImg,TestLabels,DevImg,DevLabels] = Fun_CreateFeaturesForReplay(str,4,classnum);
            case 2
                str = {'Train_LBP_Cls4','Test_LBP_Cls4','Develop_LBP_Cls4'};
                [TrainImg,TrainLabels,TestImg,TestLabels,DevImg,DevLabels] = Fun_CreateFeaturesForReplay(str,4,classnum);
            case 3
                str = {'Train_SBFD_256by256_B64_Cls4','Test_SBFD_256by256_B64_Cls4','Develop_SBFD_256by256_B64_Cls4'};
                [TrainImg,TrainLabels,TestImg,TestLabels,DevImg,DevLabels] = Fun_CreateFeaturesForReplay(str,4,classnum);
            case 4
                str = {'Train_ESBFD_S3_D27_B64_Cls4','Test_ESBFD_S3_D27_B64_Cls4','Develop_ESBFD_S3_D27_B64_Cls4'};
                [TrainImg,TrainLabels,TestImg,TestLabels,DevImg,DevLabels] = Fun_CreateFeaturesForReplay(str,4,classnum);
                gnum = 2;
        end
    case 3
        % Overall
        switch Method
            case 1
                if classnum == 2
                    str = {'Train_DoG_Cls2','Test_DoG_Cls2','Develop_DoG_Cls2'};
                    [TrainImg,TrainLabels,TestImg,TestLabels,DevImg,DevLabels] = Fun_CreateFeaturesForReplay(str,0,classnum);
                    gnum = 60;
                else
                    str = {'Train_DoG_Cls4','Test_DoG_Cls4','Develop_DoG_Cls4'};
                    [TrainImg,TrainLabels,TestImg,TestLabels,DevImg,DevLabels] = Fun_CreateFeaturesForReplay(str,0,classnum);
                    gnum = 40;
                end
            case 2
                if classnum == 2
                    str = {'Train_LBP_Cls2','Test_LBP_Cls2','Develop_LBP_Cls2'};
                    [TrainImg,TrainLabels,TestImg,TestLabels,DevImg,DevLabels] = Fun_CreateFeaturesForReplay(str,0,classnum);
                    gnum = 60;
                else
                    str = {'Train_LBP_Cls4','Test_LBP_Cls4','Develop_LBP_Cls4'};
                    [TrainImg,TrainLabels,TestImg,TestLabels,DevImg,DevLabels] = Fun_CreateFeaturesForReplay(str,0,classnum);
                    gnum = 40;
                end
            case 3
                if classnum == 2
                    str = {'Train_SBFD_256by256_B64_Cls2','Test_SBFD_256by256_B64_Cls2','Develop_SBFD_256by256_B64_Cls2'};
                    [TrainImg,TrainLabels,TestImg,TestLabels,DevImg,DevLabels] = Fun_CreateFeaturesForReplay(str,0,classnum);
                    gnum = 60;
                else
                    str = {'Train_SBFD_256by256_B64_Cls4','Test_SBFD_256by256_B64_Cls4','Develop_SBFD_256by256_B64_Cls4'};
                    [TrainImg,TrainLabels,TestImg,TestLabels,DevImg,DevLabels] = Fun_CreateFeaturesForReplay(str,0,classnum);
                    gnum = 40;
                end
            case 4
                if classnum == 2
                     str = {'Train_ESBFD_S3_D27_B64_Cls2','Test_ESBFD_S3_D27_B64_Cls2','Develop_ESBFD_S3_D27_B64_Cls2'};
                    [TrainImg,TrainLabels,TestImg,TestLabels,DevImg,DevLabels] = Fun_CreateFeaturesForReplay(str,0,classnum);
                    gnum = 6;
                else
                     str = {'Train_ESBFD_S3_D27_B64_Cls4','Test_ESBFD_S3_D27_B64_Cls4','Develop_ESBFD_S3_D27_B64_Cls4'};
                    [TrainImg,TrainLabels,TestImg,TestLabels,DevImg,DevLabels] = Fun_CreateFeaturesForReplay(str,0,classnum);
                    gnum = 4;
                end
                
        end
end


AElayersize = [16 8];
AElayernum = length(AElayersize);
sparsityParam = 0.1;   % desired average activation of the hidden units
lambda = 5e-5;         % weight decay parameter
beta = 5;              % weight of sparsity penalty term
rsizenum = gnum/classnum;

ACC_test = [];
ACC_dev = [];
proball = [];
TestLabelAll = [];

%% Normalization
TrainImg = normalization_line_2(TrainImg);
TestImg = normalization_line_2(TestImg);
DevImg = normalization_line_2(DevImg);
inputSize = size(TrainImg,1);

%% Whiting train
meanSSCA = mean(TrainImg, 2);
TrainImg = bsxfun(@minus, TrainImg, meanSSCA);
sigma = TrainImg * TrainImg' / size(TrainImg,2);
[u, s, v] = svd(sigma);
ZCAWhite = u * diag(1 ./ sqrt(diag(s) + 0.1)) * u';
TrainImg = ZCAWhite * TrainImg;

%% Whiting test
TestImg = bsxfun(@minus, TestImg, meanSSCA);
TestImg = ZCAWhite * TestImg;

%% Whiting develop
DevImg = bsxfun(@minus, DevImg, meanSSCA);
DevImg = ZCAWhite * DevImg;

%% STEP 2: Train the sparse autoencoder
%  Use minFunc to minimize the function
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
% function. Generally, for minFunc to work, you
% need a function pointer with two outputs: the
% function value and the gradient. In our problem,
% sparseAutoencoderCost.m satisfies this.
options.maxIter = 400;	  % Maximum number of iterations of L-BFGS to run
options.display = 'on';
options.Corr = 10;

saeOptTheta = cell(1,AElayernum);
saeFeatures = cell(1,AElayernum);
stack = cell(AElayernum,1);
decoderW = cell(AElayernum,1);

for i = 1:AElayernum
    if (1 == i)
        inputsize = inputSize;
        outputsize = AElayersize(i);
        input = TrainImg;
    else
        inputsize = AElayersize(i-1);
        outputsize = AElayersize(i);
        input = saeFeatures{i-1};
    end
    
    %  Randomly initialize the parameters
    saeTheta = initializeParameters(outputsize, inputsize);
    
    [saeOptThetatmp, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
        inputsize, outputsize, ...
        lambda, sparsityParam, ...
        beta, input), ...
        saeTheta, options);
    
    [saeFeaturestmp] = feedForwardAutoencoder(saeOptThetatmp, outputsize, ...
        inputsize, input);
    
    saeOptTheta{i} = saeOptThetatmp;
    saeFeatures{i} = saeFeaturestmp;
    
    stack{i}.w = reshape(saeOptThetatmp(1:outputsize*inputsize),outputsize, inputsize);
    stack{i}.b = saeOptThetatmp(2*outputsize*inputsize+1:2*outputsize*inputsize+outputsize);
    
    decoderW{i}.w = reshape(saeOptThetatmp(outputsize*inputsize+1:2*outputsize*inputsize), inputsize, outputsize);
    decoderW{i}.b = saeOptThetatmp(2*outputsize*inputsize+outputsize+1:end);
    
end
%%======================================================================
%% STEP 3: Train the softmax classifier
%  Randomly initialize the parameters
options.maxIter = 800;
softmaxModel = softmaxTrain(AElayersize(i), classnum, lambda, ...
    saeFeatures{i}, TrainLabels, options);

saeSoftmaxOptTheta = softmaxModel.optTheta(:);
%%======================================================================
%% STEP 5: Finetune softmax model
% Initialize the parameters for the deep model
[stackparams, netconfig] = stack2params(stack);
stackedAETheta = [saeSoftmaxOptTheta ; stackparams ];

DEBUG = false;
if DEBUG
    lambda = 1e-4;
    TrainImg = TrainImg(:,1:10);
    TrainLabels = TrainLabels(1:10);
    %     netconfig.layersizes = {};
    %     netconfig.layersizes = [netconfig.layersizes;64];
    %     netconfig.layersizes = [netconfig.layersizes;64];
    stackedAETheta = stackedAETheta;
    
    [cost, grad] = stackedAECost(stackedAETheta, inputSize, AElayersize(i), classnum, netconfig, lambda, TrainImg, TrainLabels);
    
    numGrad = computeNumericalGradient( @(x) stackedAECost(x, inputSize, AElayersize(i), classnum, netconfig, lambda, TrainImg, TrainLabels), stackedAETheta);
    
    % Use this to visually compare the gradients side by side
    disp([numGrad grad]);
    
    % Compare numerically computed gradients with those computed analytically
    diff = norm(numGrad-grad)/norm(numGrad+grad);
    disp(diff);
    % The difference should be small.
    % In our implementation, these values are usually less than 1e-7.
    
    % When your gradients are correct, congratulations!
end

%    lambda = 1e-4;
options.Method = 'lbfgs';
options.maxIter = 400;
options.display = 'on';

[stackedAEOptTheta, cost] = minFunc( @(p) stackedAECost(p, inputSize, AElayersize(i), ...
    classnum, netconfig, lambda, ...
    TrainImg, TrainLabels), stackedAETheta, options);

%%======================================================================
%% STEP 6: Test
% Before fine tuning
pred_bf = stackedAEPredict(stackedAETheta, AElayersize(i), classnum, netconfig, TestImg);
acc = mean(TestLabels == pred_bf');
% After fine tuning
[pred_test,prob_test] = stackedAEPredict(stackedAEOptTheta, AElayersize(i), classnum, netconfig, TestImg);
[pred_dev,prob_dev] = stackedAEPredict(stackedAEOptTheta, AElayersize(i), classnum, netconfig, DevImg);
acc_test = mean(TestLabels == pred_test');
acc_dev = mean(DevLabels == pred_dev');
disp(sprintf('Testbf:%f,Testaf:%f,Devaf;%f',acc,acc_test,acc_dev));

if (Method == 4 && classnum == 4) || (Method == 4 && TestType == 21) || (Method == 4 && TestType == 22) || (Method == 4 && TestType == 23) 
    ACC_test = [ACC_test;acc_test];
    ACC_dev = [ACC_dev;acc_dev];
    proball = [proball prob_test];
    TestLabelAll = [TestLabelAll;TestLabels'];
else
    prob_test = squeeze(mean(reshape(prob_test,[classnum rsizenum,length(pred_test)/rsizenum]),2));
    [~,pred_test] = max(prob_test);
    prob_dev = squeeze(mean(reshape(prob_dev,[classnum rsizenum,length(pred_dev)/rsizenum]),2));
    [~,pred_dev] = max(prob_dev);
    
    TestLabels = mean(reshape(TestLabels,[rsizenum,length(TestLabels)/rsizenum]));
    DevLabels = mean(reshape(DevLabels,[rsizenum,length(DevLabels)/rsizenum]));
    
    acc_test = mean(TestLabels' == pred_test');
    acc_dev = mean(DevLabels' == pred_dev');
    disp(sprintf('Average,Testaf:%f,Devaf;%f',acc_test,acc_dev));
    
    ACC_test = [ACC_test;acc_test];
    ACC_dev = [ACC_dev;acc_dev];
    proball = [proball prob_test];
    TestLabelAll = [TestLabelAll;TestLabels'];
end

cfmat = cfmatrix(TestLabels',pred_test');
cfmat = bsxfun(@rdivide, cfmat, sum(cfmat));

prob = proball(1,:)';

%% Plotting CFmat
% names = ['Real    ';'Non-real'];
% draw_cm(cfmat,names,2);
names = ['Real  ';'Print ';'Mobile';'Video '];
draw_cm(cfmat,names,4);













