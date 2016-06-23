function [ACC_avg] = Fun_Classification_DiffSDB(IMGFEATURE,LABELS,T)
gnum = 60;
Clsnum = 2;
AElayersize = [16 8];
AElayernum = length(AElayersize);
sparsityParam = 0.1;   % desired average activation of the hidden units
lambda = 5e-4;         % weight decay parameter
beta = 5;              % weight of sparsity penalty term
Ptrain = 0.5;
rsizenum = gnum/Clsnum;

if iscell(IMGFEATURE)
    for i = 1:3
        tnum = size(IMGFEATURE{i},2);
        inputSize = size(IMGFEATURE{1},1);
        if Clsnum == 2
            % Two class
            tmp = LABELS{i};
            tmp(find(tmp>1)) = 2;
            LABELS{i} = tmp;
        end
        LABELS{i} = reshape(LABELS{i},[gnum tnum/gnum]);
        IMGFEATURE{i} = normalization_line_2(IMGFEATURE{i});
        IMGFEATURE{i} = reshape(IMGFEATURE{i},[inputSize,gnum,tnum/gnum]);
    end
    totalnum = size(IMGFEATURE{1},3);
    trainnum = floor(totalnum * Ptrain);
    classnum = length(unique(LABELS{1}));
else
    tnum = size(IMGFEATURE,2);
    inputSize = size(IMGFEATURE,1);
    if Clsnum == 2
        % Two class
        LABELS(find(LABELS>1)) = 2;
    end
    LABELS = reshape(LABELS,[gnum tnum/gnum]);
    IMGFEATURE = normalization_line_2(IMGFEATURE);
    IMGFEATURE = reshape(IMGFEATURE,[inputSize,gnum,tnum/gnum]);
    
    totalnum = size(IMGFEATURE,3);
    trainnum = floor(totalnum * Ptrain);
    classnum = length(unique(LABELS));
end

CFmat = zeros(classnum,classnum);
ACC = [];
ACC_avg = [];
proball = [];
TestLabelAll = [];

for k = 1:T
    disp(k);
    %% Training and testing data
    nn = randperm(totalnum);
    n1 = nn(1:trainnum);
    n2 = nn(trainnum+1:end);
    if iscell(IMGFEATURE)
        TrainLabels = [];
        TrainImg = [];
        TestLabels = [];
        TestImg = [];
        for i = 1:3
            % Train
            LABELS_t = LABELS{i};
            IMGFEATURE_t = IMGFEATURE{i};
            TrainLabels_t = LABELS_t(:,n1);
            TrainImg_t = IMGFEATURE_t(:,:,n1);
            TrainLabels_t = TrainLabels_t(:);
            TrainImg_t = reshape(TrainImg_t,[inputSize,size(TrainImg_t,2)*size(TrainImg_t,3)]);
            % Test
            TestLabels_t = LABELS_t(:,n2);
            TestImg_t = IMGFEATURE_t(:,:,n2);
            TestLabels_t = TestLabels_t(:);
            TestImg_t = reshape(TestImg_t,[inputSize,size(TestImg_t,2)*size(TestImg_t,3)]);
            
            TrainLabels = [TrainLabels;TrainLabels_t];
            TrainImg = [TrainImg TrainImg_t];
            TestLabels = [TestLabels;TestLabels_t];
            TestImg = [TestImg TestImg_t];
        end
    else
        % Train
        TrainLabels = LABELS(:,n1);
        TrainImg = IMGFEATURE(:,:,n1);
        TrainLabels = TrainLabels(:);
        TrainImg = reshape(TrainImg,[inputSize,size(TrainImg,2)*size(TrainImg,3)]);
        % Test
        TestLabels = LABELS(:,n2);
        TestImg = IMGFEATURE(:,:,n2);
        TestLabels = TestLabels(:);
        TestImg = reshape(TestImg,[inputSize,size(TestImg,2)*size(TestImg,3)]);
    end
    
    %% Whiting train
%     meanSSCA = mean(TrainImg, 2);
%     TrainImg = bsxfun(@minus, TrainImg, meanSSCA);
%     sigma = TrainImg * TrainImg' / size(TrainImg,2);
%     [u, s, v] = svd(sigma);
%     ZCAWhite = u * diag(1 ./ sqrt(diag(s) + 0.1)) * u';
%     TrainImg = ZCAWhite * TrainImg;
    
    %% Whiting test
%     TestImg = bsxfun(@minus, TestImg, meanSSCA);
%     TestImg = ZCAWhite * TestImg;
    
    %% STEP 2: Train the sparse autoencoder
    %  Use minFunc to minimize the function
    options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
    % function. Generally, for minFunc to work, you
    % need a function pointer with two outputs: the
    % function value and the gradient. In our problem,
    % sparseAutoencoderCost.m satisfies this.
    options.maxIter = 400;	  % Maximum number of iterations of L-BFGS to run
    options.display = 'off';
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
    options.display = 'off';
    
    [stackedAEOptTheta, cost] = minFunc( @(p) stackedAECost(p, inputSize, AElayersize(i), ...
        classnum, netconfig, lambda, ...
        TrainImg, TrainLabels), stackedAETheta, options);
    
    %%======================================================================
    %% STEP 6: Test
    pred_bf = stackedAEPredict(stackedAETheta, AElayersize(i), classnum, netconfig, TestImg);
    acc = mean(TestLabels == pred_bf');
    %         [pred,prob,AEoutput,finalf] = stackedAEPredict_plot(stackedAEOptTheta, AElayersize(i), classnum, netconfig, TestImg);
    %         plot_features(finalf,TestLabels);
    [pred,prob] = stackedAEPredict(stackedAEOptTheta, AElayersize(i), classnum, netconfig, TestImg);
    acc = mean(TestLabels == pred');
    ACC = [ACC;acc];
    
    prob = squeeze(mean(reshape(prob,[Clsnum rsizenum,length(pred)/rsizenum]),2));
    [~,pred] = max(prob);
    TestLabels = mean(reshape(TestLabels,[rsizenum,length(TestLabels)/rsizenum]));
    acc = mean(TestLabels' == pred');
    ACC_avg = [ACC_avg;acc];
    
%     cfmat = cfmatrix(TestLabels',pred');
%     cfmat = bsxfun(@rdivide, cfmat, sum(cfmat));
%     
%     CFmat = CFmat + cfmat;
    
    proball = [proball prob];
    TestLabelAll = [TestLabelAll;TestLabels];
    
end
ACC_avg = sort(ACC_avg);
ACC_avg = ACC_avg(T/2);













