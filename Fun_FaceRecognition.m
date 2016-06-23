function [prob,TestLabelAll,ACC] = Fun_FaceRecognition(TestType,Method,T)

AElayersize = [16 8];
AElayernum = length(AElayersize);
sparsityParam = 0.1;   % desired average activation of the hidden units
lambda = 5e-4;         % weight decay parameter
beta = 5;              % weight of sparsity penalty term

CFmat = zeros(50,50);
ACC = [];
proball = [];
TestLabelAll = [];

for k = 1:T
    switch TestType
        case 11
            % Low Quality
            switch Method
                case 1
                    load LOW_FaceFeatures_DoG_4class;
                case 2
                    load LOW_FaceFeatures_LBP_4class;
                case 3
                    load LOW_FaceFeaturesNew_256by256_64_mean_4class;
            end
        case 12
            % Normal Quality
            switch Method
                case 1
                    load MID_FaceFeatures_DoG_4class;
                case 2
                    load MID_FaceFeatures_LBP_4class;
                case 3
                    load MID_FaceFeaturesNew_256by256_64_mean_4class;
            end
        case 13
            % High Quality
            switch Method
                case 1
                    load HR_FaceFeatures_DoG_4class;
                case 2
                    load HR_FaceFeatures_LBP_4class;
                case 3
                    load HR_FaceFeaturesNew_256by256_64_mean_4class;
            end
        case 3
            % Overall
            switch Method
                case 1
                    str = {'LOW_FaceFeatures_DoG_4class','MID_FaceFeatures_DoG_4class','HR_FaceFeatures_DoG_4class'};
                    [IMGFEATURE, LABELS] = Fun_CombineFeature(str,0);
                case 2
                    str = {'LOW_FaceFeatures_LBP_4class','MID_FaceFeatures_LBP_4class','HR_FaceFeatures_LBP_4class'};
                    [IMGFEATURE, LABELS] = Fun_CombineFeature(str,0);
                case 3
                    str = {'LOW_FaceFeaturesNew_256by256_64_mean_4class','MID_FaceFeaturesNew_256by256_64_mean_4class','HR_FaceFeaturesNew_256by256_64_mean_4class'};
                    [IMGFEATURE, LABELS] = Fun_CombineFeature(str,0);
            end
    end
    
    %% Training and testing data
    if iscell(IMGFEATURE)
        TrainLabels = [];
        TrainImg = [];
        TestLabels = [];
        TestImg = [];
        nn = randperm(10);
        for pp = 1:3
            IMGFEATURE_t = IMGFEATURE{pp};
            LABELS_t = LABELS{pp};
            IMGFEATURE_t = IMGFEATURE_t(:,find(LABELS_t == 1));
            IMGFEATURE_t = reshape(IMGFEATURE_t,[size(IMGFEATURE_t,1),10,size(IMGFEATURE_t,2)/10]);
            % Half training and half testing
            TrainImg_t = IMGFEATURE_t(:,nn(1:5),:);
            TestImg_t = IMGFEATURE_t(:,nn(6:end),:);
            TrainLabels_t = 1:size(TrainImg_t,3);
            TrainLabels_t = repmat(TrainLabels_t,[5 1]);
            TrainLabels_t = TrainLabels_t(:);
            TestLabels_t = TrainLabels_t;
            TrainImg_t = reshape(TrainImg_t,[size(TrainImg_t,1) size(TrainImg_t,2)*size(TrainImg_t,3)]);
            TestImg_t = reshape(TestImg_t,[size(TestImg_t,1) size(TestImg_t,2)*size(TestImg_t,3)]);
            
            TrainLabels = [TrainLabels;TrainLabels_t];
            TrainImg = [TrainImg TrainImg_t];
            TestLabels = [TestLabels;TestLabels_t];
            TestImg = [TestImg TestImg_t];
        end
        inputSize = size(TrainImg,1);
        classnum = length(unique(TrainLabels));
    else
        IMGFEATURE = IMGFEATURE(:,find(LABELS == 1));
        IMGFEATURE = reshape(IMGFEATURE,[size(IMGFEATURE,1),10,size(IMGFEATURE,2)/10]);
        % Half training and half testing
        nn = randperm(10);
        TrainImg = IMGFEATURE(:,nn(1:5),:);
        TestImg = IMGFEATURE(:,nn(6:end),:);
        TrainLabels = 1:size(TrainImg,3);
        TrainLabels = repmat(TrainLabels,[5 1]);
        TrainLabels = TrainLabels(:);
        TestLabels = TrainLabels;
        TrainImg = reshape(TrainImg,[size(TrainImg,1) size(TrainImg,2)*size(TrainImg,3)]);
        TestImg = reshape(TestImg,[size(TestImg,1) size(TestImg,2)*size(TestImg,3)]);
        inputSize = size(TrainImg,1);
        classnum = length(unique(TrainLabels));
    end
    
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
    pred_bf = stackedAEPredict(stackedAETheta, AElayersize(i), classnum, netconfig, TestImg);
    acc = mean(TestLabels == pred_bf')
    %         [pred,prob,AEoutput,finalf] = stackedAEPredict_plot(stackedAEOptTheta, AElayersize(i), classnum, netconfig, TestImg);
    %         plot_features(finalf,TestLabels);
    [pred,prob] = stackedAEPredict(stackedAEOptTheta, AElayersize(i), classnum, netconfig, TestImg);
    acc = mean(TestLabels == pred')
    ACC = [ACC;acc];
    
    cfmat = cfmatrix(TestLabels',pred');
    cfmat = bsxfun(@rdivide, cfmat, sum(cfmat));
    
    CFmat = CFmat + cfmat;
    
    proball = [proball prob];
    TestLabelAll = [TestLabelAll;TestLabels];
    
end

prob = proball(1,:)';
TestLabelAll = TestLabelAll';

%% Plotting CFmat
CFmat = CFmat./T;

% CFmatnew(1:46,1:46) = CFmat;
% ind = randperm(46);
% CFmatnew(47:end,47:end) = CFmat(ind(1:4),ind(1:4));
imagesc(CFmat);
colormap gray;













