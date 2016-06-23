function [prob,TestLabelAll,ACC_avg] = Fun_Classification_SAE_CASIA(TestType,Clsnum,T,Method,Ptrain)

switch TestType
    case 11
        % Low Quality
        switch Method
            case 1
                if Clsnum == 2
                    load LOW_FaceFeatures_DoG_2class;
                    gnum = 60;
                else
                    load LOW_FaceFeatures_DoG_4class;
                    gnum = 40;
                end
            case 2
                if Clsnum == 2
                    load LOW_FaceFeatures_LBP_2class;
                    gnum = 60;
                else
                    load LOW_FaceFeatures_LBP_4class;
                    gnum = 40;
                end
            case 3
                if Clsnum == 2
                    load LOW_FaceFeaturesC;%LOW_FaceFeaturesNew_256by256_64_mean_2class;
                    gnum = 60;
                else
                    load LOW_FaceFeaturesC_4;%LOW_FaceFeaturesNew_256by256_64_mean_4class;
                    gnum = 40;
                end
            case 4
                if Clsnum == 2
                    load LOW_ESBFD_S4_D27_B128_Cls2;%LOW_ESBFD_S3_D27_B64_Cls2;
                    gnum = 6;
                else
                    load LOW_ESBFD_S3_D27_B64_Cls4;
                    gnum = 4;
                end
        end
    case 12
        % Normal Quality
        switch Method
            case 1
                if Clsnum == 2
                    load MID_FaceFeatures_DoG_2class;
                    gnum = 60;
                else
                    load MID_FaceFeatures_DoG_4class;
                    gnum = 40;
                end
            case 2
                if Clsnum == 2
                    load MID_FaceFeatures_LBP_2class;
                    gnum = 60;
                else
                    load MID_FaceFeatures_LBP_4class;
                    gnum = 40;
                end
            case 3
                if Clsnum == 2
                    load MID_FaceFeaturesNew_256by256_64_mean_2class;
                    gnum = 60;
                else
                    load MID_FaceFeaturesNew_256by256_64_mean_4class;
                    gnum = 40;
                end
            case 4
                if Clsnum == 2
                    load MID_ESBFD_S3_D27_B64_Cls2;
                    gnum = 6;
                else
                    load MID_ESBFD_S3_D27_B64_Cls4;
                    gnum = 4;
                end
        end
    case 13
        % High Quality
        switch Method
            case 1
                if Clsnum == 2
                    load HR_FaceFeatures_DoG_2class;
                    gnum = 60;
                else
                    load HR_FaceFeatures_DoG_4class;
                    gnum = 40;
                end
            case 2
                if Clsnum == 2
                    load HR_FaceFeatures_LBP_2class;
                    gnum = 60;
                else
                    load HR_FaceFeatures_LBP_4class;
                    gnum = 40;
                end
            case 3
                if Clsnum == 2
%                     str = {'HR_FaceFeaturesNew_256by256_64_mean_2class','HR_FaceFeatures_LBP_2class'};
%                     [IMGFEATURE, LABELS] = Fun_CombineFeature(str,0);
                    load HR_FaceFeaturesC;%HR_FaceFeaturesNew_256by256_64_mean_2class;%HR_FaceFeaturesC;%
                    gnum = 60;
                else
                    load HR_FaceFeaturesNew_256by256_64_mean_4class;
                    gnum = 40;
                end
            case 4
                if Clsnum == 2
                    load HR_ESBFD_S3_D27_B64_Cls2;
                    gnum = 6;
                else
                    load HR_ESBFD_S3_D27_B64_Cls4;
                    gnum = 4;
                end
        end
    case 21
        % Warped Photo
        gnum = 20;
        switch Method
            case 1
                str = {'LOW_FaceFeatures_DoG_4class','MID_FaceFeatures_DoG_4class','HR_FaceFeatures_DoG_4class'};
                [IMGFEATURE, LABELS] = Fun_CombineFeature(str,2);
            case 2
                str = {'LOW_FaceFeatures_LBP_4class','MID_FaceFeatures_LBP_4class','HR_FaceFeatures_LBP_4class'};
                [IMGFEATURE, LABELS] = Fun_CombineFeature(str,2);
            case 3
                str = {'LOW_FaceFeaturesNew_256by256_64_mean_4class','MID_FaceFeaturesNew_256by256_64_mean_4class','HR_FaceFeaturesNew_256by256_64_mean_4class'};
                [IMGFEATURE, LABELS] = Fun_CombineFeature(str,2);
            case 4
                str = {'LOW_ESBFD_S3_D27_B64_Cls4','MID_ESBFD_S3_D27_B64_Cls4','HR_ESBFD_S3_D27_B64_Cls4'};
                [IMGFEATURE, LABELS] = Fun_CombineFeature(str,2);
                gnum = 2;
        end
    case 22
        % Cut Photo
        gnum = 20;
        switch Method
            case 1
                str = {'LOW_FaceFeatures_DoG_4class','MID_FaceFeatures_DoG_4class','HR_FaceFeatures_DoG_4class'};
                [IMGFEATURE, LABELS] = Fun_CombineFeature(str,3);
            case 2
                str = {'LOW_FaceFeatures_LBP_4class','MID_FaceFeatures_LBP_4class','HR_FaceFeatures_LBP_4class'};
                [IMGFEATURE, LABELS] = Fun_CombineFeature(str,3);
            case 3
                str = {'LOW_FaceFeaturesNew_256by256_64_mean_4class','MID_FaceFeaturesNew_256by256_64_mean_4class','HR_FaceFeaturesNew_256by256_64_mean_4class'};
                [IMGFEATURE, LABELS] = Fun_CombineFeature(str,3);
            case 4
                str = {'LOW_ESBFD_S3_D27_B64_Cls4','MID_ESBFD_S3_D27_B64_Cls4','HR_ESBFD_S3_D27_B64_Cls4'};
                [IMGFEATURE, LABELS] = Fun_CombineFeature(str,3);
                gnum = 2;
        end
    case 23
        % Video
        gnum = 20;
        switch Method
            case 1
                str = {'LOW_FaceFeatures_DoG_4class','MID_FaceFeatures_DoG_4class','HR_FaceFeatures_DoG_4class'};
                [IMGFEATURE, LABELS] = Fun_CombineFeature(str,4);
            case 2
                str = {'LOW_FaceFeatures_LBP_4class','MID_FaceFeatures_LBP_4class','HR_FaceFeatures_LBP_4class'};
                [IMGFEATURE, LABELS] = Fun_CombineFeature(str,4);
            case 3
                str = {'LOW_FaceFeaturesNew_256by256_64_mean_4class','MID_FaceFeaturesNew_256by256_64_mean_4class','HR_FaceFeaturesNew_256by256_64_mean_4class'};
                [IMGFEATURE, LABELS] = Fun_CombineFeature(str,4);
            case 4
                str = {'LOW_ESBFD_S3_D27_B64_Cls4','MID_ESBFD_S3_D27_B64_Cls4','HR_ESBFD_S3_D27_B64_Cls4'};
                [IMGFEATURE, LABELS] = Fun_CombineFeature(str,4);
                gnum = 2;
        end
    case 3
        % Overall
        switch Method
            case 1
                if Clsnum == 2
                    str = {'LOW_FaceFeatures_DoG_2class','MID_FaceFeatures_DoG_2class','HR_FaceFeatures_DoG_2class'};
                    [IMGFEATURE, LABELS] = Fun_CombineFeature(str,0);
                    gnum = 60;
                else
                    str = {'LOW_FaceFeatures_DoG_4class','MID_FaceFeatures_DoG_4class','HR_FaceFeatures_DoG_4class'};
                    [IMGFEATURE, LABELS] = Fun_CombineFeature(str,0);
                    gnum = 40;
                end
            case 2
                if Clsnum == 2
                    str = {'LOW_FaceFeatures_LBP_2class','MID_FaceFeatures_LBP_2class','HR_FaceFeatures_LBP_2class'};
                    [IMGFEATURE, LABELS] = Fun_CombineFeature(str,0);
                    gnum = 60;
                else
                    str = {'LOW_FaceFeatures_LBP_4class','MID_FaceFeatures_LBP_4class','HR_FaceFeatures_LBP_4class'};
                    [IMGFEATURE, LABELS] = Fun_CombineFeature(str,0);
                    gnum = 40;
                end
            case 3
                if Clsnum == 2
                    str = {'LOW_FaceFeaturesNew_256by256_64_mean_2class','MID_FaceFeaturesNew_256by256_64_mean_2class','HR_FaceFeaturesNew_256by256_64_mean_2class'};
                    [IMGFEATURE, LABELS] = Fun_CombineFeature(str,0);
                    gnum = 60;
                else
                    str = {'LOW_FaceFeaturesNew_256by256_64_mean_4class','MID_FaceFeaturesNew_256by256_64_mean_4class','HR_FaceFeaturesNew_256by256_64_mean_4class'};
                    [IMGFEATURE, LABELS] = Fun_CombineFeature(str,0);
                    gnum = 40;
                end
            case 4
                if Clsnum == 2
                    str = {'LOW_ESBFD_S3_D27_B64_Cls2','MID_ESBFD_S3_D27_B64_Cls2','HR_ESBFD_S3_D27_B64_Cls2'};
                    [IMGFEATURE, LABELS] = Fun_CombineFeature(str,0);
                    gnum = 6;
                else
                    str = {'LOW_ESBFD_S3_D27_B64_Cls4','MID_ESBFD_S3_D27_B64_Cls4','HR_ESBFD_S3_D27_B64_Cls4'};
                    [IMGFEATURE, LABELS] = Fun_CombineFeature(str,0);
                    gnum = 4;
                end
                
        end
end

%% Combining LBP and SBFD
% IMGFEATURE{1} = normalization_line(IMGFEATURE{1});
% IMGFEATURE{2} = normalization_line(IMGFEATURE{2});
% IMGFEATURE = [IMGFEATURE{1};IMGFEATURE{2}];
% LABELS = LABELS{1};

AElayersize = [32 16];
AElayernum = length(AElayersize);
sparsityParam = 0.1;   % desired average activation of the hidden units
lambda = 8e-6;         % weight decay parameter
beta = 5;              % weight of sparsity penalty term
%Ptrain = 0.6;
rsizenum = gnum/Clsnum;
IMGFEATUREORI = IMGFEATURE;
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
        %IMGFEATURE{i} = normalization_line_2(IMGFEATURE{i});
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
    %IMGFEATURE = normalization_line_2(IMGFEATURE);
    IMGFEATURE = reshape(IMGFEATURE,[inputSize,gnum,tnum/gnum]);
    IMGFEATUREORI = reshape(IMGFEATUREORI,[inputSize,gnum,tnum/gnum]);
    
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
    %% Training and testing data
    nn = 1:50;%randperm(totalnum);
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
        
        TestOri = IMGFEATUREORI(:,:,n2);
        TestOri = reshape(TestOri,[inputSize,size(TestOri,2)*size(TestOri,3)]);
    end
    
    %% Add mean value
%     indx = find(TrainLabels == 1);
%     meanvalue = mean(TrainImg(:,indx),2);
%     meanval = repmat(meanvalue,[1 size(TrainImg,2)]);
%     TrainImg = [TrainImg;meanval];
%     meanval = repmat(meanvalue,[1 size(TestImg,2)]);
%     TestImg = [TestImg;meanval];
%     inputSize = inputSize * 2;
    
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
    
%indx=find(TrainLabels~=1);TrainLabels = TrainLabels(indx)-1;TrainImg = TrainImg(:,indx);classnum = 3;
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
    options.maxIter = 400;
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
    options.maxIter = 800;
    options.display = 'on';
    
    [stackedAEOptTheta, cost,~,flg] = minFunc( @(p) stackedAECost(p, inputSize, AElayersize(i), ...
        classnum, netconfig, lambda, ...
        TrainImg, TrainLabels), stackedAETheta, options);
    plot(flg.trace.fval);
    
    %%======================================================================
    %% STEP 6: Test
    if rsizenum == 30
        rsizenum = rsizenum/3;
    end
    pred_bf = stackedAEPredict(stackedAETheta, AElayersize(i), classnum, netconfig, TestImg);
    acc = mean(TestLabels == pred_bf')
    %         [pred,prob,AEoutput,finalf] = stackedAEPredict_plot(stackedAEOptTheta, AElayersize(i), classnum, netconfig, TestImg);
    %         plot_features(finalf,TestLabels);
    [pred,prob] = stackedAEPredict(stackedAEOptTheta, AElayersize(i), classnum, netconfig, TestImg);
    acc = mean(TestLabels == pred')
    ACC = [ACC;acc];
    
    if (Method == 4 && Clsnum == 4) || (Method == 4 && TestType == 21) || (Method == 4 && TestType == 22) || (Method == 4 && TestType == 23)
        ACC_avg = [ACC_avg;acc];
        proball = [proball prob];
        TestLabelAll = [TestLabelAll;TestLabels'];
    else
        prob = squeeze(mean(reshape(prob,[Clsnum rsizenum,length(pred)/rsizenum]),2));
        [~,pred] = max(prob);
        TestLabels = mean(reshape(TestLabels,[rsizenum,length(TestLabels)/rsizenum]));
        acc = mean(TestLabels' == pred')
        ACC_avg = [ACC_avg;acc];
        proball = [proball prob];
        TestLabelAll = [TestLabelAll;TestLabels'];
    end
    
    cfmat = cfmatrix(TestLabels',pred');
    cfmat = bsxfun(@rdivide, cfmat, sum(cfmat));
    
    CFmat = CFmat + cfmat;
     % inputSize = inputSize/2;
end
prob = proball(1,:)';

%% Plotting CFmat
%CFmat = CFmat./T;
% names = ['Real    ';'Non-real'];
% draw_cm(CFmat,names,2);
% names = ['Real  ';'Warped';'Cut   ';'Video '];
% draw_cm(CFmat,names,4);
% 












