function [proball,TestLabelAll,ACC_test,ACC_dev] = Fun_Classification_SVM_Replay(TestType,classnum,Method)

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

%% STEP 6: Test
svmStruct = svmtrain(TrainLabels,TrainImg');
[pred_test, ~, prob_test] = svmpredict(TestLabels, TestImg', svmStruct);
[pred_dev, ~, prob_dev] = svmpredict(DevLabels, DevImg', svmStruct);

acc_test = mean(TestLabels == pred_test);
acc_dev = mean(DevLabels == pred_dev);
disp(sprintf('Testaf:%f,Devaf;%f',acc_test,acc_dev));

if (Method == 4 && classnum == 4) || (Method == 4 && TestType == 21) || (Method == 4 && TestType == 22) || (Method == 4 && TestType == 23)
    ACC_test = [ACC_test;acc_test];
    ACC_dev = [ACC_dev;acc_dev];
    proball = [proball prob_test];
    TestLabelAll = [TestLabelAll;TestLabels'];
else
    prob_test = mean(reshape(prob_test,[rsizenum,length(prob_test)/rsizenum]));
    pred_test = round(mean(reshape(pred_test,[rsizenum,length(pred_test)/rsizenum])));
    pred_dev = round(mean(reshape(pred_dev,[rsizenum,length(pred_dev)/rsizenum])));
    
    TestLabels = mean(reshape(TestLabels,[rsizenum,length(TestLabels)/rsizenum]));
    DevLabels = mean(reshape(DevLabels,[rsizenum,length(DevLabels)/rsizenum]));
    
    acc_test = mean(TestLabels' == pred_test');
    acc_dev = mean(DevLabels' == pred_dev');
    disp(sprintf('Average,Testaf:%f,Devaf;%f',acc_test,acc_dev));
    
    ACC_test = [ACC_test;acc_test];
    ACC_dev = [ACC_dev;acc_dev];
    proball = [proball prob_test];
    TestLabelAll = [TestLabelAll;TestLabels'];
    proball = proball';
end

cfmat = cfmatrix(TestLabels',pred_test');
cfmat = bsxfun(@rdivide, cfmat, sum(cfmat));


%% Plotting CFmat
% names = ['Real    ';'Non-real'];
% draw_cm(cfmat,names,2);
% names = ['Real  ';'Print ';'Mobile';'Video '];
% draw_cm(CFmat,names,4);























