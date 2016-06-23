clc; clear; close all;
load HR_FaceFeatures_DoG;%HR_FaceFeaturesNew_256by256_64_mean_2class
LABELS(find(LABELS>1))=2;
classnum = length(unique(LABELS));
tnum = size(IMGFEATURE,2);
gnum = 60;

inputSize = size(IMGFEATURE,1);
ImgAll = IMGFEATURE;
% ImgAll = normalization_line(IMGFEATURE);
% ImgAll = normalization_line_2(ImgAll);
LabelsAll = reshape(LABELS,[gnum tnum/gnum]);
ImgAll = reshape(ImgAll,[inputSize,gnum,tnum/gnum]);
totalnum = size(ImgAll,3);
trainnum = floor(totalnum * 0.4);

T = 1;
CFmat = zeros(classnum,classnum);
ACC = [];
ACC1 = [];

for k = 1:T
    %% Training data
    nn = randperm(totalnum);
    n1 = nn(1:trainnum);
    TrainLabels = LabelsAll(:,n1);
    TrainImg = ImgAll(:,:,n1);
    TrainLabels = TrainLabels(:);
    TrainImg = reshape(TrainImg,[inputSize,size(TrainImg,2)*size(TrainImg,3)]);
    svmStruct = svmtrain(TrainImg',TrainLabels);
    
    %% Whiting train
%     meanSSCA = mean(TrainImg, 2);
%     TrainImg = bsxfun(@minus, TrainImg, meanSSCA);
%     sigma = TrainImg * TrainImg' / size(TrainImg,2);
%     [u, s, v] = svd(sigma);
%     ZCAWhite = u * diag(1 ./ sqrt(diag(s) + 0.1)) * u';
%     TrainImg = ZCAWhite * TrainImg;
%     
    %% Testing data
    n2 = nn(trainnum+1:end);
    TestLabels = LabelsAll(:,n2);
    TestImg = ImgAll(:,:,n2);
    TestLabels = TestLabels(:);
    TestImg = reshape(TestImg,[inputSize,size(TestImg,2)*size(TestImg,3)]);
    
    %% Whiting test
%     TestImg = bsxfun(@minus, TestImg, meanSSCA);
%     TestImg = ZCAWhite * TestImg;
      
    %% Testing
    pred = svmclassify(svmStruct,TestImg');
    acc = mean(TestLabels == pred)
    ACC = [ACC;acc]; 
    % Average
    pred = round(mean(reshape(pred,[30,length(pred)/30])));
    TestLabels = mean(reshape(TestLabels,[30,length(TestLabels)/30]));
    acc = mean(TestLabels' == pred')
    ACC1 = [ACC1;acc];
    
    cfmat = cfmatrix(TestLabels,pred');
    cfmat = bsxfun(@rdivide, cfmat, sum(cfmat)); 
    CFmat = CFmat + cfmat;

end
CFmat = CFmat./T;
names = ['Real    ';'Non-real'];
draw_cm(CFmat,names,2);

figure;
boxplot([ACC ACC1]);


