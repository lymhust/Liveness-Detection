clc;clear;close all;
%% Setting parameters
spara =[2 2 2 2];
% shear = shearing_filters_Myer([32 32 32 32],spara,256);
% save shear shear;
scale = length(spara);
direction = 2^spara(1) + 2;
channel = 3;

global shear;
load shear;

Fname = {'1','2','3','4','5','6','7','8','9','10','11','12','13','15',...
    '16','17','18','19','20'};
filter = 'maxflat';
IMGFEATURE = [];
LABELS = [];

E = com_norm('maxflat',[256 256],shear);
E = E(2:end,:);
E = E';
E = E(:);

for t = 1:length(Fname)
    foldername = Fname{t};% Choose the distortion type
    disp(foldername);
    folder = sprintf('.\\train_images\\%s\\',foldername);
    Files = dir(strcat(folder,'*.*'));
    
    for num_file = 3:length(Files)
        movename = Files(num_file).name;
        labels = str2double(movename(1));  
        img = imread(sprintf('%s\\%s',folder,movename));
%         img = rgb2gray(img);
        img = double(img);
        imfeatures = zeros(scale*direction*channel,1);
        ind = 1;
        %% Shearlet feature extraction
         for i = 1:3
             im = img(:,:,i);
            % Feature extraction
            Csh = shear_trans(im,filter,shear);
%             for s = 2:length(Csh)        % Index the scales.
%                 for w = 1:size(Csh{s},3) % Index the directions.
%                     % Index the shearlet coefficients. From coarse to fine. Leave out
%                     % HSC
%                     temp = Csh{s}(:,:,w);
%                     temp = temp(:);
%                     hs = sum(abs(temp));
%                     imfeatures(ind) = hs;
%                     ind = ind + 1;
%                 end % Index the directionsover.
%             end % Index the scales over.
%          end
        IMGFEATURE = [IMGFEATURE imfeatures];
        LABELS = [LABELS;labels];
 
    end
      
end
save('FaceFeaturesNew','IMGFEATURE','LABELS');

load FaceFeaturesGray;
%IMGFEATURE = bsxfun(@rdivide, IMGFEATURE, E);

sigma = sqrt(sum(IMGFEATURE.^2)./size(IMGFEATURE,1));
IMGFEATURE = bsxfun(@rdivide, IMGFEATURE, sigma+0.001);
IMGFEATURE = log2(IMGFEATURE);
ind1 = find(LABELS == 1);
ind2 = find(LABELS == 2);
ind3 = find(LABELS == 3);
ind4 = find(LABELS == 4);
img1 = mean(IMGFEATURE(:,ind1),2);
img2 = mean(IMGFEATURE(:,ind2),2);
img3 = mean(IMGFEATURE(:,ind3),2);
img4 = mean(IMGFEATURE(:,ind4),2);
tmp = [img1 img2 img3 img4];
plot(tmp);legend('Real','Photo','Blink','Video');


