clc; clear; close all;
%% Quality Test, Fake Face Test and Overall Test
% 11 Low Quality Test; 12 Normal Quality Test; 13 High Quality Test
% 3  Overall Test

T = 1;
TestType = 11;
PROB = [];
TEST = [];
ACC = [];

for k = 3:3
    [prob,Test,ACC_avg] = Fun_FaceRecognition(TestType,k,T);
    PROB = [PROB prob];
    TEST = [TEST Test];
    ACC = [ACC ACC_avg];
end

boxplot(ACC);
