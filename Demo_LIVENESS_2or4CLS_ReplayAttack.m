clc; clear; close all;
%% Quality Test, Fake Face Test and Overall Test
% 21 Print Test; 22 Mobile Test; 23 Video Test
% (Clsnum = 2)
% 3  Overall Test
% (Clsnum = 2 or 4)

TestType = 3;
Clsnum = 4;
plot_code = ['r' 'g' 'b' 'c' 'k' 'm'];
PROB = [];
TEST = [];
ACCT = [];
ACCD = [];

%% Testing using SVM
% for k = 4:4
%     [prob,Test,ACC_test,ACC_dev] = Fun_Classification_SVM_Replay(TestType,Clsnum,k);
%     PROB = [PROB prob];
%     TEST = [TEST Test];
%     ACCT = [ACCT ACC_test];
%     ACCD = [ACCD ACC_dev];
% end

%% Testing using SAE
for k = 4:4
    [prob,Test,ACC_test,ACC_dev] = Fun_Classification_SAE_Replay(TestType,Clsnum,k);
    PROB = [PROB prob];
    TEST = [TEST Test];
    ACCT = [ACCT ACC_test];
    ACCD = [ACCD ACC_dev];
end

%% Plot ESBFD
% ACC = [];
% load ESBFD_C2;
% ACC = [ACC ACC_avg];
% load ESBFD_C4;
% ACC = [ACC ACC_avg];
% boxplot(ACC);

%% EDT plotting
Pmiss_min = 0.01;
Pmiss_max = 0.99;
Pfa_min = 0.01;
Pfa_max = 0.99;
Set_DET_limits(Pmiss_min,Pmiss_max,Pfa_min,Pfa_max);

C_miss = 1;
C_fa = 1;
P_target = 0.5;
PM = [];
PF = [];
for i = 1:size(PROB,2)
    pred = PROB(:,i);
    test = TEST(:,i);
    True_scores = pred(find(test == 1));
    False_scores = pred(find(test == 2));
    [P_miss,P_fa] = Compute_DET(True_scores,False_scores);
    hold on;
    Plot_DET (P_miss,P_fa,plot_code(i));
    Set_DCF(C_miss,C_fa,P_target);
    [DCF_opt Popt_miss Popt_fa] = Min_DCF(P_miss,P_fa);
    PM = [PM;Popt_miss];
    PF = [PF;Popt_fa];
end
legend('DoG-SVM','LBP-SVM','SBFD-SVM','DoG-SAE','LBP-SAE','SBFD-SAE');
% hold on;
% for i = 1:size(PROB,2)
%     Plot_DET (PM(i),PF(i),'ko');
% end
%% Plotting median accuracy
% DoG = [0.7368;0.7667;0.7833;0.7223;0.7278;0.7222;0.7167];
% LBP = [0.8333;0.8167;0.9167;0.8833;0.8589;0.8889;0.8667];
% SBFD = [0.9167;0.9333;0.8334;0.8611;0.9445;0.9167;0.8778];
% All = [DoG LBP SBFD];
% figure;
% plot(All);
% legend('DoG','LBP','SBFD');
