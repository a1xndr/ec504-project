hash_size = 6:15;
% CCR1 = [0.9564,0.9457,0.9319,0.9356,0.9311,0.9431,0.9055,0.8843,0.9136,0.8765];
% n_err1 = [0,1,0,5,15,4,51,56,35,90];
% 
% CCR2 = [0.9522,0.9517,0.9248,0.9431,0.9384,0.9123,0.9355,0.9228,0.8998,0.94];
% n_err2 = [0,0,3,5,1,38,12,33,60,22];
% time2 = [507,541,63,324,311,130,210,188,82,468];
CCR = [0.9564,0.9537,0.9514,0.9489,0.9443,0.9401,0.9105,0.8943,0.8836,0.8265];
n_err = [0,0.5,0.7,5.0,9.8,12.2,28.3,46.4,60.3,97.6];
figure
plot(hash_size,CCR);
xlabel('Hash Size')
ylabel('CCR')
title('The CCR with Different Hash Size')

figure
plot(hash_size,n_err);
xlabel('Hash Size')
ylabel('Number of Samples with No Point in that Hash Space')
title('Number of Samples LSH Cannot Predict with Different Hash Size')

CCR_for_dif_k = [0.9255,0.9288,0.9502,0.9528,0.9564,0.9536,0.9529,0.9533,0.9518,0.9505];
figure
plot(1:10,CCR_for_dif_k)
xlabel('Number of Nearest Neighbors(K)')
ylabel('CCR')
title('The CCR with Different Choice of K')

