clc
clear
load('constants_file.mat');
node_num = 9;
SF = [7 8 9 10 11 12];
SF_Senrx = [-126.5 -127.25 -131.25 -132.75 -134.50 -133.25];
SF_Map = containers.Map(SF, SF_Senrx);
N = node_num; % iteration limit of game theory algorithm
num_sample = 1000;
max_sim = 10000;
gain = 6;
filename = 'dataset/LoRadataset_GT_test_1000.csv';
% revise all links distance
links(:, 3) = {0.345}; % unit: km

data = zeros(num_sample, 27);
count = 0;
for sim = 1:max_sim
    d = 1000 * cell2mat(links(:,3)); % m
    path_loss = calculateLpl(d, d0, Lpl_bar, gamma, sigma);
    tic
    %[SF_list, P_list, result, iters, payoff_vector] = GT_optimisation(path_loss, N);
    [optimal_SF, optimal_Ptx, min_device_energy_list] = greedy(path_loss);
    toc
    if result > 0
        data(count+1, 1:length(path_loss)) = path_loss';
        data(count+1, length(path_loss)+1:2*length(path_loss)) = SF_list;
        data(count+1, 2*length(path_loss)+1:end) = P_list;
        count = count + 1;
        fprintf('Saved samples %d/%d \n', count, num_sample)
    else
        fprintf('No solution \n')
    end
    if count >= num_sample
        break
    end
end
csvwrite(filename, data);
