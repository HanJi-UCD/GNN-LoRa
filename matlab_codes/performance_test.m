%% calculate power comsumption and runtime
% load dataset
GNN_data = csvread('pred_labels_GNN_1000.csv');
DNN_data = csvread('pred_labels_DNN_1000.csv');
node_num = 9;
N = node_num;
% samples = length(GNN_data);
samples = 50;
GNN_results = zeros(samples, 2);
DNN_results = zeros(samples, 2);
GT_results = zeros(samples, 2);
Greedy_results = zeros(samples, 2);
Top2_results = zeros(samples, 2);
Random_results = zeros(samples, 2);

count_GNN = 0;
count_DNN = 0;
for i = 1:samples
    % calculate power comsumption
    %% GNN
    path_loss = GNN_data(i, 1:9);
    SF_list = GNN_data(i, 10:18);
    Ptx_list = GNN_data(i, 19:27);
    GNN_results(i, 1) = power_consumption(SF_list,Ptx_list);
    count_GNN = count_GNN + cal_outage_probability(SF_list, Ptx_list, path_loss);
    p_GNN = count_GNN/(i*node_num);
    %% DNN
    SF_list = DNN_data(i, 10:18);
    Ptx_list = DNN_data(i, 19:27);
    DNN_results(i, 1) = power_consumption(SF_list,Ptx_list);
    count_DNN = count_DNN + cal_outage_probability(SF_list, Ptx_list, path_loss);
    p_DNN = count_DNN/(i*node_num);
    %% GT
    tic
    [SF_list, Ptx_list, ~, ~, ~] = GT_optimisation(path_loss, N);
    t = toc;
    GT_results(i, 1) = power_consumption(SF_list, Ptx_list);
    GT_results(i, 2) = t;
    %% Greedy
    tic
    [SF_list, Ptx_list, ~] = greedy(path_loss);
    t = toc;
    Greedy_results(i, 1) = power_consumption(SF_list, Ptx_list);
    Greedy_results(i, 2) = t;
    %% Top2
    tic
    [SF_list, Ptx_list, ~] = top2(path_loss);
    t = toc;
    Top2_results(i, 1) = power_consumption(SF_list, Ptx_list);
    Top2_results(i, 2) = t;
    %% Random
    tic
    [SF_list, Ptx_list, ~] = random(path_loss);
    t = toc;
    Random_results(i, 1) = power_consumption(SF_list, Ptx_list);
    Random_results(i, 2) = t;

    fprintf('')
    fprintf('Sample %d: \n', i)
    fprintf('Outage probability of GNN and DNN: %d %d \n', p_GNN, p_DNN)
    fprintf('Power consumption (Joule) of GNN, DNN methods are: %d, %d \n', GNN_results(i, 1), DNN_results(i, 1));
    fprintf('Power consumption (Joule) of GT, greedy, Top2 and Random methods are: %d, %d, %d, %d, \n', GT_results(i, 1), Greedy_results(i, 1), Top2_results(i, 1), Random_results(i, 1));
end
GNN_final = sum(GNN_results)/samples;
DNN_final = sum(DNN_results)/samples;
GT_final = sum(GT_results)/samples;
Greedy_final = sum(Greedy_results)/samples;
Top2_final = sum(Top2_results)/samples;
Random_final = sum(Random_results)/samples;
