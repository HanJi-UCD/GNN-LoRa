function [optimal_SF, optimal_Ptx, optimal_device_energy_list] = top2(path_loss)
    % 调用函数找到每个设备的前两个最优组合
    top2_combinations = find_top2_combinations(path_loss);
    node_num = length(path_loss);
    
    % 计算所有组合数量
    num_combinations = 1;
    num_candidates_per_device = zeros(node_num, 1);
    for device_idx = 1:node_num
        num_candidates_per_device(device_idx) = size(top2_combinations{device_idx}, 2);
        num_combinations = num_combinations * num_candidates_per_device(device_idx);
    end
    
    % 初始化变量
    min_total_energy = inf;
    optimal_SF = zeros(1,node_num);
    optimal_Ptx = zeros(1,node_num);
    optimal_device_energy_list = Inf(node_num,1);

    % 遍历所有组合
    for comb_idx = 0:(num_combinations - 1)
        SF_list = zeros(1,node_num);
        P_list = zeros(1,node_num);
        Power_list = Inf(node_num, 1);

        temp_comb = comb_idx;
        for device_idx = 1:node_num
            num_candidates = num_candidates_per_device(device_idx);
            choice_idx = mod(temp_comb, num_candidates) + 1;
            temp_comb = floor(temp_comb / num_candidates);
            
            SF_list(device_idx) = top2_combinations{device_idx}(1, choice_idx);
            P_list(device_idx) = top2_combinations{device_idx}(2, choice_idx);
        end
        
        % 计算总能耗
        
        [Power_list,reward] = power_collisiontop2(SF_list,P_list,node_num);
        total_energy = reward * node_num;
        
        % 更新最小能耗和对应的SF、Ptx
        if total_energy < min_total_energy
            min_total_energy = total_energy;
            optimal_device_energy_list = Power_list;
            optimal_SF = SF_list;
            optimal_Ptx = P_list;
        end
    end
end

function top2_combinations = find_top2_combinations(path_loss)
    load('constants_file.mat','PL', 'CRC', 'H', 'BW');
    SF = [7 8 9 10 11 12];
    SF_Senrx = [-126.5 -127.25 -131.25 -132.75 -134.50 -133.25];
    SF_Map = containers.Map(SF, SF_Senrx);
    gain = 6;
    DE = 0;
    P_tx = 8:3:20;
    node_num = length(path_loss);
    
    SF_list = datasample(SF, node_num);
    P_list = datasample(P_tx, node_num);    
    
    top2_combinations = cell(node_num, 1);
    
    for device_idx = 1:node_num
        candidates = find_SF_P_candidates(SF_Map, SF_list, P_list, device_idx, path_loss, gain);
    
        numcols = size(candidates,2);
        energies = zeros(numcols, 1);
        DE = zeros(1, node_num);
        DE(SF_list > 10) = 1;  

        if numcols < 2
            if candidates(1,1)>10
                DE = 1;
            end
            energies(1) = power_cal(candidates(1,1),PL, CRC, H, BW, DE,candidates(2,1));
            top2_combinations{device_idx} = candidates(:,1);
        else
        
            for i = 1:numcols
                SF = candidates(1,i);
                P = candidates(2,i);
                if SF > 10
                    DE = 1;
                else
                    DE = 0;
                end
                energies(i) = power_cal(SF,PL,CRC, H, BW, DE,P); 
            end
            
            [sorted_energies, sorted_indices] = sort(energies);
            top2_combinations{device_idx} = candidates(:, sorted_indices(1:2));
        end
    end
end