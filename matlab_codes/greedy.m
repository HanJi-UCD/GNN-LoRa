function [optimal_SF, optimal_Ptx, min_device_energy_list] = greedy(path_loss)
    load('constants_file.mat');
    
    SF = [7 8 9 10 11 12];
    SF_Senrx = [-126.5 -127.25 -131.25 -132.75 -134.50 -133.25];
    SF_Map = containers.Map(SF, SF_Senrx);
    P_tx = 8:3:20;
    gain = 6;

    node_num = length(links);
    SF_list = datasample(SF, node_num);
    P_list = datasample(P_tx, node_num);
    Optimal_SF = SF_list;
    Optimal_Ptx = P_list;

    min_device_energy_list = inf(node_num, 1); % Initialize min energy to infinity for each device

    % Iterate through each device
    for device_idx = 1:node_num
        candidates = find_SF_P_candidates(SF_Map, SF_list, P_list, device_idx, path_loss, gain);
        
        for k = 1:size(candidates,2)
            SF_list(device_idx) = candidates(1,k);
            P_list(device_idx) = candidates(2,k);
            energy(device_idx) = power_collision(SF_list,P_list,device_idx);

            if energy(device_idx) < min_device_energy_list(device_idx)
                min_device_energy_list(device_idx) = energy(device_idx);
                optimal_SF(device_idx) = SF_list(device_idx);
                optimal_Ptx(device_idx) = P_list(device_idx);
            end
        end
        % Iterate through all SF and Ptx combinations
    end
end
