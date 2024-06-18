function [SF_list,P_list,min_device_energy_list] = random(path_loss)
    load('constants_file.mat');
    
    SF = [7 8 9 10 11 12];
    SF_Senrx = [-126.5 -127.25 -131.25 -132.75 -134.50 -133.25];
    SF_Map = containers.Map(SF, SF_Senrx);
    P_tx = 8:3:20;
    gain = 6;

    node_num = length(links);
    SF_list = datasample(SF, node_num);
    P_list = datasample(P_tx, node_num);
    
    min_device_energy_list = inf(node_num, 1); % Initialize min energy to infinity for each device

    % Iterate through each device
    for device_idx = 1:node_num
        candidates = find_SF_P_candidates(SF_Map, SF_list, P_list, device_idx, path_loss, gain);
        if candidates > 0
            numcols = size(candidates,2);
            rand_idx = randi(numcols);
            SF_list(device_idx) = candidates(1,rand_idx);
            P_list(device_idx) = candidates(2,rand_idx);
            min_device_energy_list(device_idx) = power_collision(SF_list, P_list,device_idx);
        else
            fprintf('No valid SF and Ptx combination found for device %d\n', device_idx);
            error("No valid SF and Ptx combination found");
        end
        % Iterate through all SF and Ptx combinations
    end

end

