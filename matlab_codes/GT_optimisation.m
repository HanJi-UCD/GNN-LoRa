function [final_SF, final_Ptx, result, count, payoff_vector] = GT_optimisation(path_loss, N)
    load('constants_file.mat','PL', 'CRC', 'H', 'BW', 'links');
    SF = [7 8 9 10 11 12];
    SF_Senrx = [-126.5 -127.25 -131.25 -132.75 -134.50 -133.25];
    SF_Map = containers.Map(SF, SF_Senrx);
    gain = 6;
    P_tx = 8:3:20;
    %%
    node_num = length(links);
    payoff_vector = inf;
    count = 0;
    mode = 0;
    SF_list = datasample(SF, node_num);
    P_list = datasample(P_tx, node_num);
    final_SF = SF_list;
    final_Ptx = P_list;
    while mode <= N
        DE = zeros(1, node_num);
        DE(SF_list > 10) = 1;
        payoff_list = power_cal(SF_list, PL, CRC, H, BW, DE, P_list);
        mutation_probability = payoff_list./sum(payoff_list)*2;
        % mutation_probability = ones(1, node_num);
        for i = 1:node_num
            p1 = rand(1);
            if p1 < mutation_probability(i) % apply mutation rule
                candidates = find_SF_P_candidates(SF_Map, SF_list, P_list, i, path_loss, gain);
                reward_vector = zeros(1, length(candidates(1,:)));
                if isempty(candidates(1,:))
                    mode = N + 1;
                    result = 0;
                    break
                else
                    for j = 1:length(candidates(1,:))
                        SF_list(i) = candidates(1, j);
                        P_list(i) = candidates(2, j);
                        reward_vector(j) = reward_cal(SF_list, P_list, node_num);
                    end
                    chosen_index = find(reward_vector == min(reward_vector), 1);
                    SF_list(i) = candidates(1, chosen_index);
                    P_list(i) = candidates(2, chosen_index);
                    aver_payoff = min(reward_vector);
                    
                    cons_now = final_Ptx(i) - SF_Map(final_SF(i)) - path_loss(i) + gain;
                    % fprintf('Now condition is %d \n',cons_now)
                    if aver_payoff < payoff_vector(end) || cons_now < 0 % mutation is better or power condition is not met
                        mode = 0;
                        payoff_vector = [payoff_vector, aver_payoff];
                        result = 1; % find solution
                        % record final SF and P
                        final_SF(i) = SF_list(i);
                        final_Ptx(i) = P_list(i);
                    else % mutation is worse, so do not change
                        mode = mode + 1;
                        result = 1;
                    end
                end
            end
            if mode > N
                break
            end
            count = count + 1;
        end
    end
end
