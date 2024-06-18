function [power_list, reward] = power_collisiontop2(SF_list, P_list, node_num)
    load('constants_file.mat','PL', 'CRC', 'H', 'BW');
    collision_rate = collisionRate(SF_list', 100);
    % calculate new payoff, assume retransmission times as 2
    DE = zeros(1, node_num);
    DE(SF_list > 10) = 1;
    device_energy = power_cal(SF_list, PL, CRC, H, BW, DE, P_list);
    power_list = device_energy + device_energy.*collision_rate;
    reward = sum(power_list)/node_num;
end