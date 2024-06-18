function Power_consumption = power_consumption(SF_list,Ptx_list)
    load('constants_file.mat','PL','BW','CRC','H', 'links');
    node_num = length(links);
    DE = zeros(1, node_num);
    DE(SF_list > 10) = 1; 
    Tsym = 2.^SF_list / BW;
    Tx = Tsym .* (20.25 + max(ceil((8 * PL - 4 * SF_list + 28 + 16 * CRC - 20 * H) / (4 * (SF_list - 2 * DE))) * (1 + 4), 0));
    Tx = Tx * 1e-3; %s
    Ptx_list = (10 .^(Ptx_list / 10))* 1e-3;%w
    Power_consumption = sum(Tx .* Ptx_list)/node_num; %J
end
