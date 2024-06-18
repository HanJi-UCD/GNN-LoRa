function count = cal_outage_probability(SF_list, Ptx_list, path_loss)
    SF = [7 8 9 10 11 12];
    SF_Senrx = [-126.5 -127.25 -131.25 -132.75 -134.50 -133.25];
    SF_Map = containers.Map(SF, SF_Senrx);
    count = 0;
    gain = 6;
    node_num = length(SF_list);
    for i = 1:node_num
        cons_now = Ptx_list(i) - SF_Map(SF_list(i)) - path_loss(i) + gain;
        if cons_now < 0
            count = count + 1;
        end
    end
end