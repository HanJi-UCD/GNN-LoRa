function candidate_list = find_SF_P_candidates(SF_Map, SF_list, P_list, index, path_loss, gain)
    SF = [7 8 9 10 11 12];
    P = [8 11 14 17 20];  
    candidate_list = [];
    for i = 1:length(SF)
        for j = 1:length(P)
            SF_list(index) = SF(i);
            P_list(index) = P(j);
            % if satisfy constraint
            constraint = SF_Map(SF_list(index)) + path_loss(index) - gain;
            if P_list(index) > constraint
                candidate_list = [candidate_list, [SF_list(index); P_list(index)]];
            end
        end
    end
end