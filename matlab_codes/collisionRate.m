function [Coll_Rate] = collisionRate(SF_list,simNum)
    load('constants_file.mat','PL', 'CRC', 'H', 'BW', 'NF', 'd0', 'Lpl_bar', 'gamma', 'sigma','links');
    timespan = 60 * 100; % ms
    timeinterval = 10; % ms
    numDevices = size(links, 1);
    numChannels = 2; % 可用信道数量
    numslots = timespan / timeinterval;
    sf_time_mapping = containers.Map({7, 8, 9, 10, 11, 12}, [169, 297, 533, 985, 2134, 3940]); %100 bytes
    collisionMatrix = zeros(numDevices,1);
    % distance_threshold = 0.5; %km
    
    for simmulation = 1:simNum
        
        deviceChannels = randi([1 numChannels], numDevices, 1); % 随机分配信道
        deviceSendTimes = zeros(numDevices, 1); % 初始化为0，稍后计算中继节点延迟
    
        % 记录每个设备的最优参数碰撞率
        for i = 1:numDevices
            srcDevice = links{i, 1};
            if startsWith(srcDevice, 'R') % 如果是中继节点
                % 找到所有发送到该中继节点的设备
                idx = find(strcmp(links(:,2), srcDevice));
                % 计算这些设备发送结束的最晚时间
                latestEnd = max(deviceSendTimes(idx) + arrayfun(@(sf) sf_time_mapping(sf), SF_list(idx)));
                % 中继节点的发送时间为这些设备发送结束的最晚时间
                deviceSendTimes(i) = ceil(latestEnd / timeinterval) + 10; % 加上10ms的处理延迟
            else
                deviceSendTimes(i) = randi([1, numslots], 1, 1); % 非中继节点随机分配发送时间
            end
        end
        
        % 计算碰撞率
        collision = zeros(numDevices, 1);
        channelSlotOccupancy = zeros(numslots, numChannels);
        channelSFs = cell(numslots,numChannels); 
       
        for i = 1:numDevices
            channel = deviceChannels(i);
            sendTime = deviceSendTimes(i);
            duration = sf_time_mapping(SF_list(i));
            for t = sendTime:min(sendTime + duration / timeinterval, numslots)
                if channelSlotOccupancy(t, channel) ~= 0 
                    occupiedDevice = channelSlotOccupancy(t,channel);
                    % occupiedDistance = channelDistances{t,channel};
                    % currentDistance = Distance_matrix(i, :);
                    if SF_list(i) == SF_list(occupiedDevice) 
                        % || (isLinkConnected(links, i, occupiedDevice) && ...
                        %     any(currentDistance <= distance_threshold)) 
                        collision(i) = 1;
                        collision(occupiedDevice) = 1;
                    end
                else
                    channelSlotOccupancy(t, channel) = i;
                    channelSFs{t, channel} = SF_list(i);
                    % channelDistances{t, channel} = Distance_matrix(i, :); % 记录当前设备的距离
                end
            end
        end
        
        collisionMatrix = collisionMatrix + collision;
    end

    Coll_Rate = collisionMatrix/ simmulation;
    Coll_Rate = Coll_Rate';
end

% function connected = isLinkConnected(links, device1, device2)
%     src1 = links{device1, 1};
%     dst1 = links{device1, 2};
%     src2 = links{device2, 1};
%     dst2 = links{device2, 2};
% 
%     connected = (strcmp(src1, src2) || strcmp(src1, dst2) || strcmp(dst1, src2) || strcmp(dst1, dst2));
% end

