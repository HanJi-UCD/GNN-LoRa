function noisy_points = add_noise(points)
    vertices_list = zeros(32, 2, 3); % 32个三角形的所有顶点保存在一个3维矩阵里面
    vertices_list(1, :, :) = [0, 0, 2.5; 0, 2.5, 2.5];
    vertices_list(2, :, :) = [0, 2.5, 2.5; 0, 2.5, 0];
    vertices_list(3, :, :) = [2.5, 2.5, 5; 0, 2.5, 0];
    vertices_list(4, :, :) = [2.5, 5, 5; 2.5, 2.5, 0];
    vertices_list(5, :, :) = [5, 5, 7.5; 0, 2.5, 2.5];
    vertices_list(6, :, :) = [5, 7.5, 7.5; 0, 2.5, 0];
    vertices_list(7, :, :) = [7.5, 7.5, 10; 0, 2.5, 0];
    vertices_list(8, :, :) = [7.5, 10, 10; 2.5, 2.5, 0];
    
    vertices_list(9, :, :) = [0, 0, 2.5; 2.5, 5, 2.5];
    vertices_list(10, :, :) = [0, 2.5, 2.5; 5, 5, 2.5];
    vertices_list(11, :, :) = [2.5, 2.5, 5; 2.5, 5, 5];
    vertices_list(12, :, :) = [2.5, 5, 5; 2.5, 5, 2.5];
    vertices_list(13, :, :) = [5, 5, 7.5; 2.5, 5, 2.5];
    vertices_list(14, :, :) = [5, 7.5, 7.5; 5, 5, 2.5];
    vertices_list(15, :, :) = [7.5, 7.5, 10; 2.5, 5, 5];
    vertices_list(16, :, :) = [7.5, 10, 10; 2.5, 5, 2.5];
    
    vertices_list(17, :, :) = [0, 0, 2.5; 5, 7.5, 7.5];
    vertices_list(18, :, :) = [0, 2.5, 2.5; 5, 7.5, 5];
    vertices_list(19, :, :) = [2.5, 2.5, 5; 5, 7.5, 5];
    vertices_list(20, :, :) = [2.5, 5, 5; 7.5, 7.5, 5];
    vertices_list(21, :, :) = [5, 5, 7.5; 5, 7.5, 7.5];
    vertices_list(22, :, :) = [5, 7.5, 7.5; 5, 7.5, 5];
    vertices_list(23, :, :) = [7.5, 7.5, 10; 5, 7.5, 5];
    vertices_list(24, :, :) = [7.5, 10, 10; 7.5, 7.5, 5];
    
    vertices_list(25, :, :) = [0, 0, 2.5; 7.5, 10, 7.5];
    vertices_list(26, :, :) = [0, 2.5, 2.5; 10, 10, 7.5];
    vertices_list(27, :, :) = [2.5, 2.5, 5; 7.5, 10, 10];
    vertices_list(28, :, :) = [2.5, 5, 5; 7.5, 10, 7.5];
    vertices_list(29, :, :) = [5, 5, 7.5; 7.5, 10, 7.5];
    vertices_list(30, :, :) = [5, 7.5, 7.5; 10, 10, 7.5];
    vertices_list(31, :, :) = [7.5, 7.5, 10; 7.5, 10, 10];
    vertices_list(32, :, :) = [7.5, 10, 10; 7.5, 10, 7.5];
    noisy_points = zeros(length(points), 2);
    for i = 1:length(points)
        flag = 0;
        while flag == 0
            noisy_point = points(i) + rand(1,2);
            distance_list = zeros(1, 32);
            for j = 1:32
                A = vertices_list(j, :, 1);
                B = vertices_list(j, :, 2);
                C = vertices_list(j, :, 3);
                distance_list(j) = norm(noisy_point - A) + norm(noisy_point - B) + norm(noisy_point - C);
            end
            index = find(distance_list == min(distance_list));
            v1 = vertices_list(index, :, 1);
            v2 = vertices_list(index, :, 2);
            v3 = vertices_list(index, :, 3);
            inside = point_in_triangle(v1, v2, v3, noisy_point);
            if inside == 1
                flag = 1;
            end
        end
    end
end

function inside = point_in_triangle(v1, v2, v3, p)
    function s = sign(p1, p2, p3)
        s = (p1(1) - p3(1)) * (p2(2) - p3(2)) - (p2(1) - p3(1)) * (p1(2) - p3(2));
    end
    b1 = sign(p, v1, v2) < 0.0;
    b2 = sign(p, v2, v3) < 0.0;
    b3 = sign(p, v3, v1) < 0.0;
    inside = (b1 == b2) && (b2 == b3);
end