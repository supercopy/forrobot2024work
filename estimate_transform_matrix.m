function T_AB = estimate_transform_matrix(P_A, P_B)

    % 计算坐标系A和坐标系B的质心（平均值）
    centroid_A = mean(P_A, 2);
    centroid_B = mean(P_B, 2);

    % 计算去中心化的坐标
    P_A_centered = P_A - centroid_A;
    P_B_centered = P_B - centroid_B;

    % 计算旋转矩阵 R
    H = P_A_centered * P_B_centered';
    [U, ~, V] = svd(H);
    R = V * U';

    % 计算平移向量 t
    t = centroid_B - R * centroid_A;

    % 构建变换矩阵 T_AB
    T_AB = eye(size(P_A, 1) + 1);
    T_AB(1:size(P_A, 1), 1:size(P_A, 1)) = R;
    T_AB(1:size(P_A, 1), end) = t;

end