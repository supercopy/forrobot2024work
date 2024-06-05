% % 已知的坐标
% P_A = [1, 2, 3;
%        4, 5, 6;
%        7, 8, 9;
%        10, 11, 12]; % 坐标系A的四个点
% P_B = [0.5, 1.5, 2.5; 
%        3.5, 4.5, 5.5; 
%        6.5, 7.5, 8.5; 
%        9.5, 10.5, 11.5]; % 对应于坐标系A的四个点在坐标系B的坐标
% 
% % 使用最小二乘法估计变换矩阵T_AB
% T_AB = estimate_transform_matrix(P_A', P_B');
% 
% % 验证T_AB
% P_A_transformed = T_AB * [P_A, ones(4, 1)]';
% P_A_transformed = P_A_transformed(1:3, :);
% 
% disp('验证估计得到的坐标系A到坐标系B的变换矩阵T_AB：');
% disp('变换前坐标系A的点：');
% disp(P_A);
% disp('变换后坐标系A的点投影到坐标系B的点：');
% disp(P_A_transformed');
% disp('坐标系B的对应点：');
% disp(P_B);

% 已知的坐标
P_A = [0.11820718383789063, 0.05533146667480469, 1.0028218383789063;
       -0.01996559715270996, 0.09150011444091796, 0.94927001953125;
       -0.19349075317382813, 0.07384275817871094, 0.9843123168945312;
       -0.38008270263671873, 0.05715393829345703, 1.164743896484375]; % 坐标系A的四个点
P_B = [-0.21501827451228817, -0.31151167908915567, -0.06251055943497788; 
       -0.09883309329815113, -0.4565774141020829, -0.07697371230214964; 
       0.09122137521581827, -0.4566295945409391, -0.07724894630523474; 
       0.2409934766406208, -0.2882685735221945, -0.04749755037280354]; % 对应于坐标系A的四个点在坐标系B的坐标

% 使用最小二乘法估计变换矩阵T_AB
T_AB = estimate_transform_matrix(P_A', P_B');

% 验证T_AB
P_A_transformed = T_AB * [P_A, ones(4, 1)]';
P_A_transformed = P_A_transformed(1:3, :);

disp('验证估计得到的坐标系A到坐标系B的变换矩阵T_AB：');
disp('变换前坐标系A的点：');
disp(P_A);
disp('变换后坐标系A的点投影到坐标系B的点：');
disp(P_A_transformed');
disp('坐标系B的对应点：');
disp(P_B);
disp('变换矩阵T_AB');
disp(T_AB);