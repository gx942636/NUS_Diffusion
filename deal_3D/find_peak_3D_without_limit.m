load Yfgj_i.mat;
spec1 = fid_r;
spec1 = fft(spec1, [], 1);
spec1 = fft(spec1, [], 2);
spec1 = fftshift(spec1);

% 获取第三维度的尺寸
third_dim_size = size(spec1, 3);

% 对切片进行保存:
folder_name = 'Yfgj_i_slices';
if ~exist(folder_name, 'dir')
    mkdir(folder_name);
end

% 直接遍历第三维度的所有切片
for i = 1:third_dim_size
    spec = squeeze(spec1(:, :, i));

    filename = sprintf('label_spec%d', i);    
    save(fullfile(folder_name, filename), 'spec');
end

disp(['成功保存了 ', num2str(third_dim_size), ' 个切片到文件夹: ', folder_name]);