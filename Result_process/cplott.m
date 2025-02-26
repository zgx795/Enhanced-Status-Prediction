clc;
clear;
close all;

% 设置文件路径
groundTruthFile = 'C:\Users\13844\Desktop\papers\paper4\results\482412\GroundTruth_Data.csv';
predictionFile = 'C:\Users\13844\Desktop\papers\paper4\results\482412\Prediction_Data.csv';
outputDir = 'C:\Users\13844\Desktop\papers\paper4\png\Figures481211';
% 配色方案
actualColor = [247, 150, 31] / 255; % 橙色
predictedColor = [67, 171, 140] / 255; % 绿色
errorColor = [128, 164, 251] / 255; % 蓝色
% 特征名称与单位映射表
unitMap = struct( ...
    'MFBT', '°C', 'MRBT', '°C', 'CUR', 'A', 'IP', 'kPa', ...
    'OP', 'kPa', 'FSBT', '°C', 'VBF', '%', 'VBAC', '%', ...
    'RSBT', '°C', 'TBT', '°C', 'PACT', '°C', 'PBCT', '°C', 'PCCT', '°C' ...
);
% 创建输出文件夹
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% 读取CSV文件
groundTruthData = readtable(groundTruthFile);
predictionData = readtable(predictionFile);

% 获取特征名称（列名），跳过第一列的 Sample Index
featureNames = groundTruthData.Properties.VariableNames(2:end);

% 遍历每个特征绘图
for i = 1:length(featureNames)
    % 提取当前特征的实际值和预测值
    featureName = featureNames{i};
    trueValues = groundTruthData.(featureName);
    predValues = predictionData.(featureName);

   
    % 计算误差
    errorValues = trueValues - predValues;

    % 获取特征单位
    unit = unitMap.(featureName);

    % 创建新图
    figure;

    % 左图：实际值与预测值
    subplot(1, 2, 1);
    plot(1:length(trueValues), trueValues, '-', 'LineWidth', 2, 'Color', actualColor); % 实际值：橙色实线
    hold on;
    plot(1:length(predValues), predValues, '-', 'LineWidth', 2, 'Color', predictedColor); % 预测值：蓝色虚线
    xlabel('Sample Points', 'FontSize', 16, 'FontWeight', 'bold', 'FontName', 'Times New Roman');
    ylabel(['Value (', unit, ')'], 'FontSize', 16, 'FontWeight', 'bold', 'FontName', 'Times New Roman');
    legend({'Actual Values', 'Predicted Values'}, 'FontSize', 12, 'Location', 'northeast', 'Box', 'off', 'FontName', 'Times New Roman');
    set(gca, 'LineWidth', 1.5, 'FontSize', 14, 'Box', 'on', 'FontName', 'Times New Roman');
    xlim([1, length(trueValues)]); % 自适应调整X轴范围
    grid on;

    % 右图：误差
    subplot(1, 2, 2);
    plot(1:length(errorValues), errorValues, '-', 'LineWidth', 2, 'Color', errorColor); % 误差：黄色实线
    xlabel('Sample Points', 'FontSize', 16, 'FontWeight', 'bold', 'FontName', 'Times New Roman');
    ylabel(['Error (', unit, ')'], 'FontSize', 16, 'FontWeight', 'bold', 'FontName', 'Times New Roman');
    set(gca, 'LineWidth', 1.5, 'FontSize', 14, 'Box', 'on', 'FontName', 'Times New Roman');
    xlim([1, length(errorValues)]); % 自适应调整X轴范围
    grid on;

    % 调整图形布局
    set(gcf, 'Position', [100, 100, 1600, 400]); % 调整窗口比例

    % 保存图像为 PNG 和 FIG 格式，设置 300 DPI
    exportgraphics(gcf, fullfile(outputDir, [featureName, '_Comparison.png']), 'Resolution', 300);
    savefig(fullfile(outputDir, [featureName, '_Comparison.fig']));
    close;
end

disp(['Figures saved in: ', outputDir]);
