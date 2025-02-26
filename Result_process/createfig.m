
close all
% Load performance data
PerformanceData;

% Create output folder for charts
output_folder = 'C:\Users\13844\Desktop\papers\paper4\png\Feature_Analysis_Charts';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% Extract all feature names
features = fieldnames(PerformanceData);

% Loop through each feature and generate charts
for i = 1:length(features)
    feature = features{i};
    % Ensure the feature name is valid for file saving
    feature_safe = matlab.lang.makeValidName(feature);
    data = PerformanceData.(feature);
    
    % Check for data completeness and consistency
    if ~isfield(data, 'StepLengths') || isempty(data.StepLengths) || any(structfun(@isempty, data))
        warning('Data for feature "%s" is incomplete or missing. Skipping...', feature);
        continue;
    end

    % Create figure
    figure_handle = figure('Visible', 'on'); % Keep the figure visible for adjustments
    
    % Safely plot selected metrics
    try
        plot(data.StepLengths, data.MAE, '-o', 'LineWidth', 1.5); hold on;
        plot(data.StepLengths, data.MSE, '-s', 'LineWidth', 1.5);
        plot(data.StepLengths, data.RMSE, '-^', 'LineWidth', 1.5);
        plot(data.StepLengths, data.MAPE, '-d', 'LineWidth', 1.5);
    catch err
        warning('Error plotting data for feature "%s": %s. Skipping...', feature, err.message);
        close(figure_handle);
        continue;
    end
    
    % Customize plot
    legend('MAE', 'MSE', 'RMSE', 'MAPE (%)', 'Location', 'Best');
    xlabel('Prediction Step Length (Minutes)', 'FontSize', 12);
    ylabel('Performance Metric Value', 'FontSize', 12);
    title(sprintf('%s Performance Metrics', feature), 'FontSize', 14);
    grid on;
    
    % Save figure as .png
    png_save_path = fullfile(output_folder, sprintf('%s_Performance.png', feature_safe));
    try
        saveas(figure_handle, png_save_path);
    catch err
        warning('Error saving .png file for feature "%s": %s', feature, err.message);
    end
    
    % Save figure as .fig
    fig_save_path = fullfile(output_folder, sprintf('%s_Performance.fig', feature_safe));
    try
        savefig(figure_handle, fig_save_path); % Use savefig for better compatibility
    catch err
        warning('Error saving .fig file for feature "%s": %s', feature, err.message);
    end

    % Keep the figure open for manual adjustments
    disp(['Figure for feature "', feature, '" has been generated and kept open.']);
end

disp('All feature analysis charts have been generated and are kept open for further adjustments.');
