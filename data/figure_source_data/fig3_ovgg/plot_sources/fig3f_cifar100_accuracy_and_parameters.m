%% 1. Prepare data and colors
% Three entries (categories)
category_labels = {'', '', ''};
x_positions = 1:3; 

% Data for the left and right Y axes (three values each)
data_left = [62.6, 61.1, 64.2];
data_right = [1139652, 1139652, 6860388];

% Enter hexadecimal color codes here
hex_colors = [
    "#576fa0";  % Left axis, entry 1
    "#b57979";  % Left axis, entry 2
    "#9f9f9f";  % Left axis, entry 3
    "#a7b9d7";  % Right axis, entry 1
    "#dea3a2";  % Right axis, entry 2
    "#cfcece"   % Right axis, entry 3
];

% --- Convert hexadecimal codes to a MATLAB RGB matrix ---
% This explicit conversion avoids sscanf compatibility errors
colors_rgb = zeros(length(hex_colors), 3);
for i = 1:length(hex_colors)
    % 1. Extract the current color text from the string or cell array
    if iscell(hex_colors)
        current_item = hex_colors{i};
    else
        current_item = hex_colors(i);
    end
    
    % 2. Convert it explicitly to a character vector
    current_char_vector = char(current_item);
    
    % 3. Remove the leading '#' from the character vector
    hex_part = current_char_vector(2:end);
    
    % 4. Convert with sscanf
    colors_rgb(i,:) = sscanf(hex_part, '%2x', [1 3]) / 255;
end
% ------------------------------------------------

%% 2. Create the plot
figure; 
hold on;

% --- Draw and configure the left axis ---
yyaxis left; 

bar_width = 0.3;
offset = 0.15;
b1 = bar(x_positions - offset, data_left, bar_width, 'FaceColor', 'flat');
b1.EdgeColor = 'none';

ylabel('');
ax = gca; 
ax.YAxis(1).Color = 'k';

% Configure the left-axis range and tick positions
ylim([57, 65]); 
ax.YAxis(1).TickValues = [58, 60, 62, 64];


% --- Draw and configure the right axis ---
yyaxis right; 

b2 = bar(x_positions + offset, data_right, bar_width, 'FaceColor', 'flat');
b2.EdgeColor = 'none';

ylabel('');
ax.YAxis(2).Color = 'k';

% Configure the right-axis range and tick positions
ylim([0, 8000000]); 
ax.YAxis(2).TickValues = [2000000, 4000000, 6000000, 8000000];


%% 3. Apply a custom color to each bar
b1.CData = colors_rgb(1:3, :);
b2.CData = colors_rgb(4:6, :);


%% 4. Format the chart
hold off;
title('');
xlabel('');

set(gca, 'XTick', x_positions, 'XTickLabel', category_labels, 'LineWidth', 3, 'TickLength',   [0.02, 0.02]);
ax.TickDir = 'out';
box on;
