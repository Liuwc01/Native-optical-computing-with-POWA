% ====== Data ======
x  = [1,2,4,7,10,13];
y1 = [157.2, 174.2, 272.5, 927.8, 3657.8, 6810];
y2 = [157.8, 159.5, 171.4, 253.4,  691.6, 1100];

% ====== Style (editable) ======
line1Hex = '#9f9f9f';
line2Hex = '#b57979';
fillHex  = '#dea3a2 ';
alphaFill = 0.4;      % Fill opacity, 0 to 1
lw = 5;                % Line width
method = 'spline';      % 'pchip' or 'spline'

% ====== Smooth while preserving monotonicity: pchip + cummax ======
xf  = linspace(min(x), max(x), 1600);
y1s = interp1(x, y1, xf, 'pchip');
y2s = interp1(x, y2, xf, 'pchip');
% Flatten any small decrease to enforce a nondecreasing curve
y1s = cummax(y1s);   % Requires R2018b+; use a loop in earlier releases
y2s = cummax(y2s);

% ====== Plot ======
ax1 = gca;
hold(ax1, 'on'); % Keep subsequent plot commands on these axes
% figure('Color','w'); hold on
xPoly = [xf, fliplr(xf)];
yPoly = [y1s, fliplr(y2s)];
fill(xPoly, yPoly, hex2rgb(fillHex), 'FaceAlpha',alphaFill, 'EdgeColor','none');

plot(xf, y1s, 'LineWidth', lw, 'Color', hex2rgb(line1Hex));
plot(xf, y2s, 'LineWidth', lw, 'Color', hex2rgb(line2Hex));

hold(ax1, 'off'); % Release the axes so later plot commands replace the plot

set(ax1, ...
    'XLim',         [0.9, 13.1], ...                % X-axis range
    'YLim',         [0 7500], ...                % Y-axis range
    'XTick',        [1 4 7 10 13], ... % X-axis tick positions
    'YTick',        [1000 3000 5000 7000], ...        % Y-axis tick positions
    'XTickLabel',   [], ...                        % Hide X-axis tick labels
    'YTickLabel',   [], ...                        % Hide Y-axis tick labels
    'TickDir',      'out', ...                    % Point ticks outward
    'TickLength',   [0.03, 0.03], ...  % Set tick length
    'LineWidth',    3, ...                        % Thicken axes and tick marks
    'Box',          'off');                       % Let ax2 draw the top and right borders

% 2. Remove the ax1 title and labels
title(ax1, '');
xlabel(ax1, '');
ylabel(ax1, '');
drawnow;

% 3. Create ax2 to draw the top and right borders
ax2 = axes('Position', get(ax1, 'Position'), ... % Place ax2 exactly over ax1
           'Color', 'none', ...                   % Use a transparent background
           'XAxisLocation', 'top', ...            % Place the X axis at the top
           'YAxisLocation', 'right', ...           % Place the Y axis at the right
           'XTick', [], ...                       % Remove top-border ticks
           'YTick', [], ...                       % Remove right-border ticks
           'Box', 'on', ...                       % Draw the ax2 border box
           'LineWidth', 3);                       % Match the ax1 border width




% ====== Convert hexadecimal color to [0,1] RGB ======
function rgb = hex2rgb(hex)
    if isstring(hex), hex = char(hex); end
    hex = strtrim(hex);
    if startsWith(hex, '#'), hex = hex(2:end); end
    if numel(hex)==3, hex = regexprep(hex, '(.)', '$1$1'); end
    assert(numel(hex)==6 && all(isstrprop(hex,'xdigit')), ...
        'hex2rgb: expected #RRGGBB or #RGB format.');
    vals = sscanf(hex, '%2x%2x%2x', 3)';
    rgb = vals/255;
end
