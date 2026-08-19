% ====== Data ======
x  = [20,50,80,100,150,200,300];
yL = [86.8, 87.8, 88.3, 88.6, 88.8, 89.0, 89.3];
yR = [56.4, 59.4, 60.6, 61.1,  61.4, 61.9, 62.1];

% ====== Style (editable) ======
leftHex = '#576fa0';
rightHex = '#b57979';
lw = 2;                % Line width

% ====== Axis limits and ticks ([] selects automatic values) ======
XLim     = [10 310];                  % X-axis range
YLimLeft = [85 90];              % Left Y-axis range
YLimRight= [55 63];              % Right Y-axis range

XTicks     = [20,50,80,100,150,200,300];         % X-axis ticks
YTicksLeft = [86 87 88 89];   % Left Y-axis ticks
YTicksRight= [56 58 60 62];     % Right Y-axis ticks

% ====== Style parameters ======
lwLineL   = 5.0;         % Left-curve line width
lwLineR   = 5.0;         % Right-curve line width
mkL       = 'o';         % Left-curve marker ('o','s','^','d','+','x',...)
mkR       = 's';         % Right-curve marker
mkSizeL   = 10;           % Left-marker size
mkSizeR   = 10;           % Right-marker size
axLineW   = 3;         % Axis line width


% ====== Plot ======
figure('Color','w');
ax = gca;
grid(ax,'off');                 % Disable the grid
box(ax,'on');                   % Show the top and right borders
ax.TickDir = 'out';             % Point tick marks outward
ax.TickLength = [0.02, 0.02];
ax.LineWidth = axLineW;         % Axis line width

% Left axis
yyaxis left
pL = plot(x, yL, '-','LineWidth', lwLineL, 'Color', hex2rgb(leftHex), ...
    'Marker', mkL, 'MarkerSize', mkSizeL, ...
    'MarkerFaceColor', hex2rgb(leftHex), ...
    'MarkerEdgeColor', hex2rgb(leftHex));   % LineCap/LineJoin are not set
if ~isempty(YLimLeft),    ylim(YLimLeft);      end
if ~isempty(YTicksLeft),  yticks(YTicksLeft);  end
yticklabels([]);          % Hide left Y-axis tick labels
ylabel('');               % Hide the left Y-axis label
ax.YColor = hex2rgb(leftHex); % Set the left-axis color

% Right axis
yyaxis right
pR = plot(x, yR, '-','LineWidth', lwLineR, 'Color', hex2rgb(rightHex), ...
    'Marker', mkR, 'MarkerSize', mkSizeR, ...
    'MarkerFaceColor', hex2rgb(rightHex), ...
    'MarkerEdgeColor', hex2rgb(rightHex));
if ~isempty(YLimRight),   ylim(YLimRight);     end
if ~isempty(YTicksRight), yticks(YTicksRight); end
yticklabels([]);          % Hide right Y-axis tick labels
ylabel('');               % Hide the right Y-axis label
ax.YColor = hex2rgb(rightHex); % Set the right-axis color

% X axis
if ~isempty(XLim),   xlim(XLim);     end
if ~isempty(XTicks), xticks(XTicks); end
xticklabels([]);           % Hide X-axis tick labels
xlabel('');                % Hide the X-axis label

% Hide the legend
legend('off');

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
