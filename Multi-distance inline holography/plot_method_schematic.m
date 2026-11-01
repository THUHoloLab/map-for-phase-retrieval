function outputFile = plot_method_schematic(outputFile)
%PLOT_METHOD_SCHEMATIC Draw a comparison of multi-distance algorithms.
%   OUTPUTFILE = PLOT_METHOD_SCHEMATIC() creates a publication-oriented
%   schematic and saves it as method_schematic.png in this directory.
%
%   The diagram compares:
%     (1) sequential layer-wise amplitude projection,
%     (2) global backpropagation using every distance, and
%     (3) random-layer stochastic backpropagation.

arguments
    outputFile (1,1) string = "method_schematic.png"
end

figure("Color", "white", "Position", [100, 100, 1500, 820], ...
    "Name", "Multi-distance inline holography methods");
annotation("textbox", [0.03, 0.94, 0.94, 0.045], ...
    "String", "Multi-distance inline holography: three reconstruction strategies", ...
    "EdgeColor", "none", "HorizontalAlignment", "center", ...
    "FontSize", 18, "FontWeight", "bold");

% Shared acquisition model.
annotation("textbox", [0.035, 0.79, 0.17, 0.07], ...
    "String", "Object field U(x,y)", "HorizontalAlignment", "center", ...
    "VerticalAlignment", "middle", "FontSize", 12, ...
    "BackgroundColor", [0.88, 0.93, 1], "EdgeColor", [0.25, 0.45, 0.75], ...
    "LineWidth", 1.2);
annotation("textbox", [0.25, 0.79, 0.17, 0.07], ...
    "String", "Angular-spectrum propagation", "HorizontalAlignment", "center", ...
    "VerticalAlignment", "middle", "FontSize", 12, ...
    "BackgroundColor", [0.91, 0.96, 0.91], "EdgeColor", [0.25, 0.55, 0.30], ...
    "LineWidth", 1.2);
annotation("textbox", [0.465, 0.79, 0.20, 0.07], ...
    "String", "Holograms I(z_1), I(z_2), ..., I(z_K)", ...
    "HorizontalAlignment", "center", "VerticalAlignment", "middle", ...
    "FontSize", 12, "BackgroundColor", [1, 0.94, 0.83], ...
    "EdgeColor", [0.75, 0.48, 0.12], "LineWidth", 1.2);
annotation("textbox", [0.74, 0.79, 0.20, 0.07], ...
    "String", "Amplitude constraints sqrt(I(z_k))", ...
    "HorizontalAlignment", "center", "VerticalAlignment", "middle", ...
    "FontSize", 12, "BackgroundColor", [0.95, 0.90, 0.98], ...
    "EdgeColor", [0.55, 0.30, 0.70], "LineWidth", 1.2);
add_arrow([0.205, 0.825], [0.245, 0.825]);
add_arrow([0.42, 0.825], [0.46, 0.825]);
add_arrow([0.67, 0.825], [0.735, 0.825]);

% Method panels.
panelX = [0.025, 0.345, 0.665];
panelWidth = 0.29;
panelY = 0.08;
panelHeight = 0.60;
titles = ["Sequential layer-wise", "Global backpropagation", ...
    "Random-layer backpropagation"];
colors = {[0.20, 0.43, 0.72], [0.18, 0.55, 0.34], [0.70, 0.35, 0.15]};

for panel = 1:3
    annotation("rectangle", [panelX(panel), panelY, panelWidth, panelHeight], ...
        "Color", colors{panel}, "LineWidth", 1.4);
    annotation("textbox", [panelX(panel), 0.635, panelWidth, 0.045], ...
        "String", titles(panel), "EdgeColor", "none", ...
        "HorizontalAlignment", "center", "FontSize", 13, ...
        "FontWeight", "bold");
end

% Sequential: z1 -> z2 -> z3 -> z4, with immediate object update.
draw_box(panelX(1)+0.07, 0.54, 0.15, 0.06, "U at object plane", [0.88, 0.93, 1]);
draw_box(panelX(1)+0.07, 0.43, 0.15, 0.06, "Propagate to z_k", [0.91, 0.96, 0.91]);
draw_box(panelX(1)+0.07, 0.32, 0.15, 0.06, "Replace amplitude", [1, 0.94, 0.83]);
draw_box(panelX(1)+0.07, 0.21, 0.15, 0.06, "Backpropagate", [0.95, 0.90, 0.98]);
add_arrow([panelX(1)+0.145, 0.54], [panelX(1)+0.145, 0.49]);
add_arrow([panelX(1)+0.145, 0.43], [panelX(1)+0.145, 0.38]);
add_arrow([panelX(1)+0.145, 0.32], [panelX(1)+0.145, 0.27]);
add_arrow([panelX(1)+0.145, 0.21], [panelX(1)+0.145, 0.145]);
annotation("textbox", [panelX(1)+0.035, 0.095, 0.22, 0.035], ...
    "String", "z_1 -> z_2 -> ... -> z_K", "EdgeColor", "none", ...
    "HorizontalAlignment", "center", "FontSize", 10, "Color", colors{1});

% Global: all residuals are accumulated before one update.
draw_box(panelX(2)+0.025, 0.49, 0.10, 0.055, "z_1", [1, 0.94, 0.83]);
draw_box(panelX(2)+0.145, 0.49, 0.10, 0.055, "z_2", [1, 0.94, 0.83]);
draw_box(panelX(2)+0.025, 0.37, 0.10, 0.055, "z_3", [1, 0.94, 0.83]);
draw_box(panelX(2)+0.145, 0.37, 0.10, 0.055, "z_K", [1, 0.94, 0.83]);
draw_box(panelX(2)+0.07, 0.23, 0.15, 0.06, "Sum residuals", [0.95, 0.90, 0.98]);
draw_box(panelX(2)+0.07, 0.11, 0.15, 0.06, "Global update U", [0.88, 0.93, 1]);
for x = [panelX(2)+0.075, panelX(2)+0.195]
    add_arrow([x, 0.49], [panelX(2)+0.145, 0.29]);
    add_arrow([x, 0.37], [panelX(2)+0.145, 0.29]);
end
add_arrow([panelX(2)+0.145, 0.23], [panelX(2)+0.145, 0.17]);
annotation("textbox", [panelX(2)+0.03, 0.085, 0.23, 0.03], ...
    "String", "all distances contribute to each step", "EdgeColor", "none", ...
    "HorizontalAlignment", "center", "FontSize", 10, "Color", colors{2});

% Random layer: one layer is sampled at each iteration.
draw_box(panelX(3)+0.07, 0.49, 0.15, 0.06, "Random k ~ Uniform{1,...,K}", [1, 0.94, 0.83]);
draw_box(panelX(3)+0.07, 0.37, 0.15, 0.06, "Residual at z_k", [0.95, 0.90, 0.98]);
draw_box(panelX(3)+0.07, 0.25, 0.15, 0.06, "Backpropagate", [0.91, 0.96, 0.91]);
draw_box(panelX(3)+0.07, 0.13, 0.15, 0.06, "Stochastic update", [0.88, 0.93, 1]);
add_arrow([panelX(3)+0.145, 0.49], [panelX(3)+0.145, 0.43]);
add_arrow([panelX(3)+0.145, 0.37], [panelX(3)+0.145, 0.31]);
add_arrow([panelX(3)+0.145, 0.25], [panelX(3)+0.145, 0.19]);
annotation("textbox", [panelX(3)+0.025, 0.085, 0.24, 0.03], ...
    "String", "low cost per iteration; noisy trajectory", "EdgeColor", "none", ...
    "HorizontalAlignment", "center", "FontSize", 10, "Color", colors{3});

exportgraphics(gcf, outputFile, "Resolution", 180);
if nargout == 0
    clear outputFile;
end
end

function draw_box(x, y, width, height, label, backgroundColor)
annotation("textbox", [x, y, width, height], "String", label, ...
    "HorizontalAlignment", "center", "VerticalAlignment", "middle", ...
    "FontSize", 10, "BackgroundColor", backgroundColor, ...
    "EdgeColor", [0.35, 0.35, 0.35], "LineWidth", 0.9);
end

function add_arrow(startPoint, endPoint)
annotation("arrow", [startPoint(1), endPoint(1)], ...
    [startPoint(2), endPoint(2)], "LineWidth", 1.1, ...
    "Color", [0.25, 0.25, 0.25]);
end
