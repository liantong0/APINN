function plot_trajectory_comparison_journal( ...
    traj, pos_true, ...
    pos_chan_raw, pos_data_raw, pos_fixed_raw, pos_apinn_raw, ...
    pos_chan_sm,  pos_data_sm,  pos_fixed_sm,  pos_apinn_sm, ...
    err_chan, err_data, err_fixed, err_apinn, ...
    seg_id)
% Plot journal-style trajectory comparison figures for 3D paths and error curves.

    % Colors
    c_true   = [0 0 0];
    c_chan   = [0.85, 0.325, 0.098];
    c_data   = [0.466, 0.674, 0.188];
    c_fixed  = [0.494, 0.184, 0.556];
    c_apinn  = [0, 0.447, 0.741];
    c_gray   = [0.35 0.35 0.35];

    idx12 = find(seg_id==2, 1, 'first');
    idx23 = find(seg_id==3, 1, 'first');

    set(0,'DefaultAxesFontName','Times New Roman');
    set(0,'DefaultTextFontName','Times New Roman');

    % Figure 1: 3D trajectory only
    figT = figure('Color','w','Position',[120 120 860 660],'Renderer','painters');
    ax1 = axes(figT, 'Position', [0.09 0.14 0.86 0.72]);
    hold(ax1,'on'); box(ax1,'on');

    hT = plot3(ax1, pos_true(:,1),    pos_true(:,2),    pos_true(:,3),    '--', ...
        'Color', c_true,  'LineWidth', 2.0, 'DisplayName','True');

    hC = plot3(ax1, pos_chan_sm(:,1), pos_chan_sm(:,2), pos_chan_sm(:,3), '-', ...
        'Color', c_chan,  'LineWidth', 2.3, 'DisplayName','Analytical');

    hD = plot3(ax1, pos_data_sm(:,1), pos_data_sm(:,2), pos_data_sm(:,3), '-', ...
        'Color', c_data,  'LineWidth', 2.3, 'DisplayName','Data-NN');

    hF = plot3(ax1, pos_fixed_sm(:,1),pos_fixed_sm(:,2),pos_fixed_sm(:,3),'-', ...
        'Color', c_fixed, 'LineWidth', 2.3, 'DisplayName','Fixed-PINN');

    hP = plot3(ax1, pos_apinn_sm(:,1),pos_apinn_sm(:,2),pos_apinn_sm(:,3),'-', ...
        'Color', c_apinn, 'LineWidth', 2.8, 'DisplayName','APINN');

    if ~isempty(idx12)
        plot3(ax1, pos_true(idx12,1),pos_true(idx12,2),pos_true(idx12,3), ...
            'o','MarkerSize',8,'MarkerFaceColor','y','MarkerEdgeColor','k', ...
            'LineWidth',1.0,'HandleVisibility','off');
    end
    if ~isempty(idx23)
        plot3(ax1, pos_true(idx23,1),pos_true(idx23,2),pos_true(idx23,3), ...
            'o','MarkerSize',8,'MarkerFaceColor','c','MarkerEdgeColor','k', ...
            'LineWidth',1.0,'HandleVisibility','off');
    end

    [hS3, hE3] = add_start_end_true_3d(ax1, pos_true, c_true);

    xlabel(ax1,'X (m)','FontWeight','bold','FontSize',16);
    ylabel(ax1,'Y (m)','FontWeight','bold','FontSize',16);
    zlabel(ax1,'Z (m)','FontWeight','bold','FontSize',16);

    x_all = [pos_true(:,1); pos_chan_sm(:,1); pos_data_sm(:,1); pos_fixed_sm(:,1); pos_apinn_sm(:,1)];
    x_pad = 1.0;
    xlim(ax1, [min(x_all)-x_pad, max(x_all)+x_pad]);

    ylim(ax1, [-2.5, 22.5]);
    zlim(ax1, [-34, 2]);

    view(ax1, 34, 18);

    grid(ax1,'on');
    ax1.GridAlpha = 0.10;
    ax1.MinorGridAlpha = 0.05;
    ax1.TickDir = 'out';
    ax1.LineWidth = 1.1;
    ax1.FontSize = 13;
    ax1.Clipping = 'off';

    lgd = legend(ax1, [hT hC hD hF hP hS3 hE3], ...
        'Location','northoutside', 'Orientation','horizontal');
    lgd.NumColumns = 4;
    lgd.Box = 'off';
    lgd.FontSize = 11;

    % Figure 2: error curve
    figE = figure('Color','w','Position',[140 140 820 520],'Renderer','painters');
    axE = axes(figE); hold(axE,'on'); box(axE,'on');

    d = traj.d;

    dmin_plot = 0;
    dmax_plot = 50;
    dq = linspace(dmin_plot, dmax_plot, 800)';

    d_anchor = min(d);
    d_taper  = 1.0;

    yC = fit_error_curve_with_anchor(d, err_chan,  dq, d_anchor, d_taper);
    yD = fit_error_curve_with_anchor(d, err_data,  dq, d_anchor, d_taper);
    yF = fit_error_curve_with_anchor(d, err_fixed, dq, d_anchor, d_taper);
    yP = fit_error_curve_with_anchor(d, err_apinn, dq, d_anchor, d_taper);

    plot(axE, dq, yC, '-', 'Color', c_chan,  'LineWidth', 2.6, 'DisplayName','Analytical');
    plot(axE, dq, yD, '-', 'Color', c_data,  'LineWidth', 2.6, 'DisplayName','Data-NN');
    plot(axE, dq, yF, '-', 'Color', c_fixed, 'LineWidth', 2.6, 'DisplayName','Fixed-PINN');
    plot(axE, dq, yP, '-', 'Color', c_apinn, 'LineWidth', 3.0, 'DisplayName','APINN');

    xlim(axE, [0.1 50]);
    set(axE,'XDir','reverse');

    xline(axE, 10,'--','Color',c_gray,'LineWidth',1.2,'HandleVisibility','off');
    xline(axE, 3, '--','Color',c_gray,'LineWidth',1.2,'HandleVisibility','off');

    xlabel(axE,'True Distance d (m)','FontWeight','bold');
    ylabel(axE,'3D Position Error (m)','FontWeight','bold');

    grid(axE,'on');
    axE.GridAlpha = 0.10;
    axE.MinorGridAlpha = 0.05;
    axE.TickDir = 'out';
    axE.LineWidth = 1.1;
    axE.FontSize = 14;

    lgdE = legend(axE, 'Location','northoutside', 'Orientation','horizontal');
    lgdE.NumColumns = 4;
    lgdE.Box = 'off';
    lgdE.FontSize = 13;

    fprintf('Exported:\n');
    fprintf('  Fig_Traj_Comparison_3DOnly.pdf/png\n');
    fprintf('  Fig_Traj_Error_vs_Distance.pdf/png\n');
end
