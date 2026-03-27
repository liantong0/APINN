function [X_raw, Y_true, turbidity_profile, d_all] = generate_step_turbidity_test_data(cfg, params)
% Generate step-turbidity test data for smoothness evaluation.
    d_all = linspace(cfg.d_min, cfg.d_max, cfg.num_points)';
    psi_all = cfg.psi0 * ones(cfg.num_points, 1);
    theta_all = cfg.theta0 * ones(cfg.num_points, 1);

    Y_true = [d_all, psi_all, theta_all];
    X_raw = zeros(cfg.num_points, 11);


    turbidity_profile = zeros(cfg.num_points, 1);
    for i = 1:cfg.num_points
        d = d_all(i);
        if d < cfg.boundaries(1)
            turbidity_profile(i) = cfg.turbidity_levels(1);
        elseif d < cfg.boundaries(2)
            turbidity_profile(i) = cfg.turbidity_levels(2);
        else
            turbidity_profile(i) = cfg.turbidity_levels(3);
        end
    end

    for i = 1:cfg.num_points
        d = d_all(i);
        psi = psi_all(i);
        theta = theta_all(i);
        turb = turbidity_profile(i);


        x_target = d * cos(psi) * cos(theta);
        y_target = d * sin(psi) * cos(theta);
        z_target = d * sin(theta);
        pos_target = [x_target; y_target; z_target];

        tdoas_true = compute_tdoas_from_position(pos_target, params.hydrophone_positions, params.c_water);


        if d <= 0.5
            baseline_scale = 0.001;
            normalized_dist = (d - 0.1) / (0.5 - 0.1);
            distance_scale = baseline_scale * (1 + 4 * max(normalized_dist,0));
        elseif d <= 2.0
            baseline_scale = 0.005;
            normalized_dist = (d - 0.5) / (2.0 - 0.5);
            distance_scale = baseline_scale + normalized_dist^1.5 * (0.1 - baseline_scale);
        elseif d <= 10
            normalized_dist = (d - 2.0) / (10 - 2.0);
            distance_scale = 0.1 + normalized_dist^2 * 0.3;
        else
            distance_scale = d / 15;
        end


        optical_attenuation = exp(-params.turbidity_coefficient * turb * d / 50);
        optical_attenuation = max(optical_attenuation, 0.1);

        signal_power_tdoa = mean(tdoas_true.^2);
        sigma_tdoa = sqrt(signal_power_tdoa / 10^(params.SNR_tdoa/10)) * distance_scale;
        X_raw(i,1:3) = tdoas_true' + randn(1,3) * sigma_tdoa;

        u_true = params.f_U * tan(psi);
        v_true = params.f_V * tan(theta);

        effective_SNR_camera = params.SNR_camera - 10 * log10(1/optical_attenuation);
        effective_SNR_camera = max(effective_SNR_camera, 5);

        sigma_camera_u = sqrt(max(u_true^2,1e-8) / 10^(effective_SNR_camera/10)) * distance_scale;
        sigma_camera_v = sqrt(max(v_true^2,1e-8) / 10^(effective_SNR_camera/10)) * distance_scale;

        X_raw(i,4) = u_true + randn * sigma_camera_u;
        X_raw(i,5) = v_true + randn * sigma_camera_v;

        delta_m = psi_to_delta_paper_eq1(psi, params);
        delta_n = psi_to_delta_paper_eq1(theta, params);
        delta_m0 = inverse_polynomial_fit(delta_m);
        delta_n0 = inverse_polynomial_fit(delta_n);
        [U_I, U_II, U_III, U_IV] = generate_quadrant_voltages(delta_m0, delta_n0, params);

        U_voltages = [U_I, U_II, U_III, U_IV];
        signal_power_voltage = mean(U_voltages.^2);

        effective_SNR_detector = params.SNR_detector - 10 * log10(1/optical_attenuation);
        effective_SNR_detector = max(effective_SNR_detector, 5);

        sigma_voltage = sqrt(max(signal_power_voltage,1e-8) / 10^(effective_SNR_detector/10)) * distance_scale;

        X_raw(i,6) = max(U_I   + randn * sigma_voltage, 0);
        X_raw(i,7) = max(U_II  + randn * sigma_voltage, 0);
        X_raw(i,8) = max(U_III + randn * sigma_voltage, 0);
        X_raw(i,9) = max(U_IV  + randn * sigma_voltage, 0);

        depth_true = params.depth + z_target;
        X_raw(i,10) = depth_true;

        X_raw(i,11) = turb;
    end
end

