function [X_raw, Y_true] = generate_data_tdoa(num_samples, params, min_distance)
% Generate synthetic multi-sensor training and test data for TDOA localization.
    if nargin < 3
        min_distance = 0.1;
    end

    if min_distance >= 2.0
        n_near = round(0.4 * num_samples);
        n_far = num_samples - n_near;
        d = [min_distance + (10.0-min_distance) * rand(n_near, 1);
             10.0 + 40 * rand(n_far, 1)];
    else

        segment1_end = 2.0;
        n_ultra_near = round(0.35 * num_samples);  % ?
        n_near = round(0.25 * num_samples);        % 
        n_far = num_samples - n_ultra_near - n_near;
        
        d = [min_distance + (segment1_end - min_distance) * rand(n_ultra_near, 1);   % min_distance?m
             2.0 + 8.0 * rand(n_near, 1);          % 2?0m
             10.0 + 40 * rand(n_far, 1)];          % 10?0m
    end
    d = d(randperm(num_samples));
    

    psi = (2*rand(num_samples, 1) - 1) * pi/3;
    theta = (2*rand(num_samples, 1) - 1) * pi/4;
    
    Y_true = [d, psi, theta];
    
    X_raw = zeros(num_samples, 11);
    
    for i = 1:num_samples
    turbidity_sample = 2 + 18 * rand;
    X_raw(i, 11) = turbidity_sample;
    

    if d(i) <= 0.5
        baseline_scale = 0.001;  % 0.1m
        normalized_dist = (d(i) - 0.1) / (0.5 - 0.1);
        distance_scale = baseline_scale * (1 + 4 * normalized_dist);  % ?.005
        noise_enabled = true;
        
    elseif d(i) <= 2.0
        baseline_scale = 0.005;  % 0.5m?
        normalized_dist = (d(i) - 0.5) / (2.0 - 0.5);
        distance_scale = baseline_scale + normalized_dist^1.5 * (0.1 - baseline_scale);
        noise_enabled = true;
        
    elseif d(i) <= 10
        normalized_dist = (d(i) - 2.0) / (10 - 2.0);
        distance_scale = 0.1 + normalized_dist^2 * 0.3;
        noise_enabled = true;
        
    else
        distance_scale = d(i) / 12;
        noise_enabled = true;
    end
    

    if noise_enabled
        optical_attenuation = exp(-params.turbidity_coefficient * turbidity_sample * d(i) / 50);
        optical_attenuation = max(optical_attenuation, 0.1);
    else
        optical_attenuation = 1.0;
    end

    x_target = d(i) * cos(psi(i)) * cos(theta(i));
    y_target = d(i) * sin(psi(i)) * cos(theta(i));
    z_target = d(i) * sin(theta(i));
    pos_target = [x_target; y_target; z_target];
    
    tdoas_true = compute_tdoas_from_position(pos_target, params.hydrophone_positions, params.c_water);
    
    if noise_enabled
        signal_power_tdoa = mean(tdoas_true.^2);
        sigma_tdoa = sqrt(signal_power_tdoa / 10^(params.SNR_tdoa/10)) * distance_scale;
        X_raw(i, 1:3) = tdoas_true' + randn(1, 3) * sigma_tdoa;
    else
        X_raw(i, 1:3) = tdoas_true';  % ?
    end
    
    u_true = params.f_U * tan(psi(i));
    v_true = params.f_V * tan(theta(i));
    
    if noise_enabled
        signal_power_camera_u = u_true^2;
        signal_power_camera_v = v_true^2;
        
        effective_SNR_camera = params.SNR_camera - 10 * log10(1/optical_attenuation);
        effective_SNR_camera = max(effective_SNR_camera, 5);
        
        sigma_camera_u = sqrt(signal_power_camera_u / 10^(effective_SNR_camera/10)) * distance_scale;
        sigma_camera_v = sqrt(signal_power_camera_v / 10^(effective_SNR_camera/10)) * distance_scale;
        
        X_raw(i, 4) = u_true + randn * sigma_camera_u;
        X_raw(i, 5) = v_true + randn * sigma_camera_v;
    else
        X_raw(i, 4) = u_true;
        X_raw(i, 5) = v_true;
    end
    
    delta_m = psi_to_delta_paper_eq1(psi(i), params);
    delta_n = psi_to_delta_paper_eq1(theta(i), params);
    delta_m0 = inverse_polynomial_fit(delta_m);
    delta_n0 = inverse_polynomial_fit(delta_n);
    [U_I, U_II, U_III, U_IV] = generate_quadrant_voltages(delta_m0, delta_n0, params);
    
    if noise_enabled
        U_voltages = [U_I, U_II, U_III, U_IV];
        signal_power_voltage = mean(U_voltages.^2);
        
        effective_SNR_detector = params.SNR_detector - 10 * log10(1/optical_attenuation);
        effective_SNR_detector = max(effective_SNR_detector, 5);
        
        sigma_voltage = sqrt(signal_power_voltage / 10^(effective_SNR_detector/10)) * distance_scale;
        
        X_raw(i, 6) = max(U_I + randn * sigma_voltage, 0);
        X_raw(i, 7) = max(U_II + randn * sigma_voltage, 0);
        X_raw(i, 8) = max(U_III + randn * sigma_voltage, 0);
        X_raw(i, 9) = max(U_IV + randn * sigma_voltage, 0);
    else
        X_raw(i, 6:9) = [U_I, U_II, U_III, U_IV];
    end
    
    depth_true = params.depth + z_target;
        X_raw(i, 10) = depth_true;
end
end


