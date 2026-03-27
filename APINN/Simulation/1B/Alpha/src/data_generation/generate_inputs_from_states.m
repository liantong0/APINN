function X_raw = generate_inputs_from_states(Y_true, traj_params, params)
% Generate raw sensor inputs from ground-truth trajectory states.
    N = size(Y_true,1);
    X_raw = zeros(N, 11);

    for i = 1:N
        d = Y_true(i,1);
        psi = Y_true(i,2);
        theta = Y_true(i,3);

        if traj_params.turbidity_mode == "fixed"
            turbidity_sample = traj_params.turbidity_value;
        else
            turbidity_sample = 2 + 18*rand;
        end
        X_raw(i,11) = turbidity_sample;

        distance_scale = compute_distance_scale(d);

        optical_attenuation = exp(-params.turbidity_coefficient * turbidity_sample * d / 50);
        optical_attenuation = max(optical_attenuation, 0.1);

        % True position
        x = d * cos(psi) * cos(theta);
        y = d * sin(psi) * cos(theta);
        z = d * sin(theta);
        pos = [x; y; z];

        % 1) TDOA
        tdoas_true = compute_tdoas_from_position(pos, params.hydrophone_positions, params.c_water);
        signal_power_tdoa = mean(tdoas_true.^2);
        sigma_tdoa = sqrt(signal_power_tdoa / 10^(params.SNR_tdoa/10)) * distance_scale;
        n_tdoa = generate_nongaussian_noise([1,3], sigma_tdoa, params, 'tdoa');
        X_raw(i,1:3) = tdoas_true' + n_tdoa;

        % 2) Camera
        u_true = params.f_U * tan(psi);
        v_true = params.f_V * tan(theta);

        effective_SNR_camera = params.SNR_camera - 10*log10(1/optical_attenuation);
        effective_SNR_camera = max(effective_SNR_camera, 5);

        sigma_camera_u = sqrt(max(u_true^2,1e-12) / 10^(effective_SNR_camera/10)) * distance_scale;
        sigma_camera_v = sqrt(max(v_true^2,1e-12) / 10^(effective_SNR_camera/10)) * distance_scale;

        n_u = generate_nongaussian_noise([1,1], sigma_camera_u, params, 'camera');
        n_v = generate_nongaussian_noise([1,1], sigma_camera_v, params, 'camera');
        X_raw(i,4) = u_true + n_u;
        X_raw(i,5) = v_true + n_v;

        % 3) Quadrant detector
        delta_m = psi_to_delta_paper_eq1(psi, params);
        delta_n = psi_to_delta_paper_eq1(theta, params);

        delta_m0 = inverse_polynomial_fit(delta_m);
        delta_n0 = inverse_polynomial_fit(delta_n);

        [U_I, U_II, U_III, U_IV] = generate_quadrant_voltages(delta_m0, delta_n0, params);
        U_voltages = [U_I,U_II,U_III,U_IV];
        signal_power_voltage = mean(U_voltages.^2);

        effective_SNR_detector = params.SNR_detector - 10*log10(1/optical_attenuation);
        effective_SNR_detector = max(effective_SNR_detector, 5);

        sigma_voltage = sqrt(max(signal_power_voltage,1e-12) / 10^(effective_SNR_detector/10)) * distance_scale;

        n_vlt = generate_nongaussian_noise([1,4], sigma_voltage, params, 'detector');
        X_raw(i,6) = max(U_I   + n_vlt(1), 0);
        X_raw(i,7) = max(U_II  + n_vlt(2), 0);
        X_raw(i,8) = max(U_III + n_vlt(3), 0);
        X_raw(i,9) = max(U_IV  + n_vlt(4), 0);

        % 4) Depth sensor
        depth_true = params.depth + z;
        X_raw(i,10) = depth_true;
    end
end
