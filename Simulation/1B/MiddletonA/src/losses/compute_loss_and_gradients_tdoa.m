function [loss, gradients] = compute_loss_and_gradients_tdoa(net, X_norm, X_raw, ...
    Y_pred, Y_true, activations, params, train_params)
% Compute adaptive PINN loss and gradients.
    batch_size = size(X_norm, 1);
    turbidity_batch = X_raw(:, 11);

    distance_threshold = 2.0;

    v_measured = X_raw(:, 5);
    theta_cam = atan(v_measured / params.f_V);
    depth_measured = X_raw(:, 10);

    den = max(abs(sin(theta_cam)), 0.05);
    d_est = abs((depth_measured - params.depth) ./ den);
    d_est = min(max(d_est, 0.5), 50);

    is_near = d_est <= distance_threshold;

    optical_attenuation = exp(-params.turbidity_coefficient .* turbidity_batch .* d_est / 50);
    optical_attenuation = max(optical_attenuation, 0.1);
    optical_reliability = mean(optical_attenuation);

    tdoa_scale = train_params.tdoa_loss_scale;

    adaptive_lambda_tdoa  = (train_params.lambda_tdoa * tdoa_scale) .* ...
                            (1 + 0.5 * (1 - optical_attenuation));
    adaptive_lambda_depth = train_params.lambda_depth * ones(batch_size, 1);

    adaptive_lambda_tdoa(is_near)  = train_params.lambda_tdoa * tdoa_scale * 0.02;
    adaptive_lambda_depth(is_near) = train_params.lambda_depth * 20.0;

    adaptive_lambda_camera = train_params.lambda_camera * optical_reliability;
    adaptive_lambda_detector = train_params.lambda_detector * optical_reliability;
    adaptive_lambda_consistency = train_params.lambda_consistency * optical_reliability;

    diff_data = Y_pred - Y_true;
    weighted_diff = diff_data .* train_params.output_weights;
    loss_data = sum(sum(weighted_diff.^2)) / batch_size;

    d_pred = Y_pred(:, 1);
    psi_pred = Y_pred(:, 2);
    theta_pred = Y_pred(:, 3);

    x_pred = d_pred .* cos(theta_pred) .* cos(psi_pred);
    y_pred = d_pred .* cos(theta_pred) .* sin(psi_pred);
    z_pred = d_pred .* sin(theta_pred);

    loss_tdoa = 0;
    for i = 1:batch_size
        pos_pred_i = [x_pred(i); y_pred(i); z_pred(i)];

        r1 = norm(pos_pred_i - params.hydrophone_positions(1, :)');
        r2 = norm(pos_pred_i - params.hydrophone_positions(2, :)');
        r3 = norm(pos_pred_i - params.hydrophone_positions(3, :)');
        r4 = norm(pos_pred_i - params.hydrophone_positions(4, :)');

        tau_21 = X_raw(i, 1);
        tau_31 = X_raw(i, 2);
        tau_41 = X_raw(i, 3);

        residual_2 = (r2 - r1) - params.c_water * tau_21;
        residual_3 = (r3 - r1) - params.c_water * tau_31;
        residual_4 = (r4 - r1) - params.c_water * tau_41;

        loss_tdoa = loss_tdoa + adaptive_lambda_tdoa(i) * ...
                    (residual_2^2 + residual_3^2 + residual_4^2);
    end
    loss_tdoa = loss_tdoa / batch_size;

    loss_camera = 0;
    u_measured = X_raw(:, 4);
    psi_from_camera = atan(u_measured / params.f_U);
    theta_from_camera = atan(v_measured / params.f_V);

    for i = 1:batch_size
        psi_residual = psi_pred(i) - psi_from_camera(i);
        theta_residual = theta_pred(i) - theta_from_camera(i);
        sample_optical_weight = optical_attenuation(i);

        loss_camera = loss_camera + sample_optical_weight * ...
                      (psi_residual^2 + theta_residual^2);
    end
    loss_camera = adaptive_lambda_camera * loss_camera / batch_size;

    loss_detector = 0;
    valid_count = 0;

    for i = 1:batch_size
        try
            U_I = X_raw(i, 6);
            U_II = X_raw(i, 7);
            U_III = X_raw(i, 8);
            U_IV = X_raw(i, 9);

            delta_m0 = compute_normalized_offset(U_I, U_II, U_III, U_IV, 'horizontal');
            delta_n0 = compute_normalized_offset(U_I, U_II, U_III, U_IV, 'vertical');

            delta_m = apply_polynomial_fit(delta_m0);
            delta_n = apply_polynomial_fit(delta_n0);

            psi_detector = delta_to_psi_paper_eq1(delta_m, params);
            theta_detector = delta_to_psi_paper_eq1(delta_n, params);

            if ~isnan(psi_detector) && ~isnan(theta_detector)
                sample_optical_weight = optical_attenuation(i);
                loss_detector = loss_detector + sample_optical_weight * ...
                    ((psi_pred(i) - psi_detector)^2 + (theta_pred(i) - theta_detector)^2);
                valid_count = valid_count + 1;
            end
        catch
            continue;
        end
    end

    if valid_count > 0
        loss_detector = adaptive_lambda_detector * loss_detector / batch_size;
    end

    loss_depth = 0;
    delta_h_meas = depth_measured - params.depth;

    for i = 1:batch_size
        depth_from_spherical = d_pred(i) * sin(theta_pred(i));
        depth_residual = delta_h_meas(i) - depth_from_spherical;
        loss_depth = loss_depth + adaptive_lambda_depth(i) * depth_residual^2;
    end
    loss_depth = loss_depth / batch_size;

    loss_consistency = 0;
    consistency_valid_count = 0;

    for i = 1:batch_size
        try
            U_I = X_raw(i, 6);
            U_II = X_raw(i, 7);
            U_III = X_raw(i, 8);
            U_IV = X_raw(i, 9);

            delta_m0 = compute_normalized_offset(U_I, U_II, U_III, U_IV, 'horizontal');
            delta_n0 = compute_normalized_offset(U_I, U_II, U_III, U_IV, 'vertical');

            delta_m = apply_polynomial_fit(delta_m0);
            delta_n = apply_polynomial_fit(delta_n0);

            psi_detector = delta_to_psi_paper_eq1(delta_m, params);
            theta_detector = delta_to_psi_paper_eq1(delta_n, params);

            if ~isnan(psi_detector) && ~isnan(theta_detector)
                psi_consistency = psi_from_camera(i) - psi_detector;
                theta_consistency = theta_from_camera(i) - theta_detector;

                sample_optical_weight = optical_attenuation(i);
                loss_consistency = loss_consistency + sample_optical_weight * ...
                    (psi_consistency^2 + theta_consistency^2);
                consistency_valid_count = consistency_valid_count + 1;
            end
        catch
            continue;
        end
    end

    if consistency_valid_count > 0
        loss_consistency = adaptive_lambda_consistency * loss_consistency / batch_size;
    end

    loss.data = loss_data;
    loss.tdoa = loss_tdoa;
    loss.camera = loss_camera;
    loss.detector = loss_detector;
    loss.depth = loss_depth;
    loss.consistency = loss_consistency;

    loss.total = loss_data + loss_tdoa + loss_camera + loss_detector + ...
                 loss_depth + loss_consistency;

    gradients = compute_gradients_tdoa_adaptive( ...
        net, X_norm, X_raw, Y_pred, Y_true, activations, ...
        params, train_params, ...
        delta_h_meas, psi_from_camera, theta_from_camera, ...
        adaptive_lambda_tdoa, adaptive_lambda_camera, ...
        adaptive_lambda_detector, adaptive_lambda_depth, ...
        adaptive_lambda_consistency, optical_attenuation);
end
