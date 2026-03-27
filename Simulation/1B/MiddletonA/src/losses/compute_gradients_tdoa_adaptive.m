function gradients = compute_gradients_tdoa_adaptive(net, X_norm, X_raw, Y_pred, Y_true, activations, ...
    params, train_params, delta_h_meas, psi_from_camera, theta_from_camera, ...
    lambda_tdoa, lambda_camera, lambda_detector, lambda_depth, lambda_consistency, optical_attenuation)
% Compute gradients for the adaptive PINN model.
    batch_size = size(X_norm, 1);
    num_layers = length(net.W);
    gradients = struct('W', {cell(num_layers, 1)}, 'b', {cell(num_layers, 1)});

    dL_dy = zeros(batch_size, 3);

    d_pred = Y_pred(:, 1);
    psi_pred = Y_pred(:, 2);
    theta_pred = Y_pred(:, 3);

    output_weights = train_params.output_weights;
    weighted_diff = (Y_pred - Y_true) .* (output_weights.^2);
    dL_dy = dL_dy + 2 * weighted_diff / batch_size;

    for i = 1:batch_size
        d = d_pred(i);
        psi = psi_pred(i);
        theta = theta_pred(i);

        x = d * cos(theta) * cos(psi);
        y = d * cos(theta) * sin(psi);
        z = d * sin(theta);
        pos_i = [x; y; z];

        J_pos = [
            cos(theta) * cos(psi), -d * cos(theta) * sin(psi), -d * sin(theta) * cos(psi);
            cos(theta) * sin(psi),  d * cos(theta) * cos(psi), -d * sin(theta) * sin(psi);
            sin(theta),             0,                          d * cos(theta)
        ];

        r1 = norm(pos_i - params.hydrophone_positions(1, :)');
        r2 = norm(pos_i - params.hydrophone_positions(2, :)');
        r3 = norm(pos_i - params.hydrophone_positions(3, :)');
        r4 = norm(pos_i - params.hydrophone_positions(4, :)');

        tau_measured = X_raw(i, 1:3)';

        residuals = [
            (r2 - r1) - params.c_water * tau_measured(1);
            (r3 - r1) - params.c_water * tau_measured(2);
            (r4 - r1) - params.c_water * tau_measured(3)
        ];

        grad_dist = zeros(3, 3);
        for j = 2:4
            h_j = params.hydrophone_positions(j, :)';
            h_0 = params.hydrophone_positions(1, :)';

            r_j = norm(pos_i - h_j);
            r_0 = norm(pos_i - h_0);

            if r_j > 1e-6 && r_0 > 1e-6
                grad_dist(j - 1, :) = ((pos_i - h_j)' / r_j - (pos_i - h_0)' / r_0);
            end
        end

        grad_tdoa = lambda_tdoa(i) * J_pos' * grad_dist' * (2 * residuals) / batch_size;
        dL_dy(i, :) = dL_dy(i, :) + grad_tdoa';
    end

    for i = 1:batch_size
        psi_residual = psi_pred(i) - psi_from_camera(i);
        theta_residual = theta_pred(i) - theta_from_camera(i);
        sample_weight = optical_attenuation(i);

        dL_dy(i, 2) = dL_dy(i, 2) + 2 * lambda_camera * sample_weight * psi_residual / batch_size;
        dL_dy(i, 3) = dL_dy(i, 3) + 2 * lambda_camera * sample_weight * theta_residual / batch_size;
    end

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
                sample_weight = optical_attenuation(i);

                dL_dy(i, 2) = dL_dy(i, 2) + 2 * lambda_detector * sample_weight * ...
                              (psi_pred(i) - psi_detector) / batch_size;
                dL_dy(i, 3) = dL_dy(i, 3) + 2 * lambda_detector * sample_weight * ...
                              (theta_pred(i) - theta_detector) / batch_size;
            end
        catch
            continue;
        end
    end

    for i = 1:batch_size
        depth_residual = d_pred(i) * sin(theta_pred(i)) - delta_h_meas(i);

        dL_dy(i, 1) = dL_dy(i, 1) + 2 * lambda_depth(i) * ...
                      depth_residual * sin(theta_pred(i)) / batch_size;
        dL_dy(i, 3) = dL_dy(i, 3) + 2 * lambda_depth(i) * ...
                      depth_residual * d_pred(i) * cos(theta_pred(i)) / batch_size;
    end

    y_tilde = activations{num_layers + 1}';
    dy_dy_tilde = zeros(size(y_tilde));
    dy_dy_tilde(:, 1) = 25 * (1 - tanh(y_tilde(:, 1)).^2);
    dy_dy_tilde(:, 2) = (pi / 3) * (1 - tanh(y_tilde(:, 2)).^2);
    dy_dy_tilde(:, 3) = (pi / 4) * (1 - tanh(y_tilde(:, 3)).^2);

    delta = (dL_dy .* dy_dy_tilde)';
    gradients.W{num_layers} = delta * activations{num_layers}';
    gradients.b{num_layers} = sum(delta, 2);

    for i = (num_layers - 1):-1:1
        delta = net.W{i + 1}' * delta;
        delta = delta .* (activations{i + 1} > 0);
        gradients.W{i} = delta * activations{i}';
        gradients.b{i} = sum(delta, 2);
    end
end
