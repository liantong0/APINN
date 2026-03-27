function [loss, gradients] = compute_loss_and_gradients_tdoa_fixed(net, X_norm, X_raw, ...
                                                        Y_pred, Y_true, activations, ...
                                                        params, train_params)
% Compute fixed-weight PINN loss terms and gradients.
    batch_size = size(X_norm, 1);
    

    lambda_tdoa = train_params.lambda_tdoa;
    lambda_camera = train_params.lambda_camera;
    lambda_detector = train_params.lambda_detector;
    lambda_depth = train_params.lambda_depth;
    lambda_consistency = train_params.lambda_consistency;
    
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
        
        loss_tdoa = loss_tdoa + residual_2^2 + residual_3^2 + residual_4^2;
    end
    S_TDOA = 1e4;
    loss_tdoa = (lambda_tdoa * S_TDOA) * loss_tdoa / batch_size;
    
    loss_camera = 0;
    u_measured = X_raw(:, 4);
    v_measured = X_raw(:, 5);
    psi_from_camera = atan(u_measured / params.f_U);
    theta_from_camera = atan(v_measured / params.f_V);
    
    for i = 1:batch_size
        psi_residual = psi_pred(i) - psi_from_camera(i);
        theta_residual = theta_pred(i) - theta_from_camera(i);
        loss_camera = loss_camera + (psi_residual^2 + theta_residual^2);
    end
    loss_camera = lambda_camera * loss_camera / batch_size;
    
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
                loss_detector = loss_detector + ...
                    ((psi_pred(i) - psi_detector)^2 + (theta_pred(i) - theta_detector)^2);
                valid_count = valid_count + 1;
            end
        catch
            continue;
        end
    end
    
    if valid_count > 0
        loss_detector = lambda_detector * loss_detector / batch_size;
    end
    
    loss_depth = 0;
    depth_measured = X_raw(:, 10);
    delta_h_meas = depth_measured - params.depth;
    
    for i = 1:batch_size
        depth_from_spherical = d_pred(i) * sin(theta_pred(i));
        depth_residual = delta_h_meas(i) - depth_from_spherical;
        loss_depth = loss_depth + depth_residual^2;
    end
    loss_depth = lambda_depth * loss_depth / batch_size;
    
    loss_consistency = 0;
    psi_fused_batch   = nan(batch_size,1);
    theta_fused_batch = nan(batch_size,1);
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

            if isfinite(psi_detector) && isfinite(theta_detector)
                w_cam = 0.7;
                w_det = 0.3;

                psi_fused   = w_cam * psi_from_camera(i)   + w_det * psi_detector;
                theta_fused = w_cam * theta_from_camera(i) + w_det * theta_detector;

                psi_fused_batch(i)   = psi_fused;
                theta_fused_batch(i) = theta_fused;


                loss_consistency = loss_consistency + ...
                    ((psi_pred(i) - psi_fused)^2 + (theta_pred(i) - theta_fused)^2);

                consistency_valid_count = consistency_valid_count + 1;
            end
        catch
            continue;
        end
    end

    if consistency_valid_count > 0
        loss_consistency = lambda_consistency * loss_consistency / batch_size;
    else
        loss_consistency = 0;
    end
    
    loss.data = loss_data;
    loss.tdoa = loss_tdoa;
    loss.camera = loss_camera;
    loss.detector = loss_detector;
    loss.depth = loss_depth;
    loss.consistency = loss_consistency;
    
    loss.total = loss_data + loss_tdoa + loss_camera + loss_detector + ...
                 loss_depth + loss_consistency;
    
    gradients = compute_gradients_tdoa_fixed(...
        net, X_norm, X_raw, Y_pred, Y_true, activations, ...
        params, train_params, ...
        delta_h_meas, psi_from_camera, theta_from_camera);
end

