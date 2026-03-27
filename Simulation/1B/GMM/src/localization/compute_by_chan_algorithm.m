function Y_result = compute_by_chan_algorithm(X_raw, params, train_params)
% Compute hybrid localization results using distance-aware rule switching.
    num_samples = size(X_raw, 1);
    Y_result = zeros(num_samples, 3);
    distance_threshold = 2.0;

    near_count = 0;
    far_count = 0;

    for i = 1:num_samples
        % 1. Rough distance from depth and angle
        v = X_raw(i, 5);
        theta_camera = atan(v / params.f_V);
        depth_measured = X_raw(i, 10);

        if abs(sin(theta_camera)) > 0.05
            d_rough = abs((depth_measured - params.depth) / sin(theta_camera));
            d_rough = max(0.5, min(d_rough, 50));
        else
            d_rough = 25;
        end

        % 2. Select method by rough distance
        if d_rough <= distance_threshold
            near_count = near_count + 1;

            % Camera angle
            u = X_raw(i, 4);
            psi_camera = atan(u / params.f_U);
            theta_camera = atan(v / params.f_V);

            % Detector angle
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

                if isnan(psi_detector) || isnan(theta_detector)
                    use_detector = false;
                else
                    use_detector = true;
                end
            catch
                use_detector = false;
            end

            % Angle fusion
            if use_detector
                psi_final = 0.6 * psi_camera + 0.4 * psi_detector;
                theta_final = 0.6 * theta_camera + 0.4 * theta_detector;
            else
                psi_final = psi_camera;
                theta_final = theta_camera;
            end

            % Distance from depth
            if abs(sin(theta_final)) > 0.05
                d_final = abs((depth_measured - params.depth) / sin(theta_final));
                d_final = max(0.5, min(d_final, 50));
            else
                d_final = d_rough;
            end

            Y_result(i, :) = [d_final, psi_final, theta_final];

        else
            far_count = far_count + 1;

            % Chan algorithm
            tdoas = X_raw(i, 1:3)';

            if any(isnan(tdoas)) || any(isinf(tdoas))
                use_chan = false;
                d_chan = 0;
                psi_chan = 0;
                theta_chan = 0;
            else
                try
                    pos_chan = chan_algorithm(tdoas, params.hydrophone_positions, params.c_water);

                    if any(isnan(pos_chan)) || any(isinf(pos_chan)) || norm(pos_chan) > 100
                        use_chan = false;
                        d_chan = 0;
                        psi_chan = 0;
                        theta_chan = 0;
                    else
                        use_chan = true;
                        d_chan = norm(pos_chan);
                        if d_chan > 0.1
                            psi_chan = atan2(pos_chan(2), pos_chan(1));
                            theta_chan = asin(max(min(pos_chan(3) / d_chan, 1), -1));
                        else
                            psi_chan = 0;
                            theta_chan = 0;
                        end
                    end
                catch
                    use_chan = false;
                    d_chan = 0;
                    psi_chan = 0;
                    theta_chan = 0;
                end
            end

            % Camera angle
            u = X_raw(i, 4);
            v = X_raw(i, 5);
            psi_camera = atan(u / params.f_U);
            theta_camera = atan(v / params.f_V);

            % Distance from camera
            if abs(sin(theta_camera)) > 0.01
                d_camera = (depth_measured - params.depth) / sin(theta_camera);
                d_camera = max(0.5, min(abs(d_camera), 50));
            else
                d_camera = 25;
            end

            % Detector angle
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

                if isnan(psi_detector) || isnan(theta_detector) || ...
                   isinf(psi_detector) || isinf(theta_detector)
                    use_detector = false;
                else
                    use_detector = true;
                end
            catch
                use_detector = false;
                psi_detector = psi_camera;
                theta_detector = theta_camera;
            end

            % Fusion strategy
            if use_chan && use_detector
                weight_chan = 0.5;
                weight_camera = 7;
                weight_detector = 1;
                total_weight = weight_chan + weight_camera + weight_detector;

                psi_final = (weight_chan * psi_chan + weight_camera * psi_camera + weight_detector * psi_detector) / total_weight;
                theta_final = (weight_chan * theta_chan + weight_camera * theta_camera + weight_detector * theta_detector) / total_weight;
                d_final = (weight_chan * d_chan + weight_camera * d_camera) / (weight_chan + weight_camera);
            elseif use_chan
                weight_chan = 0.3;
                weight_camera = 0.7;
                psi_final = (weight_chan * psi_chan + weight_camera * psi_camera) / (weight_chan + weight_camera);
                theta_final = (weight_chan * theta_chan + weight_camera * theta_camera) / (weight_chan + weight_camera);
                d_final = (weight_chan * d_chan + weight_camera * d_camera) / (weight_chan + weight_camera);
            else
                psi_final = psi_camera;
                theta_final = theta_camera;
                d_final = d_camera;
            end

            Y_result(i, :) = [d_final, psi_final, theta_final];
        end
    end

    fprintf('[Hybrid localization] Near range (<= %.1fm): %d samples | Far range: %d samples\n', ...
            distance_threshold, near_count, far_count);
end
