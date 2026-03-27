function X_norm = normalize_inputs_tdoa(X_raw, params)
% Normalize TDOA-based input features for neural network training.

    X_norm = zeros(size(X_raw));
    
    % 1-3. TDOA
    max_tdoa = 50 / params.c_water;
    X_norm(:, 1:3) = X_raw(:, 1:3) / max_tdoa;
    
    % 4-5. Camera
    X_norm(:, 4) = X_raw(:, 4) / params.u_scale;
    X_norm(:, 5) = X_raw(:, 5) / params.v_scale;
    
    % 6-9. Quadrant voltages
    X_norm(:, 6:9) = X_raw(:, 6:9);
    
    % 10. Depth
    X_norm(:, 10) = (X_raw(:, 10) - 10) / 50;
    
    % 11. Turbidity
    X_norm(:, 11) = X_raw(:, 11) / 20;
end
