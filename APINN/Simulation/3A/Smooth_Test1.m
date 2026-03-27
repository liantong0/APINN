clear; clc; close all;
setup_paths();

%% Load reproducible experiment state
repro_name = 'Smooth_test1_Repro_State.mat';
script_dir = fileparts(mfilename('fullpath'));
candidate_paths = {
    fullfile(script_dir, repro_name), ...
    fullfile(script_dir, 'data', repro_name), ...
    repro_name ...
};

repro_file = '';
for i = 1:numel(candidate_paths)
    if isfile(candidate_paths{i})
        repro_file = candidate_paths{i};
        break;
    end
end

if isempty(repro_file)
    repro_hits = dir(fullfile(pwd, '**', repro_name));
    if ~isempty(repro_hits)
        repro_file = fullfile(repro_hits(1).folder, repro_hits(1).name);
    end
end

if isempty(repro_file)
    checked = strjoin(candidate_paths, ', ');
    error('Reproducible state file not found: %s\nChecked: %s', repro_name, checked);
end

load(repro_file, ...
    'rng_state_before_init', ...
    'rng_state_at_training_start', ...
    'X_raw_train', 'Y_true_train', ...
    'X_raw_test', 'Y_true_test', ...
    'params', 'train_params', 'layers', ...
    'theta_eps_deg', 'theta_eps', ...
    'num_test', ...
    'net_pinn', 'adam_state_pinn', ...
    'net_pinn_fixed', 'adam_state_pinn_fixed', ...
    'net_data', 'adam_state_data');

fprintf('Loaded reproducible experiment state: %s\n', repro_file);

%% Restore saved RNG state at training start
rng(rng_state_at_training_start);
fprintf('Restored random state before training start.\n');

%% Data normalization
fprintf('\nNormalizing training data...\n');
X_norm_train = normalize_inputs_tdoa(X_raw_train, params);

fprintf('Normalizing test data...\n');
X_norm_test = normalize_inputs_tdoa(X_raw_test, params);

%% Validate refraction model
fprintf('\nValidating refraction model\n');
test_angles = [-20, -10, -5, 0, 5, 10, 20] * pi/180;

max_error = 0;
for i = 1:length(test_angles)
    psi_orig = test_angles(i);
    delta = psi_to_delta_paper_eq1(psi_orig, params);
    psi_recovered = delta_to_psi_paper_eq1(delta, params);
    err_tmp = abs(psi_recovered - psi_orig);
    max_error = max(max_error, err_tmp);

    fprintf('psi: %+6.2f deg -> Delta_m: %+8.4f mm -> psi: %+6.2f deg (error: %.2e rad)\n', ...
            rad2deg(psi_orig), delta, rad2deg(psi_recovered), err_tmp);
end

fprintf('Maximum round-trip error: %.2e rad (%.4f deg)\n\n', max_error, rad2deg(max_error));

if max_error > 1e-6
    warning('Refraction model validation failed!');
    return;
end
fprintf('Refraction model validation passed.\n\n');

%% Validate quadrant voltage conversion
fprintf('Validating quadrant voltage conversion chain\n');
test_psi = 10 * pi/180;
test_theta = 15 * pi/180;

delta_m_test = psi_to_delta_paper_eq1(test_psi, params);
delta_n_test = psi_to_delta_paper_eq1(test_theta, params);
delta_m0_test = inverse_polynomial_fit(delta_m_test);
delta_n0_test = inverse_polynomial_fit(delta_n_test);
[U_I, U_II, U_III, U_IV] = generate_quadrant_voltages(delta_m0_test, delta_n0_test, params);

delta_m0_recovered = compute_normalized_offset(U_I, U_II, U_III, U_IV, 'horizontal');
delta_n0_recovered = compute_normalized_offset(U_I, U_II, U_III, U_IV, 'vertical');
delta_m_recovered = apply_polynomial_fit(delta_m0_recovered);
delta_n_recovered = apply_polynomial_fit(delta_n0_recovered);
psi_recovered = delta_to_psi_paper_eq1(delta_m_recovered, params);
theta_recovered = delta_to_psi_paper_eq1(delta_n_recovered, params);

fprintf('Original angles: psi=%.2f deg, theta=%.2f deg\n', rad2deg(test_psi), rad2deg(test_theta));
fprintf('Quadrant voltages: U_I=%.4f, U_II=%.4f, U_III=%.4f, U_IV=%.4f\n', U_I, U_II, U_III, U_IV);
fprintf('Recovered angles: psi=%.2f deg, theta=%.2f deg\n', rad2deg(psi_recovered), rad2deg(theta_recovered));
fprintf('Angle errors: Delta_psi=%.4f deg, Delta_theta=%.4f deg\n\n', ...
    rad2deg(abs(psi_recovered-test_psi)), rad2deg(abs(theta_recovered-test_theta)));

%% Restore initial networks and retrain
fprintf('\nRestore from saved state and start APINN training\n');

loss_history_pinn = struct('total', [], 'data', [], 'tdoa', [], 'camera', [], 'detector', [], 'depth', []);
test_mae_history_pinn = struct('distance', [], 'azimuth', [], 'elevation', []);
position_error_history_pinn = [];
current_lr_pinn = train_params.learning_rate;

for epoch = 1:train_params.num_epochs
    if mod(epoch, 200) == 0 && epoch > 0
        current_lr_pinn = current_lr_pinn * 0.85;
    end

    idx = randperm(train_params.num_samples);
    X_norm_shuffled = X_norm_train(idx, :);
    Y_true_shuffled = Y_true_train(idx, :);
    X_raw_shuffled  = X_raw_train(idx, :);

    epoch_loss = struct('total', 0, 'data', 0, 'tdoa', 0, 'camera', 0, 'detector', 0, 'depth', 0);
    num_batches = ceil(train_params.num_samples / train_params.batch_size);

    epoch_camera_weights_sum = 0;
    epoch_detector_weights_sum = 0;
    epoch_samples_count = 0;

    for batch = 1:num_batches
        batch_start = (batch-1) * train_params.batch_size + 1;
        batch_end = min(batch * train_params.batch_size, train_params.num_samples);

        X_batch = X_norm_shuffled(batch_start:batch_end, :);
        Y_batch = Y_true_shuffled(batch_start:batch_end, :);
        X_raw_batch = X_raw_shuffled(batch_start:batch_end, :);

        [Y_pred, activations] = forward_pass(net_pinn, X_batch);

        [loss, gradients] = compute_loss_and_gradients_tdoa( ...
            net_pinn, X_batch, X_raw_batch, Y_pred, Y_batch, activations, ...
            params, train_params);

        [net_pinn, adam_state_pinn] = adam_update(net_pinn, gradients, adam_state_pinn, current_lr_pinn, ...
            train_params.beta1, train_params.beta2, train_params.epsilon, epoch);

        epoch_loss.total    = epoch_loss.total    + loss.total;
        epoch_loss.data     = epoch_loss.data     + loss.data;
        epoch_loss.tdoa     = epoch_loss.tdoa     + loss.tdoa;
        epoch_loss.camera   = epoch_loss.camera   + loss.camera;
        epoch_loss.detector = epoch_loss.detector + loss.detector;
        epoch_loss.depth    = epoch_loss.depth    + loss.depth;
    end

    epoch_loss.total    = epoch_loss.total    / num_batches;
    epoch_loss.data     = epoch_loss.data     / num_batches;
    epoch_loss.tdoa     = epoch_loss.tdoa     / num_batches;
    epoch_loss.camera   = epoch_loss.camera   / num_batches;
    epoch_loss.detector = epoch_loss.detector / num_batches;
    epoch_loss.depth    = epoch_loss.depth    / num_batches;

    if epoch_samples_count > 0
        epoch_avg_weight_camera = epoch_camera_weights_sum / epoch_samples_count;
        epoch_avg_weight_detector = epoch_detector_weights_sum / epoch_samples_count;
    else
        epoch_avg_weight_camera = NaN;
        epoch_avg_weight_detector = NaN;
    end

    loss_history_pinn.total(epoch)    = epoch_loss.total;
    loss_history_pinn.data(epoch)     = epoch_loss.data;
    loss_history_pinn.tdoa(epoch)     = epoch_loss.tdoa;
    loss_history_pinn.camera(epoch)   = epoch_loss.camera;
    loss_history_pinn.detector(epoch) = epoch_loss.detector;
    loss_history_pinn.depth(epoch)    = epoch_loss.depth;

    if mod(epoch, 10) == 0 || epoch == 1
        Y_test_pred_epoch = forward_pass(net_pinn, X_norm_test);
        errors_epoch = Y_test_pred_epoch - Y_true_test;

        mae_d = sum(abs(errors_epoch(:,1))) / size(errors_epoch, 1);
        mae_psi = sum(abs(errors_epoch(:,2))) / size(errors_epoch, 1);
        mae_theta = sum(abs(errors_epoch(:,3))) / size(errors_epoch, 1);

        test_mae_history_pinn.distance(end+1)  = mae_d;
        test_mae_history_pinn.azimuth(end+1)   = mae_psi;
        test_mae_history_pinn.elevation(end+1) = mae_theta;

        pos_true = [Y_true_test(:,1).*cos(Y_true_test(:,2)).*cos(Y_true_test(:,3)), ...
                    Y_true_test(:,1).*sin(Y_true_test(:,2)).*cos(Y_true_test(:,3)), ...
                    Y_true_test(:,1).*sin(Y_true_test(:,3))];
        pos_pred = [Y_test_pred_epoch(:,1).*cos(Y_test_pred_epoch(:,2)).*cos(Y_test_pred_epoch(:,3)), ...
                    Y_test_pred_epoch(:,1).*sin(Y_test_pred_epoch(:,2)).*cos(Y_test_pred_epoch(:,3)), ...
                    Y_test_pred_epoch(:,1).*sin(Y_test_pred_epoch(:,3))];
        pos_errors = sqrt(sum((pos_true - pos_pred).^2, 2));
        position_error_history_pinn(end+1) = sum(pos_errors) / length(pos_errors);
    end

    if mod(epoch, 500) == 0 || epoch == 1
        fprintf('[APINN] Epoch %d/%d\n', epoch, train_params.num_epochs);
        fprintf('       Total: %.6f | Data: %.6f\n', epoch_loss.total, epoch_loss.data);
        fprintf('       Physics: TDOA=%.6f, Camera=%.6f, Detector=%.6f, Depth=%.6f\n', ...
            epoch_loss.tdoa, epoch_loss.camera, epoch_loss.detector, epoch_loss.depth);
    end
end

fprintf('APINN retraining completed.\n\n');

%% Train fixed-weight PINN
fprintf('\nRestore from saved state and start fixed-weight PINN training\n');

loss_history_pinn_fixed = struct('total', [], 'data', [], 'tdoa', [], 'camera', [], 'detector', [], 'depth', []);
test_mae_history_pinn_fixed = struct('distance', [], 'azimuth', [], 'elevation', []);
position_error_history_pinn_fixed = [];
current_lr_pinn_fixed = train_params.learning_rate;

for epoch = 1:train_params.num_epochs
    if mod(epoch, 200) == 0 && epoch > 0
        current_lr_pinn_fixed = current_lr_pinn_fixed * 0.85;
    end

    idx = randperm(train_params.num_samples);
    X_norm_shuffled = X_norm_train(idx, :);
    Y_true_shuffled = Y_true_train(idx, :);
    X_raw_shuffled  = X_raw_train(idx, :);

    epoch_loss = struct('total', 0, 'data', 0, 'tdoa', 0, 'camera', 0, 'detector', 0, 'depth', 0);
    num_batches = ceil(train_params.num_samples / train_params.batch_size);

    for batch = 1:num_batches
        batch_start = (batch-1) * train_params.batch_size + 1;
        batch_end = min(batch * train_params.batch_size, train_params.num_samples);

        X_batch = X_norm_shuffled(batch_start:batch_end, :);
        Y_batch = Y_true_shuffled(batch_start:batch_end, :);
        X_raw_batch = X_raw_shuffled(batch_start:batch_end, :);

        [Y_pred, activations] = forward_pass(net_pinn_fixed, X_batch);

        [loss, gradients] = compute_loss_and_gradients_tdoa_fixed( ...
            net_pinn_fixed, X_batch, X_raw_batch, Y_pred, Y_batch, activations, ...
            params, train_params);

        [net_pinn_fixed, adam_state_pinn_fixed] = adam_update(net_pinn_fixed, gradients, adam_state_pinn_fixed, current_lr_pinn_fixed, ...
            train_params.beta1, train_params.beta2, train_params.epsilon, epoch);

        epoch_loss.total    = epoch_loss.total    + loss.total;
        epoch_loss.data     = epoch_loss.data     + loss.data;
        epoch_loss.tdoa     = epoch_loss.tdoa     + loss.tdoa;
        epoch_loss.camera   = epoch_loss.camera   + loss.camera;
        epoch_loss.detector = epoch_loss.detector + loss.detector;
        epoch_loss.depth    = epoch_loss.depth    + loss.depth;
    end

    epoch_loss.total    = epoch_loss.total    / num_batches;
    epoch_loss.data     = epoch_loss.data     / num_batches;
    epoch_loss.tdoa     = epoch_loss.tdoa     / num_batches;
    epoch_loss.camera   = epoch_loss.camera   / num_batches;
    epoch_loss.detector = epoch_loss.detector / num_batches;
    epoch_loss.depth    = epoch_loss.depth    / num_batches;

    loss_history_pinn_fixed.total(epoch)    = epoch_loss.total;
    loss_history_pinn_fixed.data(epoch)     = epoch_loss.data;
    loss_history_pinn_fixed.tdoa(epoch)     = epoch_loss.tdoa;
    loss_history_pinn_fixed.camera(epoch)   = epoch_loss.camera;
    loss_history_pinn_fixed.detector(epoch) = epoch_loss.detector;
    loss_history_pinn_fixed.depth(epoch)    = epoch_loss.depth;

    if mod(epoch, 10) == 0 || epoch == 1
        Y_test_pred_epoch = forward_pass(net_pinn_fixed, X_norm_test);
        errors_epoch = Y_test_pred_epoch - Y_true_test;

        mae_d = sum(abs(errors_epoch(:,1))) / size(errors_epoch, 1);
        mae_psi = sum(abs(errors_epoch(:,2))) / size(errors_epoch, 1);
        mae_theta = sum(abs(errors_epoch(:,3))) / size(errors_epoch, 1);

        test_mae_history_pinn_fixed.distance(end+1)  = mae_d;
        test_mae_history_pinn_fixed.azimuth(end+1)   = mae_psi;
        test_mae_history_pinn_fixed.elevation(end+1) = mae_theta;

        pos_true = [Y_true_test(:,1).*cos(Y_true_test(:,2)).*cos(Y_true_test(:,3)), ...
                    Y_true_test(:,1).*sin(Y_true_test(:,2)).*cos(Y_true_test(:,3)), ...
                    Y_true_test(:,1).*sin(Y_true_test(:,3))];
        pos_pred = [Y_test_pred_epoch(:,1).*cos(Y_test_pred_epoch(:,2)).*cos(Y_test_pred_epoch(:,3)), ...
                    Y_test_pred_epoch(:,1).*sin(Y_test_pred_epoch(:,2)).*cos(Y_test_pred_epoch(:,3)), ...
                    Y_test_pred_epoch(:,1).*sin(Y_test_pred_epoch(:,3))];
        pos_errors = sqrt(sum((pos_true - pos_pred).^2, 2));
        position_error_history_pinn_fixed(end+1) = sum(pos_errors) / length(pos_errors);
    end

    if mod(epoch, 500) == 0 || epoch == 1
        fprintf('[Fixed-PINN] Epoch %d/%d\n', epoch, train_params.num_epochs);
        fprintf('       Total: %.6f | Data: %.6f\n', epoch_loss.total, epoch_loss.data);
        fprintf('       Physics: TDOA=%.6f, Camera=%.6f, Detector=%.6f, Depth=%.6f\n', ...
            epoch_loss.tdoa, epoch_loss.camera, epoch_loss.detector, epoch_loss.depth);
    end
end

fprintf('Fixed-weight PINN retraining completed.\n\n');

%% Train purely data-driven network
fprintf('Restore from saved state and start purely data-driven neural network training\n');

loss_history_data = struct('total', [], 'data', []);
test_mae_history_data = struct('distance', [], 'azimuth', [], 'elevation', []);
position_error_history_data = [];
current_lr_data = train_params.learning_rate;
weight_decay = 0.05;

for epoch = 1:train_params.num_epochs
    if mod(epoch, 200) == 0 && epoch > 0
        current_lr_data = current_lr_data * 0.85;
    end

    idx = randperm(train_params.num_samples);
    X_norm_shuffled = X_norm_train(idx, :);
    Y_true_shuffled = Y_true_train(idx, :);

    epoch_loss = struct('total', 0, 'data', 0);
    num_batches = ceil(train_params.num_samples / train_params.batch_size);

    for batch = 1:num_batches
        batch_start = (batch-1) * train_params.batch_size + 1;
        batch_end = min(batch * train_params.batch_size, train_params.num_samples);

        X_batch = X_norm_shuffled(batch_start:batch_end, :);
        Y_batch = Y_true_shuffled(batch_start:batch_end, :);

        [Y_pred, activations] = forward_pass(net_data, X_batch);

        [loss, gradients] = compute_loss_data_driven(net_data, X_batch, Y_pred, Y_batch, ...
            activations, train_params, weight_decay);

        [net_data, adam_state_data] = adam_update(net_data, gradients, adam_state_data, current_lr_data, ...
            train_params.beta1, train_params.beta2, train_params.epsilon, epoch);

        epoch_loss.total = epoch_loss.total + loss.total;
        epoch_loss.data  = epoch_loss.data + loss.data;
    end

    epoch_loss.total = epoch_loss.total / num_batches;
    epoch_loss.data  = epoch_loss.data  / num_batches;

    loss_history_data.total(epoch) = epoch_loss.total;
    loss_history_data.data(epoch)  = epoch_loss.data;

    if mod(epoch, 10) == 0 || epoch == 1
        Y_test_pred_epoch = forward_pass(net_data, X_norm_test);
        errors_epoch = Y_test_pred_epoch - Y_true_test;

        mae_d = sum(abs(errors_epoch(:,1))) / size(errors_epoch, 1);
        mae_psi = sum(abs(errors_epoch(:,2))) / size(errors_epoch, 1);
        mae_theta = sum(abs(errors_epoch(:,3))) / size(errors_epoch, 1);

        test_mae_history_data.distance(end+1)  = mae_d;
        test_mae_history_data.azimuth(end+1)   = mae_psi;
        test_mae_history_data.elevation(end+1) = mae_theta;

        pos_true = [Y_true_test(:,1).*cos(Y_true_test(:,2)).*cos(Y_true_test(:,3)), ...
                    Y_true_test(:,1).*sin(Y_true_test(:,2)).*cos(Y_true_test(:,3)), ...
                    Y_true_test(:,1).*sin(Y_true_test(:,3))];
        pos_pred = [Y_test_pred_epoch(:,1).*cos(Y_test_pred_epoch(:,2)).*cos(Y_test_pred_epoch(:,3)), ...
                    Y_test_pred_epoch(:,1).*sin(Y_test_pred_epoch(:,2)).*cos(Y_test_pred_epoch(:,3)), ...
                    Y_test_pred_epoch(:,1).*sin(Y_test_pred_epoch(:,3))];
        pos_errors = sqrt(sum((pos_true - pos_pred).^2, 2));
        position_error_history_data(end+1) = sum(pos_errors) / length(pos_errors);
    end

    if mod(epoch, 500) == 0 || epoch == 1
        fprintf('[Data-driven] Epoch %d/%d - Loss: %.6f\n', ...
            epoch, train_params.num_epochs, epoch_loss.total);
    end
end

fprintf('Purely data-driven network retraining completed.\n\n');

%% Final evaluation
fprintf('\nEvaluating all models...\n');
Y_test_pred_pinn       = forward_pass(net_pinn, X_norm_test);
Y_test_pred_pinn_fixed = forward_pass(net_pinn_fixed, X_norm_test);
Y_test_pred_data       = forward_pass(net_data, X_norm_test);
Y_chan                 = compute_by_chan_algorithm(X_raw_test, params, train_params);

errors_pinn       = Y_test_pred_pinn       - Y_true_test;
errors_pinn_fixed = Y_test_pred_pinn_fixed - Y_true_test;
errors_data       = Y_test_pred_data       - Y_true_test;
errors_chan       = Y_chan                 - Y_true_test;

mae_pinn = [ ...
    sum(abs(errors_pinn(:,1))) / size(errors_pinn,1), ...
    sum(abs(errors_pinn(:,2))) / size(errors_pinn,1), ...
    sum(abs(errors_pinn(:,3))) / size(errors_pinn,1)];

mae_pinn_fixed = [ ...
    sum(abs(errors_pinn_fixed(:,1))) / size(errors_pinn_fixed,1), ...
    sum(abs(errors_pinn_fixed(:,2))) / size(errors_pinn_fixed,1), ...
    sum(abs(errors_pinn_fixed(:,3))) / size(errors_pinn_fixed,1)];

mae_data = [ ...
    sum(abs(errors_data(:,1))) / size(errors_data,1), ...
    sum(abs(errors_data(:,2))) / size(errors_data,1), ...
    sum(abs(errors_data(:,3))) / size(errors_data,1)];

mae_chan = [ ...
    sum(abs(errors_chan(:,1))) / size(errors_chan,1), ...
    sum(abs(errors_chan(:,2))) / size(errors_chan,1), ...
    sum(abs(errors_chan(:,3))) / size(errors_chan,1)];

fprintf('\nTest set performance comparison\n');
fprintf('APINN (adaptive weights):\n');
fprintf('  Distance MAE:  %.4f m\n', mae_pinn(1));
fprintf('  Azimuth MAE: %.2f deg\n', rad2deg(mae_pinn(2)));
fprintf('  Elevation MAE: %.2f deg\n', rad2deg(mae_pinn(3)));

fprintf('\nFixed-weight PINN:\n');
fprintf('  Distance MAE:  %.4f m\n', mae_pinn_fixed(1));
fprintf('  Azimuth MAE: %.2f deg\n', rad2deg(mae_pinn_fixed(2)));
fprintf('  Elevation MAE: %.2f deg\n', rad2deg(mae_pinn_fixed(3)));

fprintf('\nPurely data-driven neural network:\n');
fprintf('  Distance MAE:  %.4f m\n', mae_data(1));
fprintf('  Azimuth MAE: %.2f deg\n', rad2deg(mae_data(2)));
fprintf('  Elevation MAE: %.2f deg\n', rad2deg(mae_data(3)));

fprintf('\nChan algorithm + optical direction finding:\n');
fprintf('  Distance MAE:  %.4f m\n', mae_chan(1));
fprintf('  Azimuth MAE: %.2f deg\n', rad2deg(mae_chan(2)));
fprintf('  Elevation MAE: %.2f deg\n', rad2deg(mae_chan(3)));
fprintf('End of performance comparison\n');

%% Save full-distance test dataset (including four-method predictions)
fprintf('\nSaving test dataset (including four-method prediction outputs)...\n');

Full_Distance_Test_Dataset = array2table([X_raw_test, Y_true_test], ...
    'VariableNames', {'TDOA1_s','TDOA2_s','TDOA3_s', ...
                      'CameraU_pixel','CameraV_pixel', ...
                      'QuadrantUI_V','QuadrantUII_V','QuadrantUIII_V','QuadrantUIV_V', ...
                      'Depth_m','Turbidity_NTU', ...
                      'TrueDistance_m','TrueAzimuth_rad','TrueElevation_rad'});

Full_Distance_Test_Dataset.TrueAzimuth_deg   = rad2deg(Full_Distance_Test_Dataset.TrueAzimuth_rad);
Full_Distance_Test_Dataset.TrueElevation_deg = rad2deg(Full_Distance_Test_Dataset.TrueElevation_rad);

Full_Distance_Test_Dataset.APINN_Distance_m      = Y_test_pred_pinn(:,1);
Full_Distance_Test_Dataset.APINN_Azimuth_rad     = Y_test_pred_pinn(:,2);
Full_Distance_Test_Dataset.APINN_Elevation_rad   = Y_test_pred_pinn(:,3);

Full_Distance_Test_Dataset.FixedPINN_Distance_m    = Y_test_pred_pinn_fixed(:,1);
Full_Distance_Test_Dataset.FixedPINN_Azimuth_rad   = Y_test_pred_pinn_fixed(:,2);
Full_Distance_Test_Dataset.FixedPINN_Elevation_rad = Y_test_pred_pinn_fixed(:,3);

Full_Distance_Test_Dataset.DataNN_Distance_m    = Y_test_pred_data(:,1);
Full_Distance_Test_Dataset.DataNN_Azimuth_rad   = Y_test_pred_data(:,2);
Full_Distance_Test_Dataset.DataNN_Elevation_rad = Y_test_pred_data(:,3);

Full_Distance_Test_Dataset.Analytical_Distance_m    = Y_chan(:,1);
Full_Distance_Test_Dataset.Analytical_Azimuth_rad   = Y_chan(:,2);
Full_Distance_Test_Dataset.Analytical_Elevation_rad = Y_chan(:,3);

Full_Distance_Test_Dataset.APINN_Azimuth_deg   = rad2deg(Full_Distance_Test_Dataset.APINN_Azimuth_rad);
Full_Distance_Test_Dataset.APINN_Elevation_deg = rad2deg(Full_Distance_Test_Dataset.APINN_Elevation_rad);

Full_Distance_Test_Dataset.FixedPINN_Azimuth_deg   = rad2deg(Full_Distance_Test_Dataset.FixedPINN_Azimuth_rad);
Full_Distance_Test_Dataset.FixedPINN_Elevation_deg = rad2deg(Full_Distance_Test_Dataset.FixedPINN_Elevation_rad);

Full_Distance_Test_Dataset.DataNN_Azimuth_deg   = rad2deg(Full_Distance_Test_Dataset.DataNN_Azimuth_rad);
Full_Distance_Test_Dataset.DataNN_Elevation_deg = rad2deg(Full_Distance_Test_Dataset.DataNN_Elevation_rad);

Full_Distance_Test_Dataset.Analytical_Azimuth_deg   = rad2deg(Full_Distance_Test_Dataset.Analytical_Azimuth_rad);
Full_Distance_Test_Dataset.Analytical_Elevation_deg = rad2deg(Full_Distance_Test_Dataset.Analytical_Elevation_rad);

pos_true_test = [Y_true_test(:,1).*cos(Y_true_test(:,2)).*cos(Y_true_test(:,3)), ...
                 Y_true_test(:,1).*sin(Y_true_test(:,2)).*cos(Y_true_test(:,3)), ...
                 Y_true_test(:,1).*sin(Y_true_test(:,3))];

pos_apinn = [Y_test_pred_pinn(:,1).*cos(Y_test_pred_pinn(:,2)).*cos(Y_test_pred_pinn(:,3)), ...
             Y_test_pred_pinn(:,1).*sin(Y_test_pred_pinn(:,2)).*cos(Y_test_pred_pinn(:,3)), ...
             Y_test_pred_pinn(:,1).*sin(Y_test_pred_pinn(:,3))];

pos_fixed = [Y_test_pred_pinn_fixed(:,1).*cos(Y_test_pred_pinn_fixed(:,2)).*cos(Y_test_pred_pinn_fixed(:,3)), ...
             Y_test_pred_pinn_fixed(:,1).*sin(Y_test_pred_pinn_fixed(:,2)).*cos(Y_test_pred_pinn_fixed(:,3)), ...
             Y_test_pred_pinn_fixed(:,1).*sin(Y_test_pred_pinn_fixed(:,3))];

pos_data = [Y_test_pred_data(:,1).*cos(Y_test_pred_data(:,2)).*cos(Y_test_pred_data(:,3)), ...
            Y_test_pred_data(:,1).*sin(Y_test_pred_data(:,2)).*cos(Y_test_pred_data(:,3)), ...
            Y_test_pred_data(:,1).*sin(Y_test_pred_data(:,3))];

pos_chan = [Y_chan(:,1).*cos(Y_chan(:,2)).*cos(Y_chan(:,3)), ...
            Y_chan(:,1).*sin(Y_chan(:,2)).*cos(Y_chan(:,3)), ...
            Y_chan(:,1).*sin(Y_chan(:,3))];

Full_Distance_Test_Dataset.APINN_PosErr_m      = sqrt(sum((pos_apinn - pos_true_test).^2, 2));
Full_Distance_Test_Dataset.FixedPINN_PosErr_m  = sqrt(sum((pos_fixed - pos_true_test).^2, 2));
Full_Distance_Test_Dataset.DataNN_PosErr_m     = sqrt(sum((pos_data  - pos_true_test).^2, 2));
Full_Distance_Test_Dataset.Analytical_PosErr_m = sqrt(sum((pos_chan  - pos_true_test).^2, 2));

writetable(Full_Distance_Test_Dataset, 'Full_Distance_Test_Dataset_Reproduced.xlsx');
fprintf('Test dataset saved: Full_Distance_Test_Dataset_Reproduced.xlsx\n');

%% Fixed-PINN error metrics
fprintf('\nComputing Fixed-PINN error metrics...\n');

pos_pred_pinn_fixed = [Y_test_pred_pinn_fixed(:,1).*cos(Y_test_pred_pinn_fixed(:,2)).*cos(Y_test_pred_pinn_fixed(:,3)), ...
                       Y_test_pred_pinn_fixed(:,1).*sin(Y_test_pred_pinn_fixed(:,2)).*cos(Y_test_pred_pinn_fixed(:,3)), ...
                       Y_test_pred_pinn_fixed(:,1).*sin(Y_test_pred_pinn_fixed(:,3))];

error_pinn_fixed = sqrt(sum((pos_true_test - pos_pred_pinn_fixed).^2, 2));
fprintf('Fixed-PINN mean 3D position error: %.4f m\n', mean(error_pinn_fixed));

%% Visualization settings
set(0, 'DefaultAxesFontName', 'Times New Roman');
set(0, 'DefaultAxesFontSize', 16);
set(0, 'DefaultTextFontName', 'Times New Roman');
set(0, 'DefaultTextFontSize', 14);

color_blue   = [0, 0.447, 0.741];
color_red    = [0.85, 0.325, 0.098];
color_green  = [0.466, 0.674, 0.188];
color_purple = [0.494, 0.184, 0.556];
color_orange = [0.929, 0.694, 0.125];
color_cyan   = [0.301, 0.745, 0.933];
color_magenta = [0.8, 0.4, 0.8];
color_gray   = [0.5, 0.5, 0.5];

c_formula = color_red;
c_data    = color_green;
c_pinn    = color_blue;
c_gray    = color_gray;

%% Four fixed-turbidity error curves using saved data only
fprintf('\nPlotting smoothness comparison of three methods under four fixed turbidity levels (ISSD only)...\n');

fixed_turb_file = 'Fixed_Turbidity_Raw_Datasets.mat';

script_dir = fileparts(mfilename('fullpath'));
fixed_turb_candidates = {
    fullfile(script_dir, fixed_turb_file), ...
    fullfile(script_dir, 'data', fixed_turb_file), ...
    fixed_turb_file ...
};

fixed_turb_path = '';
for i = 1:numel(fixed_turb_candidates)
    if isfile(fixed_turb_candidates{i})
        fixed_turb_path = fixed_turb_candidates{i};
        break;
    end
end

if isempty(fixed_turb_path)
    fixed_turb_hits = dir(fullfile(pwd, '**', fixed_turb_file));
    if ~isempty(fixed_turb_hits)
        fixed_turb_path = fullfile(fixed_turb_hits(1).folder, fixed_turb_hits(1).name);
    end
end

if ~isempty(fixed_turb_path)
    load(fixed_turb_path, 'turb_levels', 'num_test_each', 'theta_eps_deg', 'theta_eps', 'fixed_turbidity_raw_sets');
    fprintf('Loaded fixed-turbidity raw data: %s\n', fixed_turb_path);

    dmin = 0.1;
    dmax = 50;
    dq = linspace(dmin, dmax, 400)';

    fig_smooth = figure('Position', [120, 80, 1400, 900], 'Color', 'w', 'Renderer', 'painters');
    tlo_smooth = tiledlayout(fig_smooth, 2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

    for k = 1:numel(turb_levels)
        turb_k = fixed_turbidity_raw_sets(k).turbidity;
        X_raw_k = fixed_turbidity_raw_sets(k).X_raw;
        Y_true_k = fixed_turbidity_raw_sets(k).Y_true;

        X_norm_k = normalize_inputs_tdoa(X_raw_k, params);

        Yp_apinn = forward_pass(net_pinn, X_norm_k);
        Yp_fixed = forward_pass(net_pinn_fixed, X_norm_k);
        Yp_data  = forward_pass(net_data, X_norm_k);

        pos_true_k = [Y_true_k(:,1).*cos(Y_true_k(:,2)).*cos(Y_true_k(:,3)), ...
                      Y_true_k(:,1).*sin(Y_true_k(:,2)).*cos(Y_true_k(:,3)), ...
                      Y_true_k(:,1).*sin(Y_true_k(:,3))];

        pos_apinn_k = [Yp_apinn(:,1).*cos(Yp_apinn(:,2)).*cos(Yp_apinn(:,3)), ...
                       Yp_apinn(:,1).*sin(Yp_apinn(:,2)).*cos(Yp_apinn(:,3)), ...
                       Yp_apinn(:,1).*sin(Yp_apinn(:,3))];

        pos_fixed_k = [Yp_fixed(:,1).*cos(Yp_fixed(:,2)).*cos(Yp_fixed(:,3)), ...
                       Yp_fixed(:,1).*sin(Yp_fixed(:,2)).*cos(Yp_fixed(:,3)), ...
                       Yp_fixed(:,1).*sin(Yp_fixed(:,3))];

        pos_data_k = [Yp_data(:,1).*cos(Yp_data(:,2)).*cos(Yp_data(:,3)), ...
                      Yp_data(:,1).*sin(Yp_data(:,2)).*cos(Yp_data(:,3)), ...
                      Yp_data(:,1).*sin(Yp_data(:,3))];

        e_apinn = sqrt(sum((pos_apinn_k - pos_true_k).^2, 2));
        e_fixed = sqrt(sum((pos_fixed_k - pos_true_k).^2, 2));
        e_data  = sqrt(sum((pos_data_k  - pos_true_k).^2, 2));
        d_true_k = Y_true_k(:,1);

        [~, yA] = smooth_curve_fallback(d_true_k, e_apinn, dq);
        [~, yF] = smooth_curve_fallback(d_true_k, e_fixed, dq);
        [~, yD] = smooth_curve_fallback(d_true_k, e_data,  dq);

        S_A = local_roughness_issd(yA, dq);
        S_F = local_roughness_issd(yF, dq);
        S_D = local_roughness_issd(yD, dq);

        ax = nexttile(tlo_smooth, k);
        hold(ax, 'on'); box(ax, 'on'); grid(ax, 'on');

        plot(ax, dq, yA, 'LineWidth', 3.0, 'Color', color_blue,   'DisplayName', 'APINN');
        plot(ax, dq, yF, 'LineWidth', 3.0, 'Color', color_purple, 'DisplayName', 'Fixed-PINN');
        plot(ax, dq, yD, 'LineWidth', 3.0, 'Color', color_green,  'DisplayName', 'Data-driven NN');

        title(ax, sprintf('Turbidity = %.1f NTU', turb_k), 'FontWeight', 'bold', 'FontSize', 18);
        xlabel(ax, 'True Distance (m)', 'FontWeight', 'bold', 'FontSize', 16);
        ylabel(ax, '3D Position Error (m)', 'FontWeight', 'bold', 'FontSize', 16);

        xlim(ax, [dmin dmax]);
        ymax_k = max([yA; yF; yD], [], 'omitnan');
        ylim(ax, [0, 1.12*ymax_k]);

        ax.FontName = 'Times New Roman';
        ax.FontSize = 14;
        ax.LineWidth = 1.2;
        ax.TickDir = 'out';
        ax.Layer = 'top';

        txt = sprintf(['Smoothness (ISSD)\n', ...
                       'APINN:      ISSD = %.2f\n', ...
                       'Fixed-PINN: ISSD = %.2f\n', ...
                       'Data-NN:    ISSD = %.2f'], ...
                       S_A, S_F, S_D);

        text(ax, dmin + 0.02*(dmax-dmin), 0.95*ymax_k, txt, ...
            'FontSize', 12, 'FontWeight', 'bold', ...
            'BackgroundColor', 'white', 'EdgeColor', [0.2 0.2 0.2], ...
            'LineWidth', 1.0, 'Margin', 6, 'VerticalAlignment', 'top');

        if k == 1
            lgd = legend(ax, 'Location', 'northoutside', 'Orientation', 'horizontal');
            lgd.Box = 'off';
            lgd.FontSize = 13;
        end

        hold(ax, 'off');
    end

    print(fig_smooth, 'Fig_Smoothness_4Turbidity_3Methods_ISSD_Reproduced.png', '-dpng', '-r300');
    fprintf('Saved: Fig_Smoothness_4Turbidity_3Methods_ISSD_Reproduced.png\n');
else
    checked = strjoin(fixed_turb_candidates, ', ');
    warning(['Fixed-turbidity raw data file Fixed_Turbidity_Raw_Datasets.mat was not found. Checked: %s. ', ...
             'Data generation is disabled in this script, so the four-fixed-turbidity reproduction section is skipped.'], checked);
end

