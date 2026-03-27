clear; clc; close all;
setup_paths();
%% Section
% layers = [11, 32, 64, 64, 32, 3];  % 11 (3DOA + u, v, U_I~U_IV, depth

% params.c_water = 1500;
% params.f_U = 800;
% params.f_V = 800;
% params.f_detector = 50;
% params.n_w = 1.33;
% params.n_g = 1.5;
% params.n_a = 1.0;
% params.l1 = 100;
% params.l2 = 20;

% % TDOA,3
% params.hydrophone_positions = [
%     0, 0, 0.004;
%     0.005, 0, 0;
%     -0.005, 0.004, 0;
%     0, -0.004, -0.004
% ];


% params.u_scale = params.f_U * tan(pi/3);
% params.v_scale = params.f_V * tan(pi/4);

% % :SNR,dB)
% params.SNR=50;
% params.SNR_tdoa = params.SNR;        % TDOA(dB)
% params.SNR_camera = params.SNR;      % (dB)
% params.SNR_detector = params.SNR;    % (dB)
% params.SNR_depth = params.SNR;       %  (dB)

% params.temperature = 15;
% params.salinity = 35;
% params.depth = 10;
% params.pressure_factor = 1.02;

% train_params.num_samples =2000;
% train_params.num_epochs = 3000;
% train_params.batch_size = 64;
% train_params.learning_rate = 0.001;
% train_params.beta1 = 0.9;
% train_params.beta2 = 0.999;
% train_params.epsilon = 1e-8;

% train_params.output_weights = [8.0, 19.0, 1.0];


% train_params.lambda_tdoa = 3.0;      % TDOA
% train_params.lambda_depth = 5.0;     % 
% train_params.lambda_consistency = 3.0;  % 
% train_params.lambda_camera = 4;     % 
% train_params.lambda_detector = 2;   % 
% train_params.lambda_smooth = 0.1;

% params.turbidity = 10;  % NTU (Nephelometric Turbidity Units)
% % : 0-5 NTU(), 5-15 NTU(, 15+ NTU()

% params.turbidity_coefficient = 0.2;  % 
% % c = c_base + k_turb * turbidity

%% Section
repro_file = 'Smooth_test2_Repro_State.mat';
if ~exist(repro_file, 'file')
    error('Reproducible state file not found: %s', repro_file);
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
    'net_data', 'adam_state_data', ...
    'step_test_data', ...
    'X_raw_step', 'Y_true_step', 'turbidity_step', 'd_step');

fprintf(' %s\n', repro_file);

%% Section
rng(rng_state_at_training_start);

%% Section
X_norm_train = normalize_inputs_tdoa(X_raw_train, params);
X_norm_test = normalize_inputs_tdoa(X_raw_test, params);

%% APINN training
fprintf('\nAPINN\n');

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
        epoch_avg_weight_camera = epoch_camera_weights_sum / epoch_samples_count; %#ok<NASGU>
        epoch_avg_weight_detector = epoch_detector_weights_sum / epoch_samples_count; %#ok<NASGU>
    else
        epoch_avg_weight_camera = NaN; %#ok<NASGU>
        epoch_avg_weight_detector = NaN; %#ok<NASGU>
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

fprintf('APINN \n\n');

%% Fixed-PINN training
fprintf('\nPINN\n');

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

fprintf(' PINN \n\n');

%% Data-driven training
fprintf('\n');

loss_history_data = struct('total', [], 'data', []);
test_mae_history_data = struct('distance', [], 'azimuth', [], 'elevation', []);
position_error_history_data = [];
current_lr_data = train_params.learning_rate;
weight_decay = 0.001;

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

fprintf('\n\n');

%% Section
fprintf('\n..\n');
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

fprintf('\nn');
fprintf('APINN ():\n');
fprintf('  MAE:  %.4f m\n', mae_pinn(1));
fprintf('  MAE: %.2f\n', rad2deg(mae_pinn(2)));
fprintf('  MAE: %.2f\n', rad2deg(mae_pinn(3)));

fprintf('\nPINN:\n');
fprintf('  MAE:  %.4f m\n', mae_pinn_fixed(1));
fprintf('  MAE: %.2f\n', rad2deg(mae_pinn_fixed(2)));
fprintf('  MAE: %.2f\n', rad2deg(mae_pinn_fixed(3)));

fprintf('\n\n');
fprintf('  MAE:  %.4f m\n', mae_data(1));
fprintf('  MAE: %.2f\n', rad2deg(mae_data(2)));
fprintf('  MAE: %.2f\n', rad2deg(mae_data(3)));

fprintf('\nChan + \n');
fprintf('  MAE:  %.4f m\n', mae_chan(1));
fprintf('  MAE: %.2f\n', rad2deg(mae_chan(2)));
fprintf('  MAE: %.2f\n', rad2deg(mae_chan(3)));
fprintf('\n');

%% Section
fprintf('\n..\n');

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
fprintf(': Full_Distance_Test_Dataset_Reproduced.xlsx\n');

%% Fixed-PINN training
fprintf('\nixed-PINN..\n');

pos_pred_pinn_fixed = [Y_test_pred_pinn_fixed(:,1).*cos(Y_test_pred_pinn_fixed(:,2)).*cos(Y_test_pred_pinn_fixed(:,3)), ...
                       Y_test_pred_pinn_fixed(:,1).*sin(Y_test_pred_pinn_fixed(:,2)).*cos(Y_test_pred_pinn_fixed(:,3)), ...
                       Y_test_pred_pinn_fixed(:,1).*sin(Y_test_pred_pinn_fixed(:,3))];

error_pinn_fixed = sqrt(sum((pos_true_test - pos_pred_pinn_fixed).^2, 2));
fprintf('Fixed-PINN3D %.4f m\n', mean(error_pinn_fixed));

%% Section
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

c_formula = color_red; %#ok<NASGU>
c_data    = color_green; %#ok<NASGU>
c_pinn    = color_blue; %#ok<NASGU>
c_gray    = color_gray; %#ok<NASGU>

%% Section
%% Section
%% Section
fprintf('\n\n');

has_step_data = false;

if exist('step_test_data', 'var') && ~isempty(step_test_data)
    if isfield(step_test_data, 'step_cfg') && ...
       isfield(step_test_data, 'X_raw_step') && ...
       isfield(step_test_data, 'Y_true_step') && ...
       isfield(step_test_data, 'turbidity_step') && ...
       isfield(step_test_data, 'd_step')

        step_cfg = step_test_data.step_cfg;
        X_raw_step = step_test_data.X_raw_step;
        Y_true_step = step_test_data.Y_true_step;
        turbidity_step = step_test_data.turbidity_step;
        d_step = step_test_data.d_step;
        has_step_data = true;
    end
end

if ~has_step_data
    if exist('X_raw_step', 'var') && exist('Y_true_step', 'var') && ...
       exist('turbidity_step', 'var') && exist('d_step', 'var')

        has_step_data = true;

        if ~exist('step_cfg', 'var')
            step_cfg = struct();
            step_cfg.turbidity_levels = [3, 12, 7];
            step_cfg.boundaries = [17, 33];
            step_cfg.num_points = length(d_step);
            step_cfg.d_min = min(d_step);
            step_cfg.d_max = max(d_step);
            step_cfg.psi0 = Y_true_step(1,2);
            step_cfg.theta0 = Y_true_step(1,3);
        end
    end
end

if has_step_data
    fprintf(' %s ', repro_file);

    X_norm_step = normalize_inputs_tdoa(X_raw_step, params);

    Y_pred_step_pinn  = forward_pass(net_pinn, X_norm_step);
    Y_pred_step_fixed = forward_pass(net_pinn_fixed, X_norm_step);
    Y_pred_step_data  = forward_pass(net_data, X_norm_step);

    P_true_step = [Y_true_step(:,1).*cos(Y_true_step(:,2)).*cos(Y_true_step(:,3)), ...
                   Y_true_step(:,1).*sin(Y_true_step(:,2)).*cos(Y_true_step(:,3)), ...
                   Y_true_step(:,1).*sin(Y_true_step(:,3))];

    P_pinn_step = [Y_pred_step_pinn(:,1).*cos(Y_pred_step_pinn(:,2)).*cos(Y_pred_step_pinn(:,3)), ...
                   Y_pred_step_pinn(:,1).*sin(Y_pred_step_pinn(:,2)).*cos(Y_pred_step_pinn(:,3)), ...
                   Y_pred_step_pinn(:,1).*sin(Y_pred_step_pinn(:,3))];

    P_fixed_step = [Y_pred_step_fixed(:,1).*cos(Y_pred_step_fixed(:,2)).*cos(Y_pred_step_fixed(:,3)), ...
                    Y_pred_step_fixed(:,1).*sin(Y_pred_step_fixed(:,2)).*cos(Y_pred_step_fixed(:,3)), ...
                    Y_pred_step_fixed(:,1).*sin(Y_pred_step_fixed(:,3))];

    P_data_step = [Y_pred_step_data(:,1).*cos(Y_pred_step_data(:,2)).*cos(Y_pred_step_data(:,3)), ...
                   Y_pred_step_data(:,1).*sin(Y_pred_step_data(:,2)).*cos(Y_pred_step_data(:,3)), ...
                   Y_pred_step_data(:,1).*sin(Y_pred_step_data(:,3))];

    err3d_pinn_step  = sqrt(sum((P_pinn_step  - P_true_step).^2, 2));
    err3d_fixed_step = sqrt(sum((P_fixed_step - P_true_step).^2, 2));
    err3d_data_step  = sqrt(sum((P_data_step  - P_true_step).^2, 2));

    metrics_pinn  = compute_step_smoothness_metrics(d_step, P_pinn_step,  step_cfg.boundaries);
    metrics_fixed = compute_step_smoothness_metrics(d_step, P_fixed_step, step_cfg.boundaries);
    metrics_data  = compute_step_smoothness_metrics(d_step, P_data_step,  step_cfg.boundaries);

    fprintf('\n--- ---\n');
    fprintf('APINN:\n');
    fprintf('  J0@17m = %.4f, J1@17m = %.4f\n', metrics_pinn.J0(1), metrics_pinn.J1(1));
    fprintf('  J0@33m = %.4f, J1@33m = %.4f\n', metrics_pinn.J0(2), metrics_pinn.J1(2));

    fprintf('Fixed-PINN:\n');
    fprintf('  J0@17m = %.4f, J1@17m = %.4f\n', metrics_fixed.J0(1), metrics_fixed.J1(1));
    fprintf('  J0@33m = %.4f, J1@33m = %.4f\n', metrics_fixed.J0(2), metrics_fixed.J1(2));

    fprintf('Data-NN:\n');
    fprintf('  J0@17m = %.4f, J1@17m = %.4f\n', metrics_data.J0(1), metrics_data.J1(1));
    fprintf('  J0@33m = %.4f, J1@33m = %.4f\n', metrics_data.J0(2), metrics_data.J1(2));


    % 0(a)

    fig10a = figure('Position', [100, 100, 820, 620], 'Color', 'w', 'Renderer', 'painters');
    hold on; box on; grid on;

    h_fixed = plot(d_step, Y_pred_step_fixed(:,1), 'LineWidth', 1.8, ...
        'Color', color_purple, 'DisplayName', 'Fixed-PINN');
    h_data  = plot(d_step, Y_pred_step_data(:,1),  'LineWidth', 1.8, ...
        'Color', color_green,  'DisplayName', 'Data-NN');
    h_ideal = plot(d_step, d_step, '--', 'LineWidth', 1.8, ...
        'Color', color_gray, 'DisplayName', 'Ideal');
    h_pinn  = plot(d_step, Y_pred_step_pinn(:,1),  'LineWidth', 2.4, ...
        'Color', color_blue, 'DisplayName', 'APINN');

    xline(step_cfg.boundaries(1), '--', 'LineWidth', 1.0, ...
        'Color', [0.6 0.6 0.6], 'HandleVisibility','off');
    xline(step_cfg.boundaries(2), '--', 'LineWidth', 1.0, ...
        'Color', [0.6 0.6 0.6], 'HandleVisibility','off');

    text(step_cfg.boundaries(1)-1.2, 48, 'T=3 \rightarrow 12', ...
        'FontSize', 11, 'Color', [0.25 0.25 0.25]);
    text(step_cfg.boundaries(2)-2.6, 48, 'T=12 \rightarrow 7', ...
        'FontSize', 11, 'Color', [0.25 0.25 0.25]);

    xlabel('True Distance (m)', 'FontWeight', 'bold', 'FontSize', 16);
    ylabel('Predicted Distance (m)', 'FontWeight', 'bold', 'FontSize', 16);

    legend([h_pinn, h_fixed, h_data, h_ideal], ...
        {'APINN', 'Fixed-PINN', 'Data-NN', 'Ideal'}, ...
        'Location', 'northwest', 'FontSize', 11);

    xlim([step_cfg.d_min, step_cfg.d_max]);
    ylim([step_cfg.d_min, step_cfg.d_max]);
    set(gca, 'LineWidth', 1.2, 'FontSize', 12);

    txt_17m = sprintf(['APINN:   J0=%.4f, J1=%.4f\n' ...
                       'Fixed:   J0=%.4f, J1=%.4f\n' ...
                       'Data-NN: J0=%.4f, J1=%.4f'], ...
                       metrics_pinn.J0(1),  metrics_pinn.J1(1), ...
                       metrics_fixed.J0(1), metrics_fixed.J1(1), ...
                       metrics_data.J0(1),  metrics_data.J1(1));

    txt_33m = sprintf(['APINN:   J0=%.4f, J1=%.4f\n' ...
                       'Fixed:   J0=%.4f, J1=%.4f\n' ...
                       'Data-NN: J0=%.4f, J1=%.4f'], ...
                       metrics_pinn.J0(2),  metrics_pinn.J1(2), ...
                       metrics_fixed.J0(2), metrics_fixed.J1(2), ...
                       metrics_data.J0(2),  metrics_data.J1(2));

    text(18.0, 9.2, txt_17m, ...
        'FontSize', 10, 'FontWeight', 'bold', ...
        'BackgroundColor', 'white', ...
        'EdgeColor', [0.7 0.7 0.7], ...
        'Margin', 6);

    text(31.2, 18.5, txt_33m, ...
        'FontSize', 10, 'FontWeight', 'bold', ...
        'BackgroundColor', 'white', ...
        'EdgeColor', [0.7 0.7 0.7], ...
        'Margin', 6);

    hold off;
    print(fig10a, 'Fig10a_Turbidity_Step_Distance_Reproduced.png', '-dpng', '-r300');
    fprintf('0(a) Fig10a_Turbidity_Step_Distance_Reproduced.png\n');


    % 0(b)

    fig10b = figure('Position', [150, 120, 820, 620], 'Color', 'w', 'Renderer', 'painters');
    hold on; box on; grid on;

    h_fixed_b = plot(d_step, err3d_fixed_step, '-', 'LineWidth', 1.2, ...
        'Color', color_purple, 'DisplayName', 'Fixed-PINN');
    h_data_b  = plot(d_step, err3d_data_step,  '-', 'LineWidth', 1.2, ...
        'Color', color_green,  'DisplayName', 'Data-NN');
    h_pinn_b  = plot(d_step, err3d_pinn_step,  '-', 'LineWidth', 1.6, ...
        'Color', color_blue,   'DisplayName', 'APINN');

    xline(step_cfg.boundaries(1), '--', 'LineWidth', 1.0, ...
          'Color', [0.6 0.6 0.6], 'HandleVisibility','off');
    xline(step_cfg.boundaries(2), '--', 'LineWidth', 1.0, ...
          'Color', [0.6 0.6 0.6], 'HandleVisibility','off');

    xlabel('True Distance (m)', 'FontWeight', 'bold', 'FontSize', 16);
    ylabel('3D Position Error (m)', 'FontWeight', 'bold', 'FontSize', 16);

    legend([h_pinn_b, h_fixed_b, h_data_b], ...
        {'APINN', 'Fixed-PINN', 'Data-NN'}, ...
        'Location', 'northwest', 'FontSize', 11);

    xlim([step_cfg.d_min, step_cfg.d_max]);
    ymax = max([err3d_pinn_step; err3d_fixed_step; err3d_data_step]);
    ylim([0, 1.05 * ymax]);
    set(gca, 'LineWidth', 1.2, 'FontSize', 12);

    hold off;
    print(fig10b, 'Fig10b_Turbidity_Step_3DError_Reproduced.png', '-dpng', '-r300');
    fprintf('0(b) Fig10b_Turbidity_Step_3DError_Reproduced.png\n');
end

fprintf('\n\n');
fprintf('\n');

