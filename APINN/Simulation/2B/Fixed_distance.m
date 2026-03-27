clear; clc; close all;
setup_paths();

% %% Parameter settings
% layers = [11, 32, 64, 64, 32, 3];
% 
% params.c_water = 1500;
% params.f_U = 800;
% params.f_V = 800;
% params.f_detector = 50;
% params.n_w = 1.33;
% params.n_g = 1.5;
% params.n_a = 1.0;
% params.l1 = 100;
% params.l2 = 20;
% 
% % TDOA hydrophone array configuration
% params.hydrophone_positions = [
%     0, 0, 0.004;
%     0.005, 0, 0;
%     -0.005, 0.004, 0;
%     0, -0.004, -0.004
% ];
% 
% % Normalization scales
% params.u_scale = params.f_U * tan(pi/3);
% params.v_scale = params.f_V * tan(pi/4);
% 
% % Noise parameters
% params.SNR=55;
% params.SNR_tdoa = params.SNR;
% params.SNR_camera = params.SNR;
% params.SNR_detector = params.SNR;
% params.SNR_depth = params.SNR;
% 
% params.temperature = 15;
% params.salinity = 35;
% params.depth = 10;
% params.pressure_factor = 1.02;
% 
% train_params.num_samples =2000;
% train_params.num_epochs = 2000;
% train_params.batch_size = 64;
% train_params.learning_rate = 0.001;
% train_params.beta1 = 0.9;
% train_params.beta2 = 0.999;
% train_params.epsilon = 1e-8;
% 
% train_params.output_weights = [7.0, 18.0, 1.0];
% 
% % Fixed weights
% train_params.lambda_tdoa = 3.0;
% train_params.lambda_depth = 5.0;
% train_params.lambda_consistency = 3.0;
% train_params.lambda_camera = 4;
% train_params.lambda_detector = 2;
% train_params.lambda_smooth = 0.1;
% 
% % Turbidity parameters
% params.turbidity = 10;
% params.turbidity_coefficient = 0.2;

%% Load reproducible experiment state
repro_mat_file = 'Fixed_Distance_Repro_State.mat';
fprintf('Loading reproducible experiment state file: %s\n', repro_mat_file);

S = load(repro_mat_file, ...
    'rng_state_before_init', ...
    'rng_state_at_training_start', ...
    'X_raw_train', 'Y_true_train', ...
    'X_raw_test', 'Y_true_test', ...
    'params', 'train_params', 'layers', ...
    'fixed_distances', ...
    'theta_eps_deg', 'theta_eps', ...
    'num_test', ...
    'net_pinn', 'adam_state_pinn', ...
    'net_pinn_fixed', 'adam_state_pinn_fixed', ...
    'net_data', 'adam_state_data');

rng_state_before_init   = S.rng_state_before_init;
rng_state_at_training_start = S.rng_state_at_training_start;

X_raw_train = S.X_raw_train;
Y_true_train = S.Y_true_train;
X_raw_test  = S.X_raw_test;
Y_true_test = S.Y_true_test;

params = S.params;
train_params = S.train_params;
layers = S.layers;

fixed_distances = S.fixed_distances;
theta_eps_deg = S.theta_eps_deg;
theta_eps = S.theta_eps;
num_test = S.num_test;

net_pinn = S.net_pinn;
adam_state_pinn = S.adam_state_pinn;

net_pinn_fixed = S.net_pinn_fixed;
adam_state_pinn_fixed = S.adam_state_pinn_fixed;

net_data = S.net_data;
adam_state_data = S.adam_state_data;

fprintf('MAT file loaded successfully.\n');
fprintf('Number of training samples: %d\n', size(X_raw_train,1));
fprintf('Number of test samples: %d\n', size(X_raw_test,1));
fprintf('Training distance points: ');
fprintf('%.1f ', fixed_distances);
fprintf('m\n');

%% Validate refraction model
fprintf('=== Validating refraction model ===\n');
test_angles = [-20, -10, -5, 0, 5, 10, 20] * pi/180;

max_error = 0;
for i = 1:length(test_angles)
    psi_orig = test_angles(i);
    delta = psi_to_delta_paper_eq1(psi_orig, params);
    psi_recovered = delta_to_psi_paper_eq1(delta, params);
    error = abs(psi_recovered - psi_orig);
    max_error = max(max_error, error);
    
    fprintf('psi: %+6.2f deg -> Delta_m: %+8.4f mm -> psi: %+6.2f deg (error: %.2e rad)\n', ...
            rad2deg(psi_orig), delta, rad2deg(psi_recovered), error);
end

fprintf('Maximum round-trip error: %.2e rad (%.4f deg)\n\n', max_error, rad2deg(max_error));

if max_error > 1e-6
    warning('Refraction model validation failed!');
    return;
end
fprintf('Refraction model validation passed.\n\n');

%% Re-normalize original data from MAT file
fprintf('Reading original data from MAT file and re-normalizing...\n');

X_norm_train = normalize_inputs_tdoa(X_raw_train, params);
X_norm_test  = normalize_inputs_tdoa(X_raw_test, params);

%% Save training dataset
fprintf('Saving training dataset...\n');

TrainingDataTable = array2table([X_raw_train, Y_true_train], ...
    'VariableNames', {'TDOA1_s', 'TDOA2_s', 'TDOA3_s', ...
                      'CameraU_pixel', 'CameraV_pixel', ...
                      'QuadrantUI_V', 'QuadrantUII_V', 'QuadrantUIII_V', 'QuadrantUIV_V', ...
                      'Depth_m', 'Turbidity_NTU', ...
                      'TrueDistance_m', 'TrueAzimuth_rad', 'TrueElevation_rad'});

TrainingDataTable.TrueAzimuth_deg = rad2deg(TrainingDataTable.TrueAzimuth_rad);
TrainingDataTable.TrueElevation_deg = rad2deg(TrainingDataTable.TrueElevation_rad);

writetable(TrainingDataTable, 'Fixed_Distance_Training_Dataset.xlsx');
fprintf('Training dataset saved to: Fixed_Distance_Training_Dataset.xlsx\n');

fprintf('Training data preview:\n');
disp(head(TrainingDataTable, 5));

rng(rng_state_at_training_start);
fprintf('Random state before training has been restored to ensure reproducibility.\n');

%% Train APINN
fprintf('\nStart APINN Training\n');
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
    X_raw_shuffled = X_raw_train(idx, :);
    
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

        [loss, gradients] = compute_loss_and_gradients_tdoa(...
            net_pinn, X_batch, X_raw_batch, Y_pred, Y_batch, activations, ...
            params, train_params);
        
        [net_pinn, adam_state_pinn] = adam_update(net_pinn, gradients, adam_state_pinn, current_lr_pinn, ...
                                       train_params.beta1, train_params.beta2, ...
                                       train_params.epsilon, epoch);
        
        epoch_loss.total = epoch_loss.total + loss.total;
        epoch_loss.data = epoch_loss.data + loss.data;
        epoch_loss.tdoa = epoch_loss.tdoa + loss.tdoa;
        epoch_loss.camera = epoch_loss.camera + loss.camera;
        epoch_loss.detector = epoch_loss.detector + loss.detector;
        epoch_loss.depth = epoch_loss.depth + loss.depth;
    end
    
    epoch_loss.total = epoch_loss.total / num_batches;
    epoch_loss.data = epoch_loss.data / num_batches;
    epoch_loss.tdoa = epoch_loss.tdoa / num_batches;
    epoch_loss.camera = epoch_loss.camera / num_batches;
    epoch_loss.detector = epoch_loss.detector / num_batches;
    epoch_loss.depth = epoch_loss.depth / num_batches;
    
    epoch_avg_weight_camera = epoch_camera_weights_sum / epoch_samples_count; %#ok<NASGU>
    epoch_avg_weight_detector = epoch_detector_weights_sum / epoch_samples_count; %#ok<NASGU>
    
    loss_history_pinn.total(epoch) = epoch_loss.total;
    loss_history_pinn.data(epoch) = epoch_loss.data;
    loss_history_pinn.tdoa(epoch) = epoch_loss.tdoa;
    loss_history_pinn.camera(epoch) = epoch_loss.camera;
    loss_history_pinn.detector(epoch) = epoch_loss.detector;
    loss_history_pinn.depth(epoch) = epoch_loss.depth;
    
    if mod(epoch, 10) == 0 || epoch == 1
        Y_test_pred_epoch = forward_pass(net_pinn, X_norm_test);
        errors_epoch = Y_test_pred_epoch - Y_true_test;
        
        mae_d = sum(abs(errors_epoch(:,1))) / size(errors_epoch, 1);
        mae_psi = sum(abs(errors_epoch(:,2))) / size(errors_epoch, 1);
        mae_theta = sum(abs(errors_epoch(:,3))) / size(errors_epoch, 1);
        
        test_mae_history_pinn.distance(end+1) = mae_d;
        test_mae_history_pinn.azimuth(end+1) = mae_psi;
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

fprintf('APINN training completed.\n\n');

%% Train fixed-weight PINN
fprintf('\nStart Fixed-Weight PINN Training\n');
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
    X_raw_shuffled = X_raw_train(idx, :);
    
    epoch_loss = struct('total', 0, 'data', 0, 'tdoa', 0, 'camera', 0, 'detector', 0, 'depth', 0);
    num_batches = ceil(train_params.num_samples / train_params.batch_size);
    
    for batch = 1:num_batches
        batch_start = (batch-1) * train_params.batch_size + 1;
        batch_end = min(batch * train_params.batch_size, train_params.num_samples);
        
        X_batch = X_norm_shuffled(batch_start:batch_end, :);
        Y_batch = Y_true_shuffled(batch_start:batch_end, :);
        X_raw_batch = X_raw_shuffled(batch_start:batch_end, :);
        
        [Y_pred, activations] = forward_pass(net_pinn_fixed, X_batch);
        
        [loss, gradients] = compute_loss_and_gradients_tdoa_fixed(...
            net_pinn_fixed, X_batch, X_raw_batch, Y_pred, Y_batch, activations, ...
            params, train_params);
        
        [net_pinn_fixed, adam_state_pinn_fixed] = adam_update(net_pinn_fixed, gradients, adam_state_pinn_fixed, current_lr_pinn_fixed, ...
                                       train_params.beta1, train_params.beta2, ...
                                       train_params.epsilon, epoch);
        
        epoch_loss.total = epoch_loss.total + loss.total;
        epoch_loss.data = epoch_loss.data + loss.data;
        epoch_loss.tdoa = epoch_loss.tdoa + loss.tdoa;
        epoch_loss.camera = epoch_loss.camera + loss.camera;
        epoch_loss.detector = epoch_loss.detector + loss.detector;
        epoch_loss.depth = epoch_loss.depth + loss.depth;
    end
    
    epoch_loss.total = epoch_loss.total / num_batches;
    epoch_loss.data = epoch_loss.data / num_batches;
    epoch_loss.tdoa = epoch_loss.tdoa / num_batches;
    epoch_loss.camera = epoch_loss.camera / num_batches;
    epoch_loss.detector = epoch_loss.detector / num_batches;
    epoch_loss.depth = epoch_loss.depth / num_batches;
    
    loss_history_pinn_fixed.total(epoch) = epoch_loss.total;
    loss_history_pinn_fixed.data(epoch) = epoch_loss.data;
    loss_history_pinn_fixed.tdoa(epoch) = epoch_loss.tdoa;
    loss_history_pinn_fixed.camera(epoch) = epoch_loss.camera;
    loss_history_pinn_fixed.detector(epoch) = epoch_loss.detector;
    loss_history_pinn_fixed.depth(epoch) = epoch_loss.depth;
    
    if mod(epoch, 10) == 0 || epoch == 1
        Y_test_pred_epoch = forward_pass(net_pinn_fixed, X_norm_test);
        errors_epoch = Y_test_pred_epoch - Y_true_test;
        
        mae_d = sum(abs(errors_epoch(:,1))) / size(errors_epoch, 1);
        mae_psi = sum(abs(errors_epoch(:,2))) / size(errors_epoch, 1);
        mae_theta = sum(abs(errors_epoch(:,3))) / size(errors_epoch, 1);
        
        test_mae_history_pinn_fixed.distance(end+1) = mae_d;
        test_mae_history_pinn_fixed.azimuth(end+1) = mae_psi;
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

fprintf('Fixed-weight PINN training completed.\n\n');

%% Train purely data-driven neural network
fprintf('Start Purely Data-Driven Neural Network Training\n');
loss_history_data = struct('total', [], 'data', []);
test_mae_history_data = struct('distance', [], 'azimuth', [], 'elevation', []);
position_error_history_data = [];
current_lr_data = train_params.learning_rate;
weight_decay = 1;

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
                                       train_params.beta1, train_params.beta2, ...
                                       train_params.epsilon, epoch);
        
        epoch_loss.total = epoch_loss.total + loss.total;
        epoch_loss.data = epoch_loss.data + loss.data;
    end
    
    epoch_loss.total = epoch_loss.total / num_batches;
    epoch_loss.data = epoch_loss.data / num_batches;
    
    loss_history_data.total(epoch) = epoch_loss.total;
    loss_history_data.data(epoch) = epoch_loss.data;
    
    if mod(epoch, 10) == 0 || epoch == 1
        Y_test_pred_epoch = forward_pass(net_data, X_norm_test);
        errors_epoch = Y_test_pred_epoch - Y_true_test;
        
        mae_d = sum(abs(errors_epoch(:,1))) / size(errors_epoch, 1);
        mae_psi = sum(abs(errors_epoch(:,2))) / size(errors_epoch, 1);
        mae_theta = sum(abs(errors_epoch(:,3))) / size(errors_epoch, 1);
        
        test_mae_history_data.distance(end+1) = mae_d;
        test_mae_history_data.azimuth(end+1) = mae_psi;
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
        fprintf('[Data-driven] Epoch %d/%d - Loss: %.6f \n', ...
                epoch, train_params.num_epochs, epoch_loss.total);
    end
end

%% Final evaluation
fprintf('\nEvaluating all models...\n');
Y_test_pred_pinn = forward_pass(net_pinn, X_norm_test);
Y_test_pred_pinn_fixed = forward_pass(net_pinn_fixed, X_norm_test);
Y_test_pred_data = forward_pass(net_data, X_norm_test);
Y_chan = compute_by_chan_algorithm(X_raw_test, params, train_params);

errors_pinn = Y_test_pred_pinn - Y_true_test;
errors_pinn_fixed = Y_test_pred_pinn_fixed - Y_true_test;
errors_data = Y_test_pred_data - Y_true_test;
errors_chan = Y_chan - Y_true_test;

mae_pinn_d = sum(abs(errors_pinn(:,1))) / size(errors_pinn, 1);
mae_pinn_psi = sum(abs(errors_pinn(:,2))) / size(errors_pinn, 1);
mae_pinn_theta = sum(abs(errors_pinn(:,3))) / size(errors_pinn, 1);
mae_pinn = [mae_pinn_d, mae_pinn_psi, mae_pinn_theta];

mae_pinn_fixed_d = sum(abs(errors_pinn_fixed(:,1))) / size(errors_pinn_fixed, 1);
mae_pinn_fixed_psi = sum(abs(errors_pinn_fixed(:,2))) / size(errors_pinn_fixed, 1);
mae_pinn_fixed_theta = sum(abs(errors_pinn_fixed(:,3))) / size(errors_pinn_fixed, 1);
mae_pinn_fixed = [mae_pinn_fixed_d, mae_pinn_fixed_psi, mae_pinn_fixed_theta];

mae_data_d = sum(abs(errors_data(:,1))) / size(errors_data, 1);
mae_data_psi = sum(abs(errors_data(:,2))) / size(errors_data, 1);
mae_data_theta = sum(abs(errors_data(:,3))) / size(errors_data, 1);
mae_data = [mae_data_d, mae_data_psi, mae_data_theta];

mae_chan_d = sum(abs(errors_chan(:,1))) / size(errors_chan, 1);
mae_chan_psi = sum(abs(errors_chan(:,2))) / size(errors_chan, 1);
mae_chan_theta = sum(abs(errors_chan(:,3))) / size(errors_chan, 1);
mae_chan = [mae_chan_d, mae_chan_psi, mae_chan_theta];

fprintf('\nTest Set Performance Comparison\n');
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

%% Save test dataset (including outputs of four methods)
fprintf('Saving test dataset (including outputs from four methods)...\n');

TestDataTable = array2table([X_raw_test, Y_true_test], ...
    'VariableNames', {'TDOA1_s', 'TDOA2_s', 'TDOA3_s', ...
                      'CameraU_pixel', 'CameraV_pixel', ...
                      'QuadrantUI_V', 'QuadrantUII_V', 'QuadrantUIII_V', 'QuadrantUIV_V', ...
                      'Depth_m','Turbidity_NTU', ...
                      'TrueDistance_m', 'TrueAzimuth_rad', 'TrueElevation_rad'});

TestDataTable.TrueAzimuth_deg   = rad2deg(TestDataTable.TrueAzimuth_rad);
TestDataTable.TrueElevation_deg = rad2deg(TestDataTable.TrueElevation_rad);

TestDataTable.APINN_Distance_m      = Y_test_pred_pinn(:,1);
TestDataTable.APINN_Azimuth_rad     = Y_test_pred_pinn(:,2);
TestDataTable.APINN_Elevation_rad   = Y_test_pred_pinn(:,3);
TestDataTable.APINN_Azimuth_deg     = rad2deg(Y_test_pred_pinn(:,2));
TestDataTable.APINN_Elevation_deg   = rad2deg(Y_test_pred_pinn(:,3));

TestDataTable.FixedPINN_Distance_m    = Y_test_pred_pinn_fixed(:,1);
TestDataTable.FixedPINN_Azimuth_rad   = Y_test_pred_pinn_fixed(:,2);
TestDataTable.FixedPINN_Elevation_rad = Y_test_pred_pinn_fixed(:,3);
TestDataTable.FixedPINN_Azimuth_deg   = rad2deg(Y_test_pred_pinn_fixed(:,2));
TestDataTable.FixedPINN_Elevation_deg = rad2deg(Y_test_pred_pinn_fixed(:,3));

TestDataTable.DataNN_Distance_m     = Y_test_pred_data(:,1);
TestDataTable.DataNN_Azimuth_rad    = Y_test_pred_data(:,2);
TestDataTable.DataNN_Elevation_rad  = Y_test_pred_data(:,3);
TestDataTable.DataNN_Azimuth_deg    = rad2deg(Y_test_pred_data(:,2));
TestDataTable.DataNN_Elevation_deg  = rad2deg(Y_test_pred_data(:,3));

TestDataTable.Analytical_Distance_m     = Y_chan(:,1);
TestDataTable.Analytical_Azimuth_rad    = Y_chan(:,2);
TestDataTable.Analytical_Elevation_rad  = Y_chan(:,3);
TestDataTable.Analytical_Azimuth_deg    = rad2deg(Y_chan(:,2));
TestDataTable.Analytical_Elevation_deg  = rad2deg(Y_chan(:,3));

pos_true = [Y_true_test(:,1).*cos(Y_true_test(:,2)).*cos(Y_true_test(:,3)), ...
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

pos_analytical = [Y_chan(:,1).*cos(Y_chan(:,2)).*cos(Y_chan(:,3)), ...
                  Y_chan(:,1).*sin(Y_chan(:,2)).*cos(Y_chan(:,3)), ...
                  Y_chan(:,1).*sin(Y_chan(:,3))];

TestDataTable.APINN_PosErr_m       = sqrt(sum((pos_true - pos_apinn).^2, 2));
TestDataTable.FixedPINN_PosErr_m   = sqrt(sum((pos_true - pos_fixed).^2, 2));
TestDataTable.DataNN_PosErr_m      = sqrt(sum((pos_true - pos_data).^2, 2));
TestDataTable.Analytical_PosErr_m  = sqrt(sum((pos_true - pos_analytical).^2, 2));

writetable(TestDataTable, 'Fixed_Distance_Test_Dataset.xlsx', 'Sheet', 'Test');
fprintf('Test dataset saved to: Fixed_Distance_Test_Dataset.xlsx\n');

%% Additional evaluation before visualization
fprintf('\nComputing error metrics for Fixed-PINN...\n');

pos_pred_pinn_fixed = [Y_test_pred_pinn_fixed(:,1).*cos(Y_test_pred_pinn_fixed(:,2)).*cos(Y_test_pred_pinn_fixed(:,3)), ...
                       Y_test_pred_pinn_fixed(:,1).*sin(Y_test_pred_pinn_fixed(:,2)).*cos(Y_test_pred_pinn_fixed(:,3)), ...
                       Y_test_pred_pinn_fixed(:,1).*sin(Y_test_pred_pinn_fixed(:,3))];

error_pinn_fixed = sqrt(sum((pos_true - pos_pred_pinn_fixed).^2, 2));

fprintf('Fixed-PINN mean 3D position error: %.4f m\n', mean(error_pinn_fixed));

%% Visualization section
set(0, 'DefaultAxesFontName', 'Times New Roman');
set(0, 'DefaultAxesFontSize', 16);
set(0, 'DefaultTextFontName', 'Times New Roman');
set(0, 'DefaultTextFontSize', 14);

color_blue = [0, 0.447, 0.741];
color_red = [0.85, 0.325, 0.098];
color_green = [0.466, 0.674, 0.188];
color_purple = [0.494, 0.184, 0.556];
color_orange = [0.929, 0.694, 0.125];
color_cyan = [0.301, 0.745, 0.933];
color_magenta = [0.8, 0.4, 0.8]; %#ok<NASGU>
color_gray = [0.5, 0.5, 0.5];

c_formula = color_red;
c_data = color_green;
c_pinn = color_blue;
c_gray = color_gray;

%% Figure 1: PINN loss curves and 3D position error evolution
fig_combined = figure('Position', [100, 100, 1400, 600], 'Color', 'w');

subplot(1,2,1);
hold on; box on; grid on;

physics_loss_total_apinn = loss_history_pinn.tdoa + loss_history_pinn.camera + ...
                           loss_history_pinn.detector + loss_history_pinn.depth;

plot(1:train_params.num_epochs, loss_history_pinn.total, 'LineWidth', 3, ...
     'Color', color_blue, 'DisplayName', 'APINN Total');
plot(1:train_params.num_epochs, physics_loss_total_apinn, 'LineWidth', 2.5, ...
     'Color', color_green, 'DisplayName', 'APINN Physics', 'LineStyle', '--');

physics_loss_total_fixed = loss_history_pinn_fixed.tdoa + loss_history_pinn_fixed.camera + ...
                           loss_history_pinn_fixed.detector + loss_history_pinn_fixed.depth;

plot(1:train_params.num_epochs, loss_history_pinn_fixed.total, 'LineWidth', 2.8, ...
     'Color', color_purple, 'DisplayName', 'Fixed-PINN Total');
plot(1:train_params.num_epochs, physics_loss_total_fixed, 'LineWidth', 2.3, ...
     'Color', color_orange, 'DisplayName', 'Fixed-PINN Physics', 'LineStyle', '--');

plot(1:train_params.num_epochs, loss_history_data.total, 'LineWidth', 2.6, ...
     'Color', color_cyan, 'DisplayName', 'Data-driven NN');

set(gca, 'YScale', 'log');
xlabel('Training Epoch', 'FontWeight', 'bold', 'FontSize', 18);
ylabel('Loss Value', 'FontWeight', 'bold', 'FontSize', 18);
legend('Location', 'northeast', 'FontSize', 15);
xlim([0, train_params.num_epochs]);
set(gca, 'LineWidth', 1.8, 'GridAlpha', 0.3, 'FontSize', 16);
hold off;

subplot(1,2,2);
hold on; box on; grid on;

n_pos_points = length(position_error_history_pinn);
pos_epoch_points = (0:(n_pos_points-1)) * 10 + 1;

plot(pos_epoch_points, position_error_history_pinn, 'LineWidth', 3, 'Color', color_blue, ...
    'Marker', 'o', 'MarkerSize', 6, 'DisplayName', 'APINN');
plot(pos_epoch_points, position_error_history_pinn_fixed, 'LineWidth', 3, 'Color', color_purple, ...
    'Marker', '^', 'MarkerSize', 6, 'DisplayName', 'Fixed-PINN');
plot(pos_epoch_points, position_error_history_data, 'LineWidth', 3, 'Color', color_green, ...
    'Marker', 's', 'MarkerSize', 6, 'DisplayName', 'Data-driven NN');

xlabel('Training Epoch', 'FontWeight', 'bold', 'FontSize', 18);
ylabel('Mean 3D Position Error (m)', 'FontWeight', 'bold', 'FontSize', 18);
legend('Location', 'best', 'FontSize', 15);
xlim([0, train_params.num_epochs]);
set(gca, 'LineWidth', 1.8, 'GridAlpha', 0.3, 'FontSize', 16);

text(train_params.num_epochs*0.55, ...
    max([position_error_history_pinn, position_error_history_pinn_fixed, position_error_history_data])*0.75, ...
    sprintf('Final Error:\nAPINN: %.3f m\nFixed-PINN: %.3f m\nData-NN: %.3f m', ...
    position_error_history_pinn(end), position_error_history_pinn_fixed(end), position_error_history_data(end)), ...
    'FontSize', 15, 'FontWeight', 'bold', ...
    'BackgroundColor', 'white', 'EdgeColor', 'k', 'LineWidth', 1.5);

hold off;

%% Figure 6: MAE comparison and improvement percentages
categories = {'Distance (m)', 'Azimuth (掳)', 'Elevation (掳)'};

mae_pinn_plot = [mae_pinn(1), rad2deg(mae_pinn(2)), rad2deg(mae_pinn(3))];
mae_pinn_fixed_plot = [mae_pinn_fixed(1), rad2deg(mae_pinn_fixed(2)), rad2deg(mae_pinn_fixed(3))];
mae_data_plot = [mae_data(1), rad2deg(mae_data(2)), rad2deg(mae_data(3))];
mae_chan_plot = [mae_chan(1), rad2deg(mae_chan(2)), rad2deg(mae_chan(3))];

improvement_apinn_vs_fixed = (mae_pinn_fixed_plot - mae_pinn_plot) ./ mae_pinn_fixed_plot * 100;
improvement_apinn_vs_data  = (mae_data_plot - mae_pinn_plot) ./ mae_data_plot * 100;
improvement_apinn_vs_chan  = (mae_chan_plot - mae_pinn_plot) ./ mae_chan_plot * 100;

x = 1:3;

fig7 = figure('Position', [120, 120, 1280, 460], 'Color', 'w', 'Renderer', 'painters');
t = tiledlayout(fig7, 1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

ax1 = nexttile(t, 1); hold(ax1,'on'); box(ax1,'on');

data_mae = [mae_chan_plot(:), mae_data_plot(:), mae_pinn_fixed_plot(:), mae_pinn_plot(:)];
b = bar(ax1, x, data_mae, 'grouped', 'BarWidth', 0.85);
b(1).FaceColor = c_formula;     b(1).EdgeColor = 'none';
b(2).FaceColor = c_data;        b(2).EdgeColor = 'none';
b(3).FaceColor = color_purple;  b(3).EdgeColor = 'none';
b(4).FaceColor = c_pinn;        b(4).EdgeColor = 'none';

set(ax1, 'XTick', x, 'XTickLabel', categories);
ylabel(ax1, 'Mean Absolute Error', 'FontWeight', 'bold', 'FontSize', 18);

grid(ax1,'on');
ax1.YGrid = 'on'; ax1.XGrid = 'off';
ax1.GridAlpha = 0.15; ax1.MinorGridAlpha = 0.08;
ax1.LineWidth = 1.2;
ax1.TickDir = 'out';
ax1.Layer = 'top';
ax1.FontName = 'Times New Roman';
ax1.FontSize = 16;

lgd1 = legend(ax1, {'Analytical','Data-driven NN','Fixed-PINN','APINN'}, ...
    'Location','northoutside','Orientation','horizontal');
lgd1.Box = 'off';
lgd1.NumColumns = 4;
lgd1.FontSize = 14;

ymax1 = max(data_mae(:));
dy1 = 0.02 * ymax1;
for i = 1:numel(x)
    for j = 1:4
        text(ax1, b(j).XEndPoints(i), b(j).YEndPoints(i) + dy1, ...
            sprintf('%.2f', data_mae(i,j)), ...
            'HorizontalAlignment','center', 'VerticalAlignment','bottom', ...
            'FontSize', 11, 'FontWeight','bold', 'Color', c_gray);
    end
end
ylim(ax1, [0, ymax1*1.25]);

ax2 = nexttile(t, 2); hold(ax2,'on'); box(ax2,'on');

data_imp = [improvement_apinn_vs_chan(:), improvement_apinn_vs_data(:), improvement_apinn_vs_fixed(:)];
b2 = bar(ax2, x, data_imp, 'grouped', 'BarWidth', 0.85);
b2(1).FaceColor = color_orange;
b2(2).FaceColor = color_green;
b2(3).FaceColor = color_purple;

set(ax2, 'XTick', x, 'XTickLabel', categories);
ylabel(ax2, 'Improvement (%)', 'FontWeight', 'bold', 'FontSize', 18);

yline(ax2, 0, '-', 'LineWidth', 1.0, 'Color', [0.2 0.2 0.2], 'HandleVisibility','off');

grid(ax2,'on');
ax2.YGrid = 'on'; ax2.XGrid = 'off';
ax2.GridAlpha = 0.15;
ax2.LineWidth = 1.2;
ax2.TickDir = 'out';
ax2.Layer = 'top';
ax2.FontName = 'Times New Roman';
ax2.FontSize = 16;

lgd2 = legend(ax2, {'APINN vs Analytical','APINN vs Data-NN','APINN vs Fixed-PINN'}, ...
    'Location','northoutside','Orientation','horizontal');
lgd2.Box = 'off';
lgd2.NumColumns = 3;
lgd2.FontSize = 14;

ymax2 = max(data_imp(:));
ymin2 = min(data_imp(:));
pad2 = 0.06 * max(1, max(abs([ymax2,ymin2])));
for i = 1:numel(x)
    for j = 1:3
        val = data_imp(i,j);
        ytxt = val + sign(val + 1e-9) * pad2;
        text(ax2, b2(j).XEndPoints(i), ytxt, ...
            sprintf('%.1f%%', val), ...
            'HorizontalAlignment','center', 'VerticalAlignment','middle', ...
            'FontSize', 11, 'FontWeight','bold', 'Color', c_gray);
    end
end
ylim(ax2, [min(0, ymin2 - 2*pad2), ymax2 + 2.5*pad2]);

print(fig7, 'Fig7_8_Combined_Four_Methods_Comparison.png', '-dpng', '-r300');

%% Figure 7: Positioning error versus distance (four methods)
fig9 = figure('Position', [120, 120, 980, 520], 'Color', 'w', 'Renderer', 'painters');
ax = axes(fig9); hold(ax,'on'); box(ax,'on');

pos_true = [Y_true_test(:,1).*cos(Y_true_test(:,2)).*cos(Y_true_test(:,3)), ...
            Y_true_test(:,1).*sin(Y_true_test(:,2)).*cos(Y_true_test(:,3)), ...
            Y_true_test(:,1).*sin(Y_true_test(:,3))];

pos_pred_pinn = [Y_test_pred_pinn(:,1).*cos(Y_test_pred_pinn(:,2)).*cos(Y_test_pred_pinn(:,3)), ...
                 Y_test_pred_pinn(:,1).*sin(Y_test_pred_pinn(:,2)).*cos(Y_test_pred_pinn(:,3)), ...
                 Y_test_pred_pinn(:,1).*sin(Y_test_pred_pinn(:,3))];

pos_pred_pinn_fixed = [Y_test_pred_pinn_fixed(:,1).*cos(Y_test_pred_pinn_fixed(:,2)).*cos(Y_test_pred_pinn_fixed(:,3)), ...
                       Y_test_pred_pinn_fixed(:,1).*sin(Y_test_pred_pinn_fixed(:,2)).*cos(Y_test_pred_pinn_fixed(:,3)), ...
                       Y_test_pred_pinn_fixed(:,1).*sin(Y_test_pred_pinn_fixed(:,3))];

pos_pred_data = [Y_test_pred_data(:,1).*cos(Y_test_pred_data(:,2)).*cos(Y_test_pred_data(:,3)), ...
                 Y_test_pred_data(:,1).*sin(Y_test_pred_data(:,2)).*cos(Y_test_pred_data(:,3)), ...
                 Y_test_pred_data(:,1).*sin(Y_test_pred_data(:,3))];

pos_pred_chan = [Y_chan(:,1).*cos(Y_chan(:,2)).*cos(Y_chan(:,3)), ...
                 Y_chan(:,1).*sin(Y_chan(:,2)).*cos(Y_chan(:,3)), ...
                 Y_chan(:,1).*sin(Y_chan(:,3))];

error_pinn = sqrt(sum((pos_true - pos_pred_pinn).^2, 2));
error_pinn_fixed = sqrt(sum((pos_true - pos_pred_pinn_fixed).^2, 2));
error_data = sqrt(sum((pos_true - pos_pred_data).^2, 2));
error_chan = sqrt(sum((pos_true - pos_pred_chan).^2, 2));

d_true = Y_true_test(:,1);

dmin = 0.1; dmax = 50;
dq = linspace(dmin, dmax, 400)';

useSpline = (exist('fit','file')==2);

if useSpline
    okC = isfinite(d_true) & isfinite(error_chan) & d_true>=dmin & d_true<=dmax & error_chan>=0;
    okD = isfinite(d_true) & isfinite(error_data) & d_true>=dmin & d_true<=dmax & error_data>=0;
    okP = isfinite(d_true) & isfinite(error_pinn) & d_true>=dmin & d_true<=dmax & error_pinn>=0;
    okF = isfinite(d_true) & isfinite(error_pinn_fixed) & d_true>=dmin & d_true<=dmax & error_pinn_fixed>=0;

    spC = fit(d_true(okC), error_chan(okC), 'smoothingspline', 'SmoothingParam', 0.85);
    spD = fit(d_true(okD), error_data(okD), 'smoothingspline', 'SmoothingParam', 0.85);
    spP = fit(d_true(okP), error_pinn(okP), 'smoothingspline', 'SmoothingParam', 0.85);
    spF = fit(d_true(okF), error_pinn_fixed(okF), 'smoothingspline', 'SmoothingParam', 0.85);

    yC = max(spC(dq), 0);
    yD = max(spD(dq), 0);
    yP = max(spP(dq), 0);
    yF = max(spF(dq), 0);
else
    [~, yC] = smooth_curve_fallback(d_true, error_chan, dq);
    [~, yD] = smooth_curve_fallback(d_true, error_data, dq);
    [~, yP] = smooth_curve_fallback(d_true, error_pinn, dq);
    [~, yF] = smooth_curve_fallback(d_true, error_pinn_fixed, dq);
end

hC = plot(ax, dq, yC, 'LineWidth', 3.0, 'Color', color_red, 'DisplayName', 'Analytical');
hD = plot(ax, dq, yD, 'LineWidth', 3.0, 'Color', color_green, 'DisplayName', 'Data-driven NN');
hF = plot(ax, dq, yF, 'LineWidth', 3.0, 'Color', color_purple, 'DisplayName', 'Fixed-PINN');
hP = plot(ax, dq, yP, 'LineWidth', 3.0, 'Color', color_blue, 'DisplayName', 'APINN');

xlabel(ax, 'True Distance (m)', 'FontWeight', 'bold', 'FontSize', 18);
ylabel(ax, '3D Position Error (m)', 'FontWeight', 'bold', 'FontSize', 18);

xlim(ax, [dmin, dmax]);
ymax = max([yC; yD; yP; yF], [], 'omitnan');
ylim(ax, [0, 1.12*ymax]);

grid(ax,'on');
ax.XMinorTick = 'on'; ax.YMinorTick = 'on';
ax.MinorGridLineStyle = ':';
ax.GridAlpha = 0.15; ax.MinorGridAlpha = 0.08;
ax.LineWidth = 1.2;
ax.TickDir = 'out';
ax.Layer = 'top';
ax.FontName = 'Times New Roman';
ax.FontSize = 16;

lgd = legend(ax, [hC,hD,hF,hP], 'Location','northoutside', 'Orientation','horizontal');
lgd.NumColumns = 4;
lgd.Box = 'off';
lgd.FontSize = 15;

print(fig9, 'Fig9_Error_vs_Distance_FourMethods.png', '-dpng', '-r300');

fprintf('\nVisualization complete\n');
fprintf('All visualizations are complete!\n');

