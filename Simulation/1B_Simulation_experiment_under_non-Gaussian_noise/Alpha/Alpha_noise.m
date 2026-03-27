clear; clc; close all;
setup_paths();
project_root = fileparts(mfilename('fullpath'));

%% Parameter setup
% layers = [11, 32, 64, 64, 32, 3];  % Input dimension = 11 (3 TDOA + u, v, U_I~U_IV, depth, turbidity)
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
% % Noise parameters: unified SNR in dB
% params.SNR = 55;
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
% train_params.num_samples = 2000;
% train_params.num_epochs = 3000;
% train_params.batch_size = 64;
% train_params.learning_rate = 0.001;
% train_params.beta1 = 0.9;
% train_params.beta2 = 0.999;
% train_params.epsilon = 1e-8;
%
% train_params.output_weights = [9.0, 19.0, 3.0];
%
% % Fixed weights
% train_params.lambda_tdoa = 3.0;
% train_params.lambda_depth = 5.0;
% train_params.lambda_consistency = 3.0;
% train_params.lambda_camera = 4;
% train_params.lambda_detector = 2;
% train_params.tdoa_loss_scale = 1e6;
%
% % Noise type switch: use GMM
% params.noise.gmm.eps   = 0.1;
% params.noise.gmm.kappa = 10;
%
% % Alpha-stable parameters
% params.noise.alpha.alpha = 1.6;
% params.noise.alpha.beta  = 0;
%
% % Middleton Class A parameters
% params.noise.midA.A     = 0.2;
% params.noise.midA.Gamma = 0.01;
% params.noise.midA.Mmax  = 30;
%
% params.noise.tdoa.type     = 'alpha';
% params.noise.camera.type   = 'alpha';
% params.noise.detector.type = 'alpha';
% params.noise.depth.type    = 'alpha';
%
% % Turbidity parameters
% params.turbidity = 10;  % NTU
% params.turbidity_coefficient = 0.2;

%% 0. Read reproducible state
repro_file = fullfile(project_root, 'data', 'Alpha_Distance_Repro_State.mat');
traj_file  = fullfile(project_root, 'data', 'Alpha_Trajectory_Test_Data.mat');

if ~isfile(repro_file)
    error('File not found: %s', repro_file);
end
if ~isfile(traj_file)
    error('File not found: %s', traj_file);
end

fprintf('Loading reproducible training state: %s\n', repro_file);
S = load(repro_file);

fprintf('Loading trajectory test data: %s\n', traj_file);
T = load(traj_file);

% Restore variables from MAT files
params = S.params;
train_params = S.train_params;
layers = S.layers;

X_raw_train = S.X_raw_train;
Y_true_train = S.Y_true_train;
X_raw_test = S.X_raw_test;
Y_true_test = S.Y_true_test;

net_pinn = S.net_pinn;
adam_state_pinn = S.adam_state_pinn;

net_pinn_fixed = S.net_pinn_fixed;
adam_state_pinn_fixed = S.adam_state_pinn_fixed;

net_data = S.net_data;
adam_state_data = S.adam_state_data;

rng_state_before_init = S.rng_state_before_init;
rng_state_at_training_start = S.rng_state_at_training_start;

traj_params = T.traj_params;
traj = T.traj;
X_raw_traj = T.X_raw_traj;
Y_true_traj = T.Y_true_traj;
seg_id = T.seg_id;

fprintf('Training samples: %d\n', size(X_raw_train,1));
fprintf('Test samples: %d\n', size(X_raw_test,1));
fprintf('Trajectory samples: %d\n\n', size(X_raw_traj,1));

%% 1. Data normalization
fprintf('Normalizing training, test, and trajectory data...\n');
X_norm_train = normalize_inputs_tdoa(X_raw_train, params);
X_norm_test  = normalize_inputs_tdoa(X_raw_test,  params);
X_norm_traj  = normalize_inputs_tdoa(X_raw_traj,  params);

fprintf('\nNormalization range check\n');
fprintf('Train TDOA1: [%.3f, %.3f]\n', min(X_norm_train(:,1)), max(X_norm_train(:,1)));
fprintf('Train TDOA2: [%.3f, %.3f]\n', min(X_norm_train(:,2)), max(X_norm_train(:,2)));
fprintf('Train TDOA3: [%.3f, %.3f]\n', min(X_norm_train(:,3)), max(X_norm_train(:,3)));
fprintf('Train U:     [%.3f, %.3f]\n', min(X_norm_train(:,4)), max(X_norm_train(:,4)));
fprintf('Train V:     [%.3f, %.3f]\n', min(X_norm_train(:,5)), max(X_norm_train(:,5)));
fprintf('Train Depth: [%.3f, %.3f]\n\n', min(X_norm_train(:,10)), max(X_norm_train(:,10)));

%% 2. Restore RNG state at training start
rng(rng_state_at_training_start);
fprintf('Random state at training start has been restored.\n\n');

%% 3. Retrain APINN
fprintf('\nStart reproducing APINN training\n');

loss_history_pinn = struct('total', [], 'data', [], 'tdoa', [], 'camera', [], 'detector', [], 'depth', [], 'consistency', [], 'physics', []);
test_mae_history_pinn = struct('distance', [], 'azimuth', [], 'elevation', []);
position_error_history_pinn = [];
current_lr_pinn = train_params.learning_rate;

num_train = size(X_norm_train, 1);

for epoch = 1:train_params.num_epochs
    if mod(epoch, 200) == 0 && epoch > 0
        current_lr_pinn = current_lr_pinn * 0.85;
    end

    idx = randperm(num_train);
    X_norm_shuffled = X_norm_train(idx, :);
    Y_true_shuffled = Y_true_train(idx, :);
    X_raw_shuffled  = X_raw_train(idx, :);

    epoch_loss = struct('total', 0, 'data', 0, 'tdoa', 0, 'camera', 0, 'detector', 0, 'depth', 0, 'consistency', 0);
    num_batches = ceil(num_train / train_params.batch_size);

    for batch = 1:num_batches
        batch_start = (batch-1) * train_params.batch_size + 1;
        batch_end   = min(batch * train_params.batch_size, num_train);

        X_batch     = X_norm_shuffled(batch_start:batch_end, :);
        Y_batch     = Y_true_shuffled(batch_start:batch_end, :);
        X_raw_batch = X_raw_shuffled(batch_start:batch_end, :);

        [Y_pred, activations] = forward_pass(net_pinn, X_batch);

        [loss, gradients] = compute_loss_and_gradients_tdoa( ...
            net_pinn, X_batch, X_raw_batch, Y_pred, Y_batch, activations, ...
            params, train_params);

        [net_pinn, adam_state_pinn] = adam_update(net_pinn, gradients, adam_state_pinn, current_lr_pinn, ...
            train_params.beta1, train_params.beta2, train_params.epsilon, epoch);

        epoch_loss.total       = epoch_loss.total       + loss.total;
        epoch_loss.data        = epoch_loss.data        + loss.data;
        epoch_loss.tdoa        = epoch_loss.tdoa        + loss.tdoa;
        epoch_loss.camera      = epoch_loss.camera      + loss.camera;
        epoch_loss.detector    = epoch_loss.detector    + loss.detector;
        epoch_loss.depth       = epoch_loss.depth       + loss.depth;
        epoch_loss.consistency = epoch_loss.consistency + loss.consistency;
    end

    epoch_loss.total       = epoch_loss.total       / num_batches;
    epoch_loss.data        = epoch_loss.data        / num_batches;
    epoch_loss.tdoa        = epoch_loss.tdoa        / num_batches;
    epoch_loss.camera      = epoch_loss.camera      / num_batches;
    epoch_loss.detector    = epoch_loss.detector    / num_batches;
    epoch_loss.depth       = epoch_loss.depth       / num_batches;
    epoch_loss.consistency = epoch_loss.consistency / num_batches;
    epoch_physics = epoch_loss.total - epoch_loss.data;

    loss_history_pinn.total(epoch)       = epoch_loss.total;
    loss_history_pinn.data(epoch)        = epoch_loss.data;
    loss_history_pinn.tdoa(epoch)        = epoch_loss.tdoa;
    loss_history_pinn.camera(epoch)      = epoch_loss.camera;
    loss_history_pinn.detector(epoch)    = epoch_loss.detector;
    loss_history_pinn.depth(epoch)       = epoch_loss.depth;
    loss_history_pinn.consistency(epoch) = epoch_loss.consistency;
    loss_history_pinn.physics(epoch)     = epoch_physics;

    if mod(epoch, 10) == 0 || epoch == 1
        Y_test_pred_epoch = forward_pass(net_pinn, X_norm_test);
        errors_epoch = Y_test_pred_epoch - Y_true_test;

        mae_d     = mean(abs(errors_epoch(:,1)));
        mae_psi   = mean(abs(errors_epoch(:,2)));
        mae_theta = mean(abs(errors_epoch(:,3)));

        test_mae_history_pinn.distance(end+1)  = mae_d;
        test_mae_history_pinn.azimuth(end+1)   = mae_psi;
        test_mae_history_pinn.elevation(end+1) = mae_theta;

        pos_true_epoch = [Y_true_test(:,1).*cos(Y_true_test(:,2)).*cos(Y_true_test(:,3)), ...
                          Y_true_test(:,1).*sin(Y_true_test(:,2)).*cos(Y_true_test(:,3)), ...
                          Y_true_test(:,1).*sin(Y_true_test(:,3))];

        pos_pred_epoch = [Y_test_pred_epoch(:,1).*cos(Y_test_pred_epoch(:,2)).*cos(Y_test_pred_epoch(:,3)), ...
                          Y_test_pred_epoch(:,1).*sin(Y_test_pred_epoch(:,2)).*cos(Y_test_pred_epoch(:,3)), ...
                          Y_test_pred_epoch(:,1).*sin(Y_test_pred_epoch(:,3))];

        pos_errors = sqrt(sum((pos_true_epoch - pos_pred_epoch).^2, 2));
        position_error_history_pinn(end+1) = mean(pos_errors);
    end

    if mod(epoch, 500) == 0 || epoch == 1
        fprintf('[APINN] Epoch %d/%d\n', epoch, train_params.num_epochs);
        fprintf('        Total: %.6f | Data: %.6f | Physics: %.6f\n', ...
            epoch_loss.total, epoch_loss.data, epoch_physics);
    end
end

fprintf('APINN reproduction training completed.\n\n');

%% 4. Retrain Fixed-PINN
fprintf('\nStart reproducing Fixed-PINN training\n');

loss_history_pinn_fixed = struct('total', [], 'data', [], 'tdoa', [], 'camera', [], 'detector', [], 'depth', [], 'consistency', [], 'physics', []);
test_mae_history_pinn_fixed = struct('distance', [], 'azimuth', [], 'elevation', []);
position_error_history_pinn_fixed = [];
current_lr_pinn_fixed = train_params.learning_rate;

for epoch = 1:train_params.num_epochs
    if mod(epoch, 200) == 0 && epoch > 0
        current_lr_pinn_fixed = current_lr_pinn_fixed * 0.85;
    end

    idx = randperm(num_train);
    X_norm_shuffled = X_norm_train(idx, :);
    Y_true_shuffled = Y_true_train(idx, :);
    X_raw_shuffled  = X_raw_train(idx, :);

    epoch_loss = struct('total', 0, 'data', 0, 'tdoa', 0, 'camera', 0, 'detector', 0, 'depth', 0, 'consistency', 0);
    num_batches = ceil(num_train / train_params.batch_size);

    for batch = 1:num_batches
        batch_start = (batch-1) * train_params.batch_size + 1;
        batch_end   = min(batch * train_params.batch_size, num_train);

        X_batch     = X_norm_shuffled(batch_start:batch_end, :);
        Y_batch     = Y_true_shuffled(batch_start:batch_end, :);
        X_raw_batch = X_raw_shuffled(batch_start:batch_end, :);

        [Y_pred, activations] = forward_pass(net_pinn_fixed, X_batch);

        [loss, gradients] = compute_loss_and_gradients_tdoa_fixed( ...
            net_pinn_fixed, X_batch, X_raw_batch, Y_pred, Y_batch, activations, ...
            params, train_params);

        [net_pinn_fixed, adam_state_pinn_fixed] = adam_update(net_pinn_fixed, gradients, adam_state_pinn_fixed, current_lr_pinn_fixed, ...
            train_params.beta1, train_params.beta2, train_params.epsilon, epoch);

        epoch_loss.total       = epoch_loss.total       + loss.total;
        epoch_loss.data        = epoch_loss.data        + loss.data;
        epoch_loss.tdoa        = epoch_loss.tdoa        + loss.tdoa;
        epoch_loss.camera      = epoch_loss.camera      + loss.camera;
        epoch_loss.detector    = epoch_loss.detector    + loss.detector;
        epoch_loss.depth       = epoch_loss.depth       + loss.depth;
        epoch_loss.consistency = epoch_loss.consistency + loss.consistency;
    end

    epoch_loss.total       = epoch_loss.total       / num_batches;
    epoch_loss.data        = epoch_loss.data        / num_batches;
    epoch_loss.tdoa        = epoch_loss.tdoa        / num_batches;
    epoch_loss.camera      = epoch_loss.camera      / num_batches;
    epoch_loss.detector    = epoch_loss.detector    / num_batches;
    epoch_loss.depth       = epoch_loss.depth       / num_batches;
    epoch_loss.consistency = epoch_loss.consistency / num_batches;
    epoch_physics = epoch_loss.total - epoch_loss.data;

    loss_history_pinn_fixed.total(epoch)       = epoch_loss.total;
    loss_history_pinn_fixed.data(epoch)        = epoch_loss.data;
    loss_history_pinn_fixed.tdoa(epoch)        = epoch_loss.tdoa;
    loss_history_pinn_fixed.camera(epoch)      = epoch_loss.camera;
    loss_history_pinn_fixed.detector(epoch)    = epoch_loss.detector;
    loss_history_pinn_fixed.depth(epoch)       = epoch_loss.depth;
    loss_history_pinn_fixed.consistency(epoch) = epoch_loss.consistency;
    loss_history_pinn_fixed.physics(epoch)     = epoch_physics;

    if mod(epoch, 10) == 0 || epoch == 1
        Y_test_pred_epoch = forward_pass(net_pinn_fixed, X_norm_test);
        errors_epoch = Y_test_pred_epoch - Y_true_test;

        mae_d     = mean(abs(errors_epoch(:,1)));
        mae_psi   = mean(abs(errors_epoch(:,2)));
        mae_theta = mean(abs(errors_epoch(:,3)));

        test_mae_history_pinn_fixed.distance(end+1)  = mae_d;
        test_mae_history_pinn_fixed.azimuth(end+1)   = mae_psi;
        test_mae_history_pinn_fixed.elevation(end+1) = mae_theta;

        pos_true_epoch = [Y_true_test(:,1).*cos(Y_true_test(:,2)).*cos(Y_true_test(:,3)), ...
                          Y_true_test(:,1).*sin(Y_true_test(:,2)).*cos(Y_true_test(:,3)), ...
                          Y_true_test(:,1).*sin(Y_true_test(:,3))];

        pos_pred_epoch = [Y_test_pred_epoch(:,1).*cos(Y_test_pred_epoch(:,2)).*cos(Y_test_pred_epoch(:,3)), ...
                          Y_test_pred_epoch(:,1).*sin(Y_test_pred_epoch(:,2)).*cos(Y_test_pred_epoch(:,3)), ...
                          Y_test_pred_epoch(:,1).*sin(Y_test_pred_epoch(:,3))];

        pos_errors = sqrt(sum((pos_true_epoch - pos_pred_epoch).^2, 2));
        position_error_history_pinn_fixed(end+1) = mean(pos_errors);
    end

    if mod(epoch, 500) == 0 || epoch == 1
        fprintf('[Fixed-PINN] Epoch %d/%d\n', epoch, train_params.num_epochs);
        fprintf('             Total: %.6f | Data: %.6f | Physics: %.6f\n', ...
            epoch_loss.total, epoch_loss.data, epoch_physics);
    end
end

fprintf('Fixed-PINN reproduction training completed.\n\n');

%% 5. Retrain Data-driven NN
fprintf('\nStart reproducing Data-driven NN training\n');

loss_history_data = struct('total', [], 'data', []);
test_mae_history_data = struct('distance', [], 'azimuth', [], 'elevation', []);
position_error_history_data = [];
current_lr_data = train_params.learning_rate;
weight_decay = 0.05;

for epoch = 1:train_params.num_epochs
    if mod(epoch, 200) == 0 && epoch > 0
        current_lr_data = current_lr_data * 0.85;
    end

    idx = randperm(num_train);
    X_norm_shuffled = X_norm_train(idx, :);
    Y_true_shuffled = Y_true_train(idx, :);

    epoch_loss = struct('total', 0, 'data', 0);
    num_batches = ceil(num_train / train_params.batch_size);

    for batch = 1:num_batches
        batch_start = (batch-1) * train_params.batch_size + 1;
        batch_end   = min(batch * train_params.batch_size, num_train);

        X_batch = X_norm_shuffled(batch_start:batch_end, :);
        Y_batch = Y_true_shuffled(batch_start:batch_end, :);

        [Y_pred, activations] = forward_pass(net_data, X_batch);

        [loss, gradients] = compute_loss_data_driven(net_data, X_batch, Y_pred, Y_batch, ...
            activations, train_params, weight_decay);

        [net_data, adam_state_data] = adam_update(net_data, gradients, adam_state_data, current_lr_data, ...
            train_params.beta1, train_params.beta2, train_params.epsilon, epoch);

        epoch_loss.total = epoch_loss.total + loss.total;
        epoch_loss.data  = epoch_loss.data  + loss.data;
    end

    epoch_loss.total = epoch_loss.total / num_batches;
    epoch_loss.data  = epoch_loss.data  / num_batches;

    loss_history_data.total(epoch) = epoch_loss.total;
    loss_history_data.data(epoch)  = epoch_loss.data;

    if mod(epoch, 10) == 0 || epoch == 1
        Y_test_pred_epoch = forward_pass(net_data, X_norm_test);
        errors_epoch = Y_test_pred_epoch - Y_true_test;

        mae_d     = mean(abs(errors_epoch(:,1)));
        mae_psi   = mean(abs(errors_epoch(:,2)));
        mae_theta = mean(abs(errors_epoch(:,3)));

        test_mae_history_data.distance(end+1)  = mae_d;
        test_mae_history_data.azimuth(end+1)   = mae_psi;
        test_mae_history_data.elevation(end+1) = mae_theta;

        pos_true_epoch = [Y_true_test(:,1).*cos(Y_true_test(:,2)).*cos(Y_true_test(:,3)), ...
                          Y_true_test(:,1).*sin(Y_true_test(:,2)).*cos(Y_true_test(:,3)), ...
                          Y_true_test(:,1).*sin(Y_true_test(:,3))];

        pos_pred_epoch = [Y_test_pred_epoch(:,1).*cos(Y_test_pred_epoch(:,2)).*cos(Y_test_pred_epoch(:,3)), ...
                          Y_test_pred_epoch(:,1).*sin(Y_test_pred_epoch(:,2)).*cos(Y_test_pred_epoch(:,3)), ...
                          Y_test_pred_epoch(:,1).*sin(Y_test_pred_epoch(:,3))];

        pos_errors = sqrt(sum((pos_true_epoch - pos_pred_epoch).^2, 2));
        position_error_history_data(end+1) = mean(pos_errors);
    end

    if mod(epoch, 500) == 0 || epoch == 1
        fprintf('[Data-driven] Epoch %d/%d - Loss: %.6f\n', ...
            epoch, train_params.num_epochs, epoch_loss.total);
    end
end

fprintf('Data-driven NN reproduction training completed.\n\n');

%% 6. Final evaluation
fprintf('\nFinal evaluation\n');

Y_test_pred_pinn       = forward_pass(net_pinn, X_norm_test);
Y_test_pred_pinn_fixed = forward_pass(net_pinn_fixed, X_norm_test);
Y_test_pred_data       = forward_pass(net_data, X_norm_test);
Y_chan                 = compute_by_chan_algorithm(X_raw_test, params, train_params);

errors_pinn       = Y_test_pred_pinn       - Y_true_test;
errors_pinn_fixed = Y_test_pred_pinn_fixed - Y_true_test;
errors_data       = Y_test_pred_data       - Y_true_test;
errors_chan       = Y_chan                 - Y_true_test;

mae_pinn = [mean(abs(errors_pinn(:,1))), ...
            mean(abs(errors_pinn(:,2))), ...
            mean(abs(errors_pinn(:,3)))];

mae_pinn_fixed = [mean(abs(errors_pinn_fixed(:,1))), ...
                  mean(abs(errors_pinn_fixed(:,2))), ...
                  mean(abs(errors_pinn_fixed(:,3)))];

mae_data = [mean(abs(errors_data(:,1))), ...
            mean(abs(errors_data(:,2))), ...
            mean(abs(errors_data(:,3)))];

mae_chan = [mean(abs(errors_chan(:,1))), ...
            mean(abs(errors_chan(:,2))), ...
            mean(abs(errors_chan(:,3)))];

fprintf('APINN:\n');
fprintf('  Distance MAE  = %.4f m\n', mae_pinn(1));
fprintf('  Azimuth MAE   = %.2f deg\n', rad2deg(mae_pinn(2)));
fprintf('  Elevation MAE = %.2f deg\n\n', rad2deg(mae_pinn(3)));

fprintf('Fixed-PINN:\n');
fprintf('  Distance MAE  = %.4f m\n', mae_pinn_fixed(1));
fprintf('  Azimuth MAE   = %.2f deg\n', rad2deg(mae_pinn_fixed(2)));
fprintf('  Elevation MAE = %.2f deg\n\n', rad2deg(mae_pinn_fixed(3)));

fprintf('Data-driven NN:\n');
fprintf('  Distance MAE  = %.4f m\n', mae_data(1));
fprintf('  Azimuth MAE   = %.2f deg\n', rad2deg(mae_data(2)));
fprintf('  Elevation MAE = %.2f deg\n\n', rad2deg(mae_data(3)));

fprintf('Analytical:\n');
fprintf('  Distance MAE  = %.4f m\n', mae_chan(1));
fprintf('  Azimuth MAE   = %.2f deg\n', rad2deg(mae_chan(2)));
fprintf('  Elevation MAE = %.2f deg\n\n', rad2deg(mae_chan(3)));

%% 7. Compute 3D error on the test set
to_cart = @(Y) [Y(:,1).*cos(Y(:,2)).*cos(Y(:,3)), ...
                Y(:,1).*sin(Y(:,2)).*cos(Y(:,3)), ...
                Y(:,1).*sin(Y(:,3))];

pos_true = to_cart(Y_true_test);
pos_pred_pinn       = to_cart(Y_test_pred_pinn);
pos_pred_pinn_fixed = to_cart(Y_test_pred_pinn_fixed);
pos_pred_data       = to_cart(Y_test_pred_data);
pos_pred_chan       = to_cart(Y_chan);

error_pinn       = sqrt(sum((pos_true - pos_pred_pinn).^2, 2));
error_pinn_fixed = sqrt(sum((pos_true - pos_pred_pinn_fixed).^2, 2));
error_data       = sqrt(sum((pos_true - pos_pred_data).^2, 2));
error_chan       = sqrt(sum((pos_true - pos_pred_chan).^2, 2));

fprintf('APINN mean 3D error      = %.4f m\n', mean(error_pinn));
fprintf('Fixed-PINN mean 3D error = %.4f m\n', mean(error_pinn_fixed));
fprintf('Data-NN mean 3D error    = %.4f m\n', mean(error_data));
fprintf('Analytical mean 3D error = %.4f m\n\n', mean(error_chan));

%% 8. Fig. 6: MAE comparison and improvement percentage
set(0, 'DefaultAxesFontName', 'Times New Roman');
set(0, 'DefaultAxesFontSize', 16);
set(0, 'DefaultTextFontName', 'Times New Roman');
set(0, 'DefaultTextFontSize', 14);

color_blue    = [0, 0.447, 0.741];
color_red     = [0.85, 0.325, 0.098];
color_green   = [0.466, 0.674, 0.188];
color_purple  = [0.494, 0.184, 0.556];
color_orange  = [0.929, 0.694, 0.125];
color_gray    = [0.5, 0.5, 0.5];
c_formula     = color_red;
c_data        = color_green;
c_pinn        = color_blue;
c_gray        = color_gray;

categories = {'Distance (m)', 'Azimuth (°)', 'Elevation (°)'};

mae_pinn_plot       = [mae_pinn(1),       rad2deg(mae_pinn(2)),       rad2deg(mae_pinn(3))];
mae_pinn_fixed_plot = [mae_pinn_fixed(1), rad2deg(mae_pinn_fixed(2)), rad2deg(mae_pinn_fixed(3))];
mae_data_plot       = [mae_data(1),       rad2deg(mae_data(2)),       rad2deg(mae_data(3))];
mae_chan_plot       = [mae_chan(1),       rad2deg(mae_chan(2)),       rad2deg(mae_chan(3))];

improvement_apinn_vs_fixed = (mae_pinn_fixed_plot - mae_pinn_plot) ./ mae_pinn_fixed_plot * 100;
improvement_apinn_vs_data  = (mae_data_plot       - mae_pinn_plot) ./ mae_data_plot       * 100;
improvement_apinn_vs_chan  = (mae_chan_plot       - mae_pinn_plot) ./ mae_chan_plot       * 100;

x = 1:3;

fig6 = figure('Position', [120, 120, 1280, 460], 'Color', 'w', 'Renderer', 'painters');
t = tiledlayout(fig6, 1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

% (a) MAE comparison
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
ax1.YGrid = 'on';
ax1.XGrid = 'off';
ax1.GridAlpha = 0.15;
ax1.MinorGridAlpha = 0.08;
ax1.LineWidth = 1.2;
ax1.TickDir = 'out';
ax1.Layer = 'top';
ax1.FontName = 'Times New Roman';
ax1.FontSize = 16;

lgd1 = legend(ax1, {'Analytical','Data-driven NN','Fixed-PINN','APINN'}, ...
    'Location','northoutside','Orientation','horizontal');
lgd1.Box = 'off';
lgd1.NumColumns = 4;
lgd1.FontSize = 12;

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

% (b) Improvement percentage
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
ax2.YGrid = 'on';
ax2.XGrid = 'off';
ax2.GridAlpha = 0.15;
ax2.LineWidth = 1.2;
ax2.TickDir = 'out';
ax2.Layer = 'top';
ax2.FontName = 'Times New Roman';
ax2.FontSize = 14;

lgd2 = legend(ax2, {'APINN vs Analytical','APINN vs Data-NN','APINN vs Fixed-PINN'}, ...
    'Location','northoutside','Orientation','horizontal');
lgd2.Box = 'off';
lgd2.NumColumns = 3;
lgd2.FontSize = 12;

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


%% 9. Fig. 7: Localization error curves at different distances
fig7 = figure('Position', [120, 120, 980, 520], 'Color', 'w', 'Renderer', 'painters');
ax = axes(fig7); hold(ax,'on'); box(ax,'on');

d_true = Y_true_test(:,1);
dmin = 0.2;
dmax = 50;
dq = linspace(dmin, dmax, 400)';

% Unified fallback smoothing without extra custom dependencies
okC = isfinite(d_true) & isfinite(error_chan) & d_true>=dmin & d_true<=dmax & error_chan>=0;
okD = isfinite(d_true) & isfinite(error_data) & d_true>=dmin & d_true<=dmax & error_data>=0;
okP = isfinite(d_true) & isfinite(error_pinn) & d_true>=dmin & d_true<=dmax & error_pinn>=0;
okF = isfinite(d_true) & isfinite(error_pinn_fixed) & d_true>=dmin & d_true<=dmax & error_pinn_fixed>=0;

dC = d_true(okC); eC = error_chan(okC);
dD = d_true(okD); eD = error_data(okD);
dP = d_true(okP); eP = error_pinn(okP);
dF = d_true(okF); eF = error_pinn_fixed(okF);

[dC, idx] = sort(dC); eC = eC(idx);
[dD, idx] = sort(dD); eD = eD(idx);
[dP, idx] = sort(dP); eP = eP(idx);
[dF, idx] = sort(dF); eF = eF(idx);

[dC, iu] = unique(dC, 'stable'); eC = eC(iu);
[dD, iu] = unique(dD, 'stable'); eD = eD(iu);
[dP, iu] = unique(dP, 'stable'); eP = eP(iu);
[dF, iu] = unique(dF, 'stable'); eF = eF(iu);

winC = max(15, round(0.06 * numel(dC)));
winD = max(15, round(0.06 * numel(dD)));
winP = max(15, round(0.06 * numel(dP)));
winF = max(15, round(0.06 * numel(dF)));

eC_sm = smoothdata(eC, 'movmean', winC);
eD_sm = smoothdata(eD, 'movmean', winD);
eP_sm = smoothdata(eP, 'movmean', winP);
eF_sm = smoothdata(eF, 'movmean', winF);

yC = interp1(dC, eC_sm, dq, 'pchip', 'extrap'); yC = max(yC, 0);
yD = interp1(dD, eD_sm, dq, 'pchip', 'extrap'); yD = max(yD, 0);
yP = interp1(dP, eP_sm, dq, 'pchip', 'extrap'); yP = max(yP, 0);
yF = interp1(dF, eF_sm, dq, 'pchip', 'extrap'); yF = max(yF, 0);

hC = plot(ax, dq, yC, 'LineWidth', 3.0, 'Color', color_red,    'DisplayName', 'Analytical');
hD = plot(ax, dq, yD, 'LineWidth', 3.0, 'Color', color_green,  'DisplayName', 'Data-driven NN');
hF = plot(ax, dq, yF, 'LineWidth', 3.0, 'Color', color_purple, 'DisplayName', 'Fixed-PINN');
hP = plot(ax, dq, yP, 'LineWidth', 3.0, 'Color', color_blue,   'DisplayName', 'APINN');

xlabel(ax, 'True Distance (m)', 'FontWeight', 'bold', 'FontSize', 18);
ylabel(ax, '3D Position Error (m)', 'FontWeight', 'bold', 'FontSize', 18);

xlim(ax, [dmin, dmax]);
ymax = max([yC; yD; yP; yF], [], 'omitnan');
ylim(ax, [0, 1.12*ymax]);

grid(ax,'on');
ax.XMinorTick = 'on';
ax.YMinorTick = 'on';
ax.MinorGridLineStyle = ':';
ax.GridAlpha = 0.15;
ax.MinorGridAlpha = 0.08;
ax.LineWidth = 1.2;
ax.TickDir = 'out';
ax.Layer = 'top';
ax.FontName = 'Times New Roman';
ax.FontSize = 16;

lgd = legend(ax, [hC,hD,hF,hP], 'Location','northoutside', 'Orientation','horizontal');
lgd.NumColumns = 4;
lgd.Box = 'off';
lgd.FontSize = 15;


%% 10. Trajectory test reproduction
fprintf('\nTrajectory test reproduction\n');

Y_pred_traj_apinn = forward_pass(net_pinn, X_norm_traj);
Y_pred_traj_fixed = forward_pass(net_pinn_fixed, X_norm_traj);
Y_pred_traj_data  = forward_pass(net_data, X_norm_traj);
Y_pred_traj_chan  = compute_by_chan_algorithm(X_raw_traj, params, train_params);

to_cart_traj = @(Y) [Y(:,1).*cos(Y(:,2)).*cos(Y(:,3)), ...
                     Y(:,1).*sin(Y(:,2)).*cos(Y(:,3)), ...
                     Y(:,1).*sin(Y(:,3))];

pos_true_traj  = to_cart_traj(Y_true_traj);
pos_apinn_traj = to_cart_traj(Y_pred_traj_apinn);
pos_fixed_traj = to_cart_traj(Y_pred_traj_fixed);
pos_data_traj  = to_cart_traj(Y_pred_traj_data);
pos_chan_traj  = to_cart_traj(Y_pred_traj_chan);

win = 21;
if mod(win,2)==0
    win = win + 1;
end

pos_apinn_sm = zeros(size(pos_apinn_traj));
pos_fixed_sm = zeros(size(pos_fixed_traj));
pos_data_sm  = zeros(size(pos_data_traj));
pos_chan_sm  = zeros(size(pos_chan_traj));

seg_list = unique(seg_id(:))';
for s = seg_list
    idx_seg = find(seg_id == s);

    p = pos_apinn_traj(idx_seg,:);
    p = smoothdata(p, 1, 'sgolay', min(win, size(p,1) - mod(size(p,1)+1,2)));
    p = smoothdata(p, 1, 'movmean', max(5, round(win/2)));
    pos_apinn_sm(idx_seg,:) = p;

    p = pos_fixed_traj(idx_seg,:);
    p = smoothdata(p, 1, 'sgolay', min(win, size(p,1) - mod(size(p,1)+1,2)));
    p = smoothdata(p, 1, 'movmean', max(5, round(win/2)));
    pos_fixed_sm(idx_seg,:) = p;

    p = pos_data_traj(idx_seg,:);
    p = smoothdata(p, 1, 'sgolay', min(win, size(p,1) - mod(size(p,1)+1,2)));
    p = smoothdata(p, 1, 'movmean', max(5, round(win/2)));
    pos_data_sm(idx_seg,:) = p;

    p = pos_chan_traj(idx_seg,:);
    p = smoothdata(p, 1, 'sgolay', min(win, size(p,1) - mod(size(p,1)+1,2)));
    p = smoothdata(p, 1, 'movmean', max(5, round(win/2)));
    pos_chan_sm(idx_seg,:) = p;
end

err_apinn    = sqrt(sum((pos_apinn_traj - pos_true_traj).^2, 2));
err_fixed    = sqrt(sum((pos_fixed_traj - pos_true_traj).^2, 2));
err_data     = sqrt(sum((pos_data_traj  - pos_true_traj).^2, 2));
err_chan     = sqrt(sum((pos_chan_traj  - pos_true_traj).^2, 2));

err_apinn_sm = sqrt(sum((pos_apinn_sm - pos_true_traj).^2, 2));
err_fixed_sm = sqrt(sum((pos_fixed_sm - pos_true_traj).^2, 2));
err_data_sm  = sqrt(sum((pos_data_sm  - pos_true_traj).^2, 2));
err_chan_sm  = sqrt(sum((pos_chan_sm  - pos_true_traj).^2, 2));

fprintf('\nTrajectory segment error statistics (3D error, m)\n');
fprintf('%-12s | %-10s %-10s %-10s %-10s\n', 'Segment', 'Analytical', 'DataNN', 'Fixed', 'APINN');

segment_names = {'Far(45-10m)','Arc(10-3m)','Near(<3m)'};
for s = 1:3
    idx_seg = (seg_id == s);
    fprintf('%-12s | %-10.4f %-10.4f %-10.4f %-10.4f\n', ...
        segment_names{s}, ...
        mean(err_chan(idx_seg)), ...
        mean(err_data(idx_seg)), ...
        mean(err_fixed(idx_seg)), ...
        mean(err_apinn(idx_seg)));
end
fprintf('\n');

%% 11. Trajectory figure 1: 3D trajectory comparison
c_true   = [0 0 0];
c_chan   = [0.85, 0.325, 0.098];
c_data2  = [0.466, 0.674, 0.188];
c_fixed  = [0.494, 0.184, 0.556];
c_apinn2 = [0, 0.447, 0.741];
c_gray2  = [0.35 0.35 0.35];

idx12 = find(seg_id==2, 1, 'first');
idx23 = find(seg_id==3, 1, 'first');

fig_traj_3d = figure('Color','w','Position',[120 120 860 660],'Renderer','painters');
ax1 = axes(fig_traj_3d, 'Position', [0.09 0.14 0.86 0.72]);
hold(ax1,'on'); box(ax1,'on');

hT = plot3(ax1, pos_true_traj(:,1), pos_true_traj(:,2), pos_true_traj(:,3), '--', ...
    'Color', c_true, 'LineWidth', 2.0, 'DisplayName','True');

hC = plot3(ax1, pos_chan_sm(:,1), pos_chan_sm(:,2), pos_chan_sm(:,3), '-', ...
    'Color', c_chan, 'LineWidth', 2.3, 'DisplayName','Analytical');

hD = plot3(ax1, pos_data_sm(:,1), pos_data_sm(:,2), pos_data_sm(:,3), '-', ...
    'Color', c_data2, 'LineWidth', 2.3, 'DisplayName','Data-NN');

hF = plot3(ax1, pos_fixed_sm(:,1), pos_fixed_sm(:,2), pos_fixed_sm(:,3), '-', ...
    'Color', c_fixed, 'LineWidth', 2.3, 'DisplayName','Fixed-PINN');

hP = plot3(ax1, pos_apinn_sm(:,1), pos_apinn_sm(:,2), pos_apinn_sm(:,3), '-', ...
    'Color', c_apinn2, 'LineWidth', 2.8, 'DisplayName','APINN');

if ~isempty(idx12)
    plot3(ax1, pos_true_traj(idx12,1), pos_true_traj(idx12,2), pos_true_traj(idx12,3), ...
        'o', 'MarkerSize', 8, 'MarkerFaceColor', 'y', 'MarkerEdgeColor', 'k', ...
        'LineWidth', 1.0, 'HandleVisibility','off');
end
if ~isempty(idx23)
    plot3(ax1, pos_true_traj(idx23,1), pos_true_traj(idx23,2), pos_true_traj(idx23,3), ...
        'o', 'MarkerSize', 8, 'MarkerFaceColor', 'c', 'MarkerEdgeColor', 'k', ...
        'LineWidth', 1.0, 'HandleVisibility','off');
end

hS = plot3(ax1, pos_true_traj(1,1), pos_true_traj(1,2), pos_true_traj(1,3), ...
    'p', 'MarkerSize',11, 'MarkerFaceColor',c_true, 'MarkerEdgeColor','k', ...
    'LineWidth',1.0, 'LineStyle','none', 'DisplayName','Start (True)');
text(ax1, pos_true_traj(1,1), pos_true_traj(1,2), pos_true_traj(1,3), '  Start', ...
    'FontWeight','bold', 'Color','k');

hE = plot3(ax1, pos_true_traj(end,1), pos_true_traj(end,2), pos_true_traj(end,3), ...
    'h', 'MarkerSize',11, 'MarkerFaceColor',c_true, 'MarkerEdgeColor','k', ...
    'LineWidth',1.0, 'LineStyle','none', 'DisplayName','End (True)');
text(ax1, pos_true_traj(end,1), pos_true_traj(end,2), pos_true_traj(end,3), '  End', ...
    'FontWeight','bold', 'Color','k');

xlabel(ax1,'X (m)','FontWeight','bold','FontSize',16);
ylabel(ax1,'Y (m)','FontWeight','bold','FontSize',16);
zlabel(ax1,'Z (m)','FontWeight','bold','FontSize',16);

x_all = [pos_true_traj(:,1); pos_chan_sm(:,1); pos_data_sm(:,1); pos_fixed_sm(:,1); pos_apinn_sm(:,1)];
x_pad = 1.0;
xlim(ax1, [min(x_all)-x_pad, max(x_all)+x_pad]);
ylim(ax1, [-2.5, 22.5]);
zlim(ax1, [-34, 2]);

view(ax1, 34, 18);

grid(ax1,'on');
ax1.GridAlpha = 0.10;
ax1.MinorGridAlpha = 0.05;
ax1.TickDir = 'out';
ax1.LineWidth = 1.1;
ax1.FontSize = 13;
ax1.Clipping = 'off';

lgd = legend(ax1, [hT hC hD hF hP hS hE], ...
    'Location','northoutside', 'Orientation','horizontal');
lgd.NumColumns = 4;
lgd.Box = 'off';
lgd.FontSize = 11;

%% 12. Trajectory figure 2: error-distance curve
fig_traj_err = figure('Color','w','Position',[140 140 820 520],'Renderer','painters');
axE = axes(fig_traj_err); hold(axE,'on'); box(axE,'on');

d = traj.d;
dmin_plot = 0.1;
dmax_plot = 50;
dq = linspace(dmin_plot, dmax_plot, 800)';

% Inline anchored error smoothing
d_anchor = min(d);
d_taper = 1.0;

% Analytical
ok = isfinite(d) & isfinite(err_chan_sm) & d>=0 & err_chan_sm>=0;
d0 = d(ok); e0 = err_chan_sm(ok);
[d0, idx] = sort(d0); e0 = e0(idx);
[d0, iu] = unique(d0, 'stable'); e0 = e0(iu);
e0_sm = smoothdata(e0, 'movmean', max(15, round(0.06*numel(e0))));
yC = interp1(d0, e0_sm, dq, 'pchip', 'extrap'); yC = max(yC, 0);
s = (dq - d_anchor) / max(d_taper, 1e-9); s = min(max(s,0),1); w = s.^2 .* (3 - 2*s); yC = yC .* w; yC(dq <= d_anchor) = 0;

% Data-NN
ok = isfinite(d) & isfinite(err_data_sm) & d>=0 & err_data_sm>=0;
d0 = d(ok); e0 = err_data_sm(ok);
[d0, idx] = sort(d0); e0 = e0(idx);
[d0, iu] = unique(d0, 'stable'); e0 = e0(iu);
e0_sm = smoothdata(e0, 'movmean', max(15, round(0.06*numel(e0))));
yD = interp1(d0, e0_sm, dq, 'pchip', 'extrap'); yD = max(yD, 0);
s = (dq - d_anchor) / max(d_taper, 1e-9); s = min(max(s,0),1); w = s.^2 .* (3 - 2*s); yD = yD .* w; yD(dq <= d_anchor) = 0;

% Fixed-PINN
ok = isfinite(d) & isfinite(err_fixed_sm) & d>=0 & err_fixed_sm>=0;
d0 = d(ok); e0 = err_fixed_sm(ok);
[d0, idx] = sort(d0); e0 = e0(idx);
[d0, iu] = unique(d0, 'stable'); e0 = e0(iu);
e0_sm = smoothdata(e0, 'movmean', max(15, round(0.06*numel(e0))));
yF = interp1(d0, e0_sm, dq, 'pchip', 'extrap'); yF = max(yF, 0);
s = (dq - d_anchor) / max(d_taper, 1e-9); s = min(max(s,0),1); w = s.^2 .* (3 - 2*s); yF = yF .* w; yF(dq <= d_anchor) = 0;

% APINN
ok = isfinite(d) & isfinite(err_apinn_sm) & d>=0 & err_apinn_sm>=0;
d0 = d(ok); e0 = err_apinn_sm(ok);
[d0, idx] = sort(d0); e0 = e0(idx);
[d0, iu] = unique(d0, 'stable'); e0 = e0(iu);
e0_sm = smoothdata(e0, 'movmean', max(15, round(0.06*numel(e0))));
yP = interp1(d0, e0_sm, dq, 'pchip', 'extrap'); yP = max(yP, 0);
s = (dq - d_anchor) / max(d_taper, 1e-9); s = min(max(s,0),1); w = s.^2 .* (3 - 2*s); yP = yP .* w; yP(dq <= d_anchor) = 0;

plot(axE, dq, yC, '-', 'Color', c_chan,   'LineWidth', 2.6, 'DisplayName','Analytical');
plot(axE, dq, yD, '-', 'Color', c_data2,  'LineWidth', 2.6, 'DisplayName','Data-NN');
plot(axE, dq, yF, '-', 'Color', c_fixed,  'LineWidth', 2.6, 'DisplayName','Fixed-PINN');
plot(axE, dq, yP, '-', 'Color', c_apinn2, 'LineWidth', 3.0, 'DisplayName','APINN');

xlim(axE, [0.1 50]);
set(axE,'XDir','reverse');

xline(axE, 10, '--', 'Color', [0.35 0.35 0.35], 'LineWidth', 1.2, 'HandleVisibility','off');
xline(axE, 3,  '--', 'Color', [0.35 0.35 0.35], 'LineWidth', 1.2, 'HandleVisibility','off');

xlabel(axE,'True Distance d (m)','FontWeight','bold');
ylabel(axE,'3D Position Error (m)','FontWeight','bold');

grid(axE,'on');
axE.GridAlpha = 0.10;
axE.MinorGridAlpha = 0.05;
axE.TickDir = 'out';
axE.LineWidth = 1.1;
axE.FontSize = 14;

lgdE = legend(axE, 'Location','northoutside', 'Orientation','horizontal');
lgdE.NumColumns = 4;
lgdE.Box = 'off';
lgdE.FontSize = 13;




