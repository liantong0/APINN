clear; clc; close all;
setup_paths();

%% Parameter notes
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
% params.hydrophone_positions = [
%     0, 0, 0.004;
%     0.005, 0, 0;
%     -0.005, 0.004, 0;
%     0, -0.004, -0.004
% ];
%
% params.u_scale = params.f_U * tan(pi/3);
% params.v_scale = params.f_V * tan(pi/4);
%
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
% train_params.output_weights = [8.0, 19.0, 1.0];
%
% train_params.lambda_tdoa = 3.0;
% train_params.lambda_depth = 5.0;
% train_params.lambda_consistency = 3.0;
% train_params.lambda_camera = 4;
% train_params.lambda_detector = 2;
% train_params.lambda_smooth = 0.1;
%
% params.turbidity = 10;
% params.turbidity_coefficient = 0.2;

%% Load reproducible state
load('Full_Distance_Repro_State.mat', ...
    'rng_state_at_training_start', ...
    'X_raw_train', 'Y_true_train', ...
    'X_raw_test', 'Y_true_test', ...
    'params', 'train_params', 'layers', ...
    'net_pinn', 'adam_state_pinn', ...
    'net_pinn_fixed', 'adam_state_pinn_fixed', ...
    'net_data', 'adam_state_data');

rng(rng_state_at_training_start);

train_params.num_samples = size(X_raw_train, 1);
num_test = size(X_raw_test, 1);

%% Normalize data
fprintf('\nNormalizing training data...\n');
X_norm_train = normalize_inputs_tdoa(X_raw_train, params);

fprintf('Normalizing test data...\n');
X_norm_test = normalize_inputs_tdoa(X_raw_test, params);

%% Train APINN
fprintf('\nStart training APINN\n');
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

    for batch = 1:num_batches
        batch_start = (batch - 1) * train_params.batch_size + 1;
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

    loss_history_pinn.total(epoch) = epoch_loss.total;
    loss_history_pinn.data(epoch) = epoch_loss.data;
    loss_history_pinn.tdoa(epoch) = epoch_loss.tdoa;
    loss_history_pinn.camera(epoch) = epoch_loss.camera;
    loss_history_pinn.detector(epoch) = epoch_loss.detector;
    loss_history_pinn.depth(epoch) = epoch_loss.depth;

    if mod(epoch, 10) == 0 || epoch == 1
        Y_test_pred_epoch = forward_pass(net_pinn, X_norm_test);
        errors_epoch = Y_test_pred_epoch - Y_true_test;

        mae_d = mean(abs(errors_epoch(:, 1)));
        mae_psi = mean(abs(errors_epoch(:, 2)));
        mae_theta = mean(abs(errors_epoch(:, 3)));

        test_mae_history_pinn.distance(end + 1) = mae_d;
        test_mae_history_pinn.azimuth(end + 1) = mae_psi;
        test_mae_history_pinn.elevation(end + 1) = mae_theta;

        pos_true = [Y_true_test(:,1).*cos(Y_true_test(:,2)).*cos(Y_true_test(:,3)), ...
                    Y_true_test(:,1).*sin(Y_true_test(:,2)).*cos(Y_true_test(:,3)), ...
                    Y_true_test(:,1).*sin(Y_true_test(:,3))];
        pos_pred = [Y_test_pred_epoch(:,1).*cos(Y_test_pred_epoch(:,2)).*cos(Y_test_pred_epoch(:,3)), ...
                    Y_test_pred_epoch(:,1).*sin(Y_test_pred_epoch(:,2)).*cos(Y_test_pred_epoch(:,3)), ...
                    Y_test_pred_epoch(:,1).*sin(Y_test_pred_epoch(:,3))];
        pos_errors = sqrt(sum((pos_true - pos_pred).^2, 2));
        position_error_history_pinn(end + 1) = mean(pos_errors);
    end

    if mod(epoch, 500) == 0 || epoch == 1
        physics_loss = epoch_loss.tdoa + epoch_loss.camera + epoch_loss.detector + epoch_loss.depth;
        fprintf('[APINN] Epoch %d/%d\n', epoch, train_params.num_epochs);
        fprintf('       Total: %.6f | Data: %.6f | Physics: %.6f\n', ...
            epoch_loss.total, epoch_loss.data, physics_loss);
    end
end

fprintf('APINN training finished.\n\n');

%% Train fixed-weight PINN
fprintf('\nStart training fixed-weight PINN\n');
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
        batch_start = (batch - 1) * train_params.batch_size + 1;
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

        mae_d = mean(abs(errors_epoch(:,1)));
        mae_psi = mean(abs(errors_epoch(:,2)));
        mae_theta = mean(abs(errors_epoch(:,3)));

        test_mae_history_pinn_fixed.distance(end + 1) = mae_d;
        test_mae_history_pinn_fixed.azimuth(end + 1) = mae_psi;
        test_mae_history_pinn_fixed.elevation(end + 1) = mae_theta;

        pos_true = [Y_true_test(:,1).*cos(Y_true_test(:,2)).*cos(Y_true_test(:,3)), ...
                    Y_true_test(:,1).*sin(Y_true_test(:,2)).*cos(Y_true_test(:,3)), ...
                    Y_true_test(:,1).*sin(Y_true_test(:,3))];
        pos_pred = [Y_test_pred_epoch(:,1).*cos(Y_test_pred_epoch(:,2)).*cos(Y_test_pred_epoch(:,3)), ...
                    Y_test_pred_epoch(:,1).*sin(Y_test_pred_epoch(:,2)).*cos(Y_test_pred_epoch(:,3)), ...
                    Y_test_pred_epoch(:,1).*sin(Y_test_pred_epoch(:,3))];
        pos_errors = sqrt(sum((pos_true - pos_pred).^2, 2));
        position_error_history_pinn_fixed(end + 1) = mean(pos_errors);
    end

    if mod(epoch, 500) == 0 || epoch == 1
        physics_loss = epoch_loss.tdoa + epoch_loss.camera + epoch_loss.detector + epoch_loss.depth;
        fprintf('[Fixed-PINN] Epoch %d/%d\n', epoch, train_params.num_epochs);
        fprintf('       Total: %.6f | Data: %.6f | Physics: %.6f\n', ...
            epoch_loss.total, epoch_loss.data, physics_loss);
    end
end

fprintf('Fixed-weight PINN training finished.\n\n');

%% Train data-driven neural network
fprintf('Start training data-driven neural network\n');
loss_history_data = struct('total', [], 'data', []);
test_mae_history_data = struct('distance', [], 'azimuth', [], 'elevation', []);
position_error_history_data = [];
current_lr_data = train_params.learning_rate;
weight_decay = 0.01;

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
        batch_start = (batch - 1) * train_params.batch_size + 1;
        batch_end = min(batch * train_params.batch_size, train_params.num_samples);

        X_batch = X_norm_shuffled(batch_start:batch_end, :);
        Y_batch = Y_true_shuffled(batch_start:batch_end, :);

        [Y_pred, activations] = forward_pass(net_data, X_batch);

        [loss, gradients] = compute_loss_data_driven(net_data, X_batch, Y_pred, Y_batch, ...
            activations, train_params, weight_decay);

        [net_data, adam_state_data] = adam_update(net_data, gradients, adam_state_data, current_lr_data, ...
            train_params.beta1, train_params.beta2, train_params.epsilon, epoch);

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

        mae_d = mean(abs(errors_epoch(:,1)));
        mae_psi = mean(abs(errors_epoch(:,2)));
        mae_theta = mean(abs(errors_epoch(:,3)));

        test_mae_history_data.distance(end + 1) = mae_d;
        test_mae_history_data.azimuth(end + 1) = mae_psi;
        test_mae_history_data.elevation(end + 1) = mae_theta;

        pos_true = [Y_true_test(:,1).*cos(Y_true_test(:,2)).*cos(Y_true_test(:,3)), ...
                    Y_true_test(:,1).*sin(Y_true_test(:,2)).*cos(Y_true_test(:,3)), ...
                    Y_true_test(:,1).*sin(Y_true_test(:,3))];
        pos_pred = [Y_test_pred_epoch(:,1).*cos(Y_test_pred_epoch(:,2)).*cos(Y_test_pred_epoch(:,3)), ...
                    Y_test_pred_epoch(:,1).*sin(Y_test_pred_epoch(:,2)).*cos(Y_test_pred_epoch(:,3)), ...
                    Y_test_pred_epoch(:,1).*sin(Y_test_pred_epoch(:,3))];
        pos_errors = sqrt(sum((pos_true - pos_pred).^2, 2));
        position_error_history_data(end + 1) = mean(pos_errors);
    end

    if mod(epoch, 500) == 0 || epoch == 1
        fprintf('[Data-driven] Epoch %d/%d - Loss: %.6f\n', ...
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

mae_pinn = [mean(abs(errors_pinn(:,1))), mean(abs(errors_pinn(:,2))), mean(abs(errors_pinn(:,3)))];
mae_pinn_fixed = [mean(abs(errors_pinn_fixed(:,1))), mean(abs(errors_pinn_fixed(:,2))), mean(abs(errors_pinn_fixed(:,3)))];
mae_data = [mean(abs(errors_data(:,1))), mean(abs(errors_data(:,2))), mean(abs(errors_data(:,3)))];
mae_chan = [mean(abs(errors_chan(:,1))), mean(abs(errors_chan(:,2))), mean(abs(errors_chan(:,3)))];

fprintf('\nTest set performance comparison\n');
fprintf('APINN (adaptive weights):\n');
fprintf('  Distance MAE:  %.4f m\n', mae_pinn(1));
fprintf('  Azimuth MAE:   %.2f°\n', rad2deg(mae_pinn(2)));
fprintf('  Elevation MAE: %.2f°\n', rad2deg(mae_pinn(3)));

fprintf('\nFixed-weight PINN:\n');
fprintf('  Distance MAE:  %.4f m\n', mae_pinn_fixed(1));
fprintf('  Azimuth MAE:   %.2f°\n', rad2deg(mae_pinn_fixed(2)));
fprintf('  Elevation MAE: %.2f°\n', rad2deg(mae_pinn_fixed(3)));

fprintf('\nData-driven neural network:\n');
fprintf('  Distance MAE:  %.4f m\n', mae_data(1));
fprintf('  Azimuth MAE:   %.2f°\n', rad2deg(mae_data(2)));
fprintf('  Elevation MAE: %.2f°\n', rad2deg(mae_data(3)));

fprintf('\nChan algorithm + optical direction finding:\n');
fprintf('  Distance MAE:  %.4f m\n', mae_chan(1));
fprintf('  Azimuth MAE:   %.2f°\n', rad2deg(mae_chan(2)));
fprintf('  Elevation MAE: %.2f°\n', rad2deg(mae_chan(3)));

%% Compute 3D position error
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

fprintf('\nFixed-PINN mean 3D position error: %.4f m\n', mean(error_pinn_fixed));

%% Visualization settings
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
color_gray = [0.5, 0.5, 0.5];

c_formula = color_red;
c_data = color_green;
c_pinn = color_blue;
c_gray = color_gray;

%% Fig. 1: PINN loss curves and 3D position error evolution
fig1 = figure('Position', [100, 100, 1400, 600], 'Color', 'w');

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
pos_epoch_points = (0:(n_pos_points - 1)) * 10 + 1;

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

text(train_params.num_epochs * 0.55, ...
    max([position_error_history_pinn, position_error_history_pinn_fixed, position_error_history_data]) * 0.75, ...
    sprintf('Final Error:\nAPINN: %.3f m\nFixed-PINN: %.3f m\nData-NN: %.3f m', ...
    position_error_history_pinn(end), position_error_history_pinn_fixed(end), position_error_history_data(end)), ...
    'FontSize', 15, 'FontWeight', 'bold', ...
    'BackgroundColor', 'white', 'EdgeColor', 'k', 'LineWidth', 1.5);

hold off;

%% Fig. 2: MAE comparison and improvement percentage
categories = {'Distance (m)', 'Azimuth (°)', 'Elevation (°)'};

mae_pinn_plot = [mae_pinn(1), rad2deg(mae_pinn(2)), rad2deg(mae_pinn(3))];
mae_pinn_fixed_plot = [mae_pinn_fixed(1), rad2deg(mae_pinn_fixed(2)), rad2deg(mae_pinn_fixed(3))];
mae_data_plot = [mae_data(1), rad2deg(mae_data(2)), rad2deg(mae_data(3))];
mae_chan_plot = [mae_chan(1), rad2deg(mae_chan(2)), rad2deg(mae_chan(3))];

improvement_apinn_vs_fixed = (mae_pinn_fixed_plot - mae_pinn_plot) ./ mae_pinn_fixed_plot * 100;
improvement_apinn_vs_data  = (mae_data_plot - mae_pinn_plot) ./ mae_data_plot * 100;
improvement_apinn_vs_chan  = (mae_chan_plot - mae_pinn_plot) ./ mae_chan_plot * 100;

x = 1:3;

fig6 = figure('Position', [120, 120, 1280, 460], 'Color', 'w', 'Renderer', 'painters');
t = tiledlayout(fig6, 1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

ax1 = nexttile(t, 1); hold(ax1, 'on'); box(ax1, 'on');
data_mae = [mae_chan_plot(:), mae_data_plot(:), mae_pinn_fixed_plot(:), mae_pinn_plot(:)];
b = bar(ax1, x, data_mae, 'grouped', 'BarWidth', 0.85);
b(1).FaceColor = c_formula;      b(1).EdgeColor = 'none';
b(2).FaceColor = c_data;         b(2).EdgeColor = 'none';
b(3).FaceColor = color_purple;   b(3).EdgeColor = 'none';
b(4).FaceColor = c_pinn;         b(4).EdgeColor = 'none';

set(ax1, 'XTick', x, 'XTickLabel', categories);
ylabel(ax1, 'Mean Absolute Error', 'FontWeight', 'bold', 'FontSize', 18);
grid(ax1, 'on');
ax1.YGrid = 'on'; ax1.XGrid = 'off';
ax1.GridAlpha = 0.15; ax1.MinorGridAlpha = 0.08;
ax1.LineWidth = 1.2;
ax1.TickDir = 'out';
ax1.Layer = 'top';
ax1.FontName = 'Times New Roman';
ax1.FontSize = 16;

lgd1 = legend(ax1, {'Analytical', 'Data-driven NN', 'Fixed-PINN', 'APINN'}, ...
    'Location', 'northoutside', 'Orientation', 'horizontal');
lgd1.Box = 'off';
lgd1.NumColumns = 4;
lgd1.FontSize = 14;

ymax1 = max(data_mae(:));
dy1 = 0.02 * ymax1;
for i = 1:numel(x)
    for j = 1:4
        text(ax1, b(j).XEndPoints(i), b(j).YEndPoints(i) + dy1, ...
            sprintf('%.2f', data_mae(i, j)), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
            'FontSize', 11, 'FontWeight', 'bold', 'Color', c_gray);
    end
end
ylim(ax1, [0, ymax1 * 1.25]);

ax2 = nexttile(t, 2); hold(ax2, 'on'); box(ax2, 'on');
data_imp = [improvement_apinn_vs_chan(:), improvement_apinn_vs_data(:), improvement_apinn_vs_fixed(:)];
b2 = bar(ax2, x, data_imp, 'grouped', 'BarWidth', 0.85);
b2(1).FaceColor = color_orange;
b2(2).FaceColor = color_green;
b2(3).FaceColor = color_purple;

set(ax2, 'XTick', x, 'XTickLabel', categories);
ylabel(ax2, 'Improvement (%)', 'FontWeight', 'bold', 'FontSize', 18);
yline(ax2, 0, '-', 'LineWidth', 1.0, 'Color', [0.2 0.2 0.2], 'HandleVisibility', 'off');

grid(ax2, 'on');
ax2.YGrid = 'on'; ax2.XGrid = 'off';
ax2.GridAlpha = 0.15;
ax2.LineWidth = 1.2;
ax2.TickDir = 'out';
ax2.Layer = 'top';
ax2.FontName = 'Times New Roman';
ax2.FontSize = 14;

lgd2 = legend(ax2, {'APINN vs Analytical', 'APINN vs Data-NN', 'APINN vs Fixed-PINN'}, ...
    'Location', 'northoutside', 'Orientation', 'horizontal');
lgd2.Box = 'off';
lgd2.NumColumns = 3;
lgd2.FontSize = 14;

ymax2 = max(data_imp(:));
ymin2 = min(data_imp(:));
pad2 = 0.06 * max(1, max(abs([ymax2, ymin2])));
for i = 1:numel(x)
    for j = 1:3
        val = data_imp(i, j);
        ytxt = val + sign(val + 1e-9) * pad2;
        text(ax2, b2(j).XEndPoints(i), ytxt, ...
            sprintf('%.1f%%', val), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
            'FontSize', 11, 'FontWeight', 'bold', 'Color', c_gray);
    end
end
ylim(ax2, [min(0, ymin2 - 2 * pad2), ymax2 + 2.5 * pad2]);

%% Fig. 3: Localization error versus distance
fig7 = figure('Position', [120, 120, 980, 520], 'Color', 'w', 'Renderer', 'painters');
ax = axes(fig7); hold(ax, 'on'); box(ax, 'on');

d_true = Y_true_test(:,1);
dmin = 0.1; dmax = 50;
dq = linspace(dmin, dmax, 400)';

useSpline = (exist('fit', 'file') == 2);

if useSpline
    okC = isfinite(d_true) & isfinite(error_chan) & d_true >= dmin & d_true <= dmax & error_chan >= 0;
    okD = isfinite(d_true) & isfinite(error_data) & d_true >= dmin & d_true <= dmax & error_data >= 0;
    okP = isfinite(d_true) & isfinite(error_pinn) & d_true >= dmin & d_true <= dmax & error_pinn >= 0;
    okF = isfinite(d_true) & isfinite(error_pinn_fixed) & d_true >= dmin & d_true <= dmax & error_pinn_fixed >= 0;

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
ylim(ax, [0, 1.12 * ymax]);

grid(ax, 'on');
ax.XMinorTick = 'on'; ax.YMinorTick = 'on';
ax.MinorGridLineStyle = ':';
ax.GridAlpha = 0.15; ax.MinorGridAlpha = 0.08;
ax.LineWidth = 1.2;
ax.TickDir = 'out';
ax.Layer = 'top';
ax.FontName = 'Times New Roman';
ax.FontSize = 16;

lgd = legend(ax, [hC, hD, hF, hP], 'Location', 'northoutside', 'Orientation', 'horizontal');
lgd.NumColumns = 4;
lgd.Box = 'off';
lgd.FontSize = 15;

%% Fig. 4: Error CDF
fig9 = figure('Position', [100, 100, 1000, 650], 'Color', 'w', 'Renderer', 'painters');
ax = axes(fig9); hold(ax, 'on'); box(ax, 'on');

e_chan = error_chan(:); e_chan = e_chan(isfinite(e_chan) & e_chan >= 0);
e_data = error_data(:); e_data = e_data(isfinite(e_data) & e_data >= 0);
e_fixed = error_pinn_fixed(:); e_fixed = e_fixed(isfinite(e_fixed) & e_fixed >= 0);
e_pinn = error_pinn(:); e_pinn = e_pinn(isfinite(e_pinn) & e_pinn >= 0);

[f_chan, x_chan] = ecdf(e_chan);
[f_data, x_data] = ecdf(e_data);
[f_fixed, x_fixed] = ecdf(e_fixed);
[f_pinn, x_pinn] = ecdf(e_pinn);

h_chan = plot(ax, x_chan, f_chan, 'LineWidth', 3.0, 'Color', color_red, 'DisplayName', 'Formula');
h_data = plot(ax, x_data, f_data, 'LineWidth', 3.0, 'Color', color_green, 'DisplayName', 'Data-driven NN');
h_fixed = plot(ax, x_fixed, f_fixed, 'LineWidth', 3.0, 'Color', color_purple, 'DisplayName', 'Fixed-PINN');
h_pinn = plot(ax, x_pinn, f_pinn, 'LineWidth', 3.2, 'Color', color_blue, 'DisplayName', 'APINN');

percentiles = [50, 90, 95];
p_chan = prctile(e_chan, percentiles);
p_data = prctile(e_data, percentiles);
p_fixed = prctile(e_fixed, percentiles);
p_pinn = prctile(e_pinn, percentiles);

markers = {'o', 's', 'd'};
marker_sizes = [12, 12, 13];

for i = 1:3
   plot(ax, p_chan(i), percentiles(i) / 100, markers{i}, ...
       'MarkerSize', marker_sizes(i), 'MarkerFaceColor', color_red, ...
       'MarkerEdgeColor', 'w', 'LineWidth', 1.5, 'HandleVisibility', 'off');
   plot(ax, p_data(i), percentiles(i) / 100, markers{i}, ...
       'MarkerSize', marker_sizes(i), 'MarkerFaceColor', color_green, ...
       'MarkerEdgeColor', 'w', 'LineWidth', 1.5, 'HandleVisibility', 'off');
   plot(ax, p_fixed(i), percentiles(i) / 100, markers{i}, ...
       'MarkerSize', marker_sizes(i), 'MarkerFaceColor', color_purple, ...
       'MarkerEdgeColor', 'w', 'LineWidth', 1.5, 'HandleVisibility', 'off');
   plot(ax, p_pinn(i), percentiles(i) / 100, markers{i}, ...
       'MarkerSize', marker_sizes(i), 'MarkerFaceColor', color_blue, ...
       'MarkerEdgeColor', 'w', 'LineWidth', 1.5, 'HandleVisibility', 'off');
end

x_max = max([prctile(e_chan, 99), prctile(e_data, 99), prctile(e_fixed, 99), prctile(e_pinn, 99)]);
text_offset_x = x_max * 0.035;
text_offset_y = 0.03;

for i = 1:3
   y_pos = percentiles(i) / 100 - text_offset_y * (1.2 + 0.2 * i);
   if i == 2
       y_pos = percentiles(i) / 100 - text_offset_y * 2.2;
   end
   text(ax, p_chan(i) + text_offset_x, y_pos, ...
       sprintf(' P%d: %.3fm %s', percentiles(i), p_chan(i), markers{i}), ...
       'FontSize', 14, 'FontWeight', 'bold', 'Color', color_red, ...
       'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', ...
       'EdgeColor', 'none', 'Margin', 2);
end

for i = 1:3
   x_pos = p_pinn(i) + text_offset_x * 6;
   if i == 2
       y_pos = percentiles(i) / 100 - text_offset_y * 1.2;
   else
       y_pos = percentiles(i) / 100;
   end
   text(ax, x_pos, y_pos, ...
       sprintf('P%d: %.3fm %s ', percentiles(i), p_pinn(i), markers{i}), ...
       'FontSize', 14, 'FontWeight', 'bold', 'Color', color_blue, ...
       'VerticalAlignment', 'middle', 'HorizontalAlignment', 'right', ...
       'EdgeColor', 'none', 'Margin', 2);
end

for i = 1:3
   x_pos = p_pinn(i) + text_offset_x * 6;
   if i == 2
       y_pos = percentiles(i) / 100 - text_offset_y * 2.2;
   else
       y_pos = percentiles(i) / 100 - text_offset_y;
   end
   text(ax, x_pos, y_pos, ...
       sprintf('P%d: %.3fm %s ', percentiles(i), p_fixed(i), markers{i}), ...
       'FontSize', 14, 'FontWeight', 'bold', 'Color', color_purple, ...
       'VerticalAlignment', 'middle', 'HorizontalAlignment', 'right', ...
       'EdgeColor', 'none', 'Margin', 2);
end

for i = 1:3
   x_pos = p_pinn(i) + text_offset_x * 6;
   if i == 2
       y_pos = percentiles(i) / 100 - text_offset_y * 3.2;
   else
       y_pos = percentiles(i) / 100 - text_offset_y * 2;
   end
   text(ax, x_pos, y_pos, ...
       sprintf('P%d: %.3fm %s ', percentiles(i), p_data(i), markers{i}), ...
       'FontSize', 14, 'FontWeight', 'bold', 'Color', color_green, ...
       'VerticalAlignment', 'middle', 'HorizontalAlignment', 'right', ...
       'EdgeColor', 'none', 'Margin', 2);
end

for p = percentiles
   yline(ax, p / 100, ':', 'LineWidth', 1.0, 'Color', [0.7 0.7 0.7], ...
       'Alpha', 0.3, 'HandleVisibility', 'off');
end

xlabel(ax, '3D Position Error (m)', 'FontWeight', 'bold', 'FontSize', 18);
ylabel(ax, 'Cumulative Distribution Function (CDF)', 'FontWeight', 'bold', 'FontSize', 18);
ylim(ax, [0, 1]);
xlim(ax, [0, x_max * 1.05]);

grid(ax, 'on');
ax.XMinorTick = 'on'; ax.YMinorTick = 'on';
ax.MinorGridLineStyle = ':';
ax.GridAlpha = 0.18; ax.MinorGridAlpha = 0.10;
ax.LineWidth = 1.5;
ax.TickDir = 'out';
ax.Layer = 'top';
ax.FontName = 'Times New Roman';
ax.FontSize = 16;

lgd = legend(ax, [h_chan, h_data, h_fixed, h_pinn], 'Location', 'southeast');
lgd.Box = 'on';
lgd.FontSize = 15;
lgd.LineWidth = 1.2;

hold(ax, 'off');

fprintf('\nAll visualizations have been completed.\n');