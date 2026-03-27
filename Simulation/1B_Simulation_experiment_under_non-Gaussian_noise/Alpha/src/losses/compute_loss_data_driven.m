function [loss, gradients] = compute_loss_data_driven(net, X_norm, Y_pred, Y_true, activations, train_params, weight_decay)
% Compute pure data-driven loss and gradients with L2 regularization.
    batch_size = size(X_norm, 1);

    % Data loss
    diff_data = Y_pred - Y_true;
    weighted_diff = diff_data .* train_params.output_weights;
    loss_data = sum(sum(weighted_diff.^2)) / batch_size;

    loss_reg = 0;
    num_layers = length(net.W);
    for i = 1:num_layers
        loss_reg = loss_reg + sum(sum(net.W{i}.^2));
    end
    loss_reg = weight_decay * loss_reg / (2 * batch_size);

    loss.data = loss_data;
    loss.reg = loss_reg;
    loss.total = loss_data + loss_reg;

    % Gradients
    gradients = struct('W', {cell(num_layers, 1)}, 'b', {cell(num_layers, 1)});

    dL_dy = 2 * (Y_pred - Y_true) .* (train_params.output_weights.^2) / batch_size;

    y_tilde = activations{num_layers + 1}';
    dy_dy_tilde = zeros(size(y_tilde));
    dy_dy_tilde(:, 1) = 25 * (1 - tanh(y_tilde(:, 1)).^2);
    dy_dy_tilde(:, 2) = (pi/3) * (1 - tanh(y_tilde(:, 2)).^2);
    dy_dy_tilde(:, 3) = (pi/4) * (1 - tanh(y_tilde(:, 3)).^2);

    delta = (dL_dy .* dy_dy_tilde)';
    gradients.W{num_layers} = delta * activations{num_layers}' + weight_decay * net.W{num_layers} / batch_size;
    gradients.b{num_layers} = sum(delta, 2);

    for i = (num_layers - 1):-1:1
        delta = net.W{i+1}' * delta;
        delta = delta .* (activations{i+1} > 0);
        gradients.W{i} = delta * activations{i}' + weight_decay * net.W{i} / batch_size;
        gradients.b{i} = sum(delta, 2);
    end
end
