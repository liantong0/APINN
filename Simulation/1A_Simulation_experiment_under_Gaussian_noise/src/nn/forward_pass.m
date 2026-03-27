function [Y_pred, activations] = forward_pass(net, X)
% Compute network outputs and store layer activations during the forward pass.

    num_layers = length(net.W);
    activations = cell(num_layers + 1, 1);
    activations{1} = X';
    
    for i = 1:(num_layers - 1)
        z = net.W{i} * activations{i} + net.b{i};
        activations{i + 1} = max(0, z);
    end
    
    z_out = net.W{num_layers} * activations{num_layers} + net.b{num_layers};
    activations{num_layers + 1} = z_out;
    y_tilde = z_out';
    
    Y_pred = zeros(size(y_tilde));
    Y_pred(:, 1) = 25 * (1 + tanh(y_tilde(:, 1)));
    Y_pred(:, 2) = (pi / 3) * tanh(y_tilde(:, 2));
    Y_pred(:, 3) = (pi / 4) * tanh(y_tilde(:, 3));
end
