function net = initialize_network(layers)
% Initialize network weights and biases using He initialization.

    num_layers = length(layers) - 1;
    net = struct('W', {{}}, 'b', {{}});
    for i = 1:num_layers
        net.W{i} = randn(layers(i + 1), layers(i)) * sqrt(2 / layers(i));
        net.b{i} = zeros(layers(i + 1), 1);
    end
end
