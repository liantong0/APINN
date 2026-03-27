function adam_state = initialize_adam(net)
% Initialize Adam optimizer states.
    num_layers = length(net.W);
    adam_state = struct('m_W', {{}}, 'm_b', {{}}, 'v_W', {{}}, 'v_b', {{}});
    for i = 1:num_layers
        adam_state.m_W{i} = zeros(size(net.W{i}));
        adam_state.m_b{i} = zeros(size(net.b{i}));
        adam_state.v_W{i} = zeros(size(net.W{i}));
        adam_state.v_b{i} = zeros(size(net.b{i}));
    end
end
