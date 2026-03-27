function [net, adam_state] = adam_update(net, gradients, adam_state, lr, beta1, beta2, epsilon, t)
% Update network parameters with the Adam optimization rule.
    num_layers = length(net.W);
    for i = 1:num_layers
        adam_state.m_W{i} = beta1 * adam_state.m_W{i} + (1 - beta1) * gradients.W{i};
        adam_state.m_b{i} = beta1 * adam_state.m_b{i} + (1 - beta1) * gradients.b{i};
        
        adam_state.v_W{i} = beta2 * adam_state.v_W{i} + (1 - beta2) * (gradients.W{i}.^2);
        adam_state.v_b{i} = beta2 * adam_state.v_b{i} + (1 - beta2) * (gradients.b{i}.^2);
        
        m_W_hat = adam_state.m_W{i} / (1 - beta1^t);
        m_b_hat = adam_state.m_b{i} / (1 - beta1^t);
        v_W_hat = adam_state.v_W{i} / (1 - beta2^t);
        v_b_hat = adam_state.v_b{i} / (1 - beta2^t);
        
        net.W{i} = net.W{i} - lr * m_W_hat ./ (sqrt(v_W_hat) + epsilon);
        net.b{i} = net.b{i} - lr * m_b_hat ./ (sqrt(v_b_hat) + epsilon);
    end
end

