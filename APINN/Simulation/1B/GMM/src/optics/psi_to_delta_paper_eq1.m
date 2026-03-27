function delta_m = psi_to_delta_paper_eq1(psi_l, params)
% Map optical angle to detector offset based on the paper model.
    sin_alpha_prime = params.n_w * sin(psi_l) / params.n_a;
    sin_alpha_prime = max(min(sin_alpha_prime, 1), -1);
    alpha_prime = asin(sin_alpha_prime);

    options = optimset('Display', 'off', 'TolX', 1e-10);
    equation = @(a2) compute_alpha_prime_from_a2_paper(a2, params) - alpha_prime;
    a2_init = alpha_prime;

    try
        a2 = fzero(equation, a2_init, options);
    catch
        a2 = alpha_prime * params.n_g / params.n_a;
    end

    delta_m = params.f_detector * tan(a2);
end
