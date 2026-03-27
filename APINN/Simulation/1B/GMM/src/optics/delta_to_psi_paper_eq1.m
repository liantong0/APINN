function psi_l = delta_to_psi_paper_eq1(delta_m, params)
% Map detector offset back to optical angle based on the paper model.
    a2 = atan(delta_m / params.f_detector);

    sin_a1 = -params.l2 * sin(a2) / params.l1;
    sin_a1 = max(min(sin_a1, 1), -1);
    a1 = asin(sin_a1);

    if abs(a1) < 1e-10
        alpha_prime = a2;
    else
        sin_term = params.n_a * sin(a1) / params.n_g;
        sin_term = max(min(sin_term, 1), -1);
        k = (a2 / a1 - 1);
        alpha_prime = a1 + k * asin(sin_term);
    end

    sin_psi_l = params.n_a * sin(alpha_prime) / params.n_w;
    sin_psi_l = max(min(sin_psi_l, 1), -1);
    psi_l = asin(sin_psi_l);
end
