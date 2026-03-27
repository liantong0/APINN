function alpha_prime = compute_alpha_prime_from_a2_paper(a2, params)
% Compute the refracted angle alpha_prime from a2 using the optical model in the paper.

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
end
