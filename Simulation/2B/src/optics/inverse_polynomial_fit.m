function delta_0 = inverse_polynomial_fit(delta)
% Invert the calibration polynomial to recover normalized offset from physical offset.
    poly_func = @(x) 0.8835*x^5 + 0.4699*x^3 + 2.0737*x - delta;
    options = optimset('Display', 'off', 'TolX', 1e-10);
    initial_guess = delta / 2.0737;
    try
        delta_0 = fzero(poly_func, initial_guess, options);
    catch
        delta_0 = delta / 2.0737;
    end
    delta_0 = max(min(delta_0, 3), -3);
end

