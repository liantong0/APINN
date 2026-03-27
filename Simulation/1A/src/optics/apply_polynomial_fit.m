function delta = apply_polynomial_fit(delta_0)
% Apply the polynomial mapping from normalized offset to corrected offset.

    delta = 0.8835 * delta_0^5 + 0.4699 * delta_0^3 + 2.0737 * delta_0;
end
