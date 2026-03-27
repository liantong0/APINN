function delta = apply_polynomial_fit(delta_0)
% Map normalized detector offset through the fitted polynomial model.
    delta = 0.8835*delta_0^5 + 0.4699*delta_0^3 + 2.0737*delta_0;
end
