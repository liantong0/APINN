function n = noise_gmm(sz, sigma, eps, kappa)
% Generate Gaussian mixture noise with impulsive outliers.
    mask = rand(sz) < eps;
    n = randn(sz) .* (sigma .* ones(sz));
    n(mask) = randn(nnz(mask), 1) .* (kappa * sigma);
end
