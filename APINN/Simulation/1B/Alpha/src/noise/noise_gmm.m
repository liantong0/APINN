function n = noise_gmm(sz, sigma, eps, kappa)
% Generate Gaussian mixture noise with background and impulsive components.
% n ~ (1-eps)*N(0,sigma^2) + eps*N(0,(kappa*sigma)^2)
    mask = rand(sz) < eps;
    n = randn(sz) .* (sigma .* ones(sz));
    n(mask) = randn(nnz(mask), 1) .* (kappa * sigma);
end
