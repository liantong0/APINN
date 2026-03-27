function n = noise_alpha_stable(sz, sigma, alpha, beta)
% Generate alpha-stable noise with RMS scaling.
    n0 = stbl_cms(sz, alpha, beta);
    rms0 = sqrt(mean(n0(:).^2) + 1e-12);
    n = (sigma / rms0) * n0;
end
