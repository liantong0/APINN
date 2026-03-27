function n = noise_middletonA(sz, sigma, A, Gamma, Mmax)
% Generate Middleton Class A impulsive noise and rescale its RMS.
    m = (0:Mmax)';
    pm = exp(-A) .* (A.^m) ./ factorial(m);
    pm = pm / sum(pm);

    cdf = cumsum(pm);
    r = rand(sz);
    mm = zeros(sz);
    for k = 1:numel(r)
        mm(k) = find(r(k) <= cdf, 1, 'first') - 1;
    end

    sig2 = ((mm ./ max(A, 1e-12)) + Gamma) ./ (1 + Gamma);
    n0 = randn(sz) .* sqrt(sig2);

    rms0 = sqrt(mean(n0(:).^2) + 1e-12);
    n = (sigma / rms0) * n0;
end
