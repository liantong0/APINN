function x = stbl_cms(sz, alpha, beta)
% Generate alpha-stable random variables using the CMS method.
    U = pi * (rand(sz) - 0.5);
    W = exprnd(1, sz);

    if abs(alpha - 1) > 1e-8
        phi = (1 / alpha) * atan(beta * tan(pi * alpha / 2));
        S = (1 + (beta^2) * (tan(pi * alpha / 2))^2)^(1 / (2 * alpha));

        numerator = sin(alpha * (U + phi));
        denom = (cos(U)).^(1 / alpha);
        frac = numerator ./ denom;

        term = (cos(U - alpha * (U + phi)) ./ W).^((1 - alpha) / alpha);

        x = S .* frac .* term;
    else
        b = (2 / pi) * beta;
        x = b .* ((pi / 2 + U) .* tan(U) - log((pi / 2) .* W .* cos(U) ./ (pi / 2 + U)));
    end
end
