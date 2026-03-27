function S = local_roughness_issd(y, x)
% Compute local roughness score using the ISSD metric over a sampled curve.
    y = y(:); 
    x = x(:);

    ok = isfinite(y) & isfinite(x);
    y = y(ok); 
    x = x(ok);

    if numel(y) < 5
        S = NaN;
        return;
    end

    dx = mean(diff(x));
    y2 = diff(y, 2) / (dx^2);
    S = sum(y2.^2) * dx;
end

