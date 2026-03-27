function [dq, yq] = smooth_curve_fallback(d, e, dq)
% Smooth and interpolate curves when toolbox fitting is unavailable.
    dmin = min(dq); dmax = max(dq);

    ok = isfinite(d) & isfinite(e) & d>=dmin & d<=dmax & e>=0;
    d = d(ok); e = e(ok);

    [d, idx] = sort(d);
    e = e(idx);

    if numel(d) < 20
        yq = nan(size(dq));
        return;
    end

    win = max(15, round(0.06 * numel(d)));   % 6%?
    e_sm = smoothdata(e, 'movmean', win);

    yq = interp1(d, e_sm, dq, 'pchip', 'extrap');
    yq = max(yq, 0);
end

