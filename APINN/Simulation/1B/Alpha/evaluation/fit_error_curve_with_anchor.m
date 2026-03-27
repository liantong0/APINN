function yq = fit_error_curve_with_anchor(d, e, dq, d_anchor, d_taper)
% Fit a smooth error curve and force it to taper to zero near the anchor point.
% Smooth error curve fitting with an endpoint anchor
    if nargin < 4 || isempty(d_anchor)
        d_anchor = min(d);
    end
    if nargin < 5 || isempty(d_taper)
        d_taper = 1.0;
    end

    ok = isfinite(d) & isfinite(e) & d>=0 & e>=0;
    d = d(ok); e = e(ok);
    [d, idx] = sort(d);
    e = e(idx);

    if numel(d) < 20
        yq = nan(size(dq));
        return;
    end

    [d, iu] = unique(d, 'stable');
    e = e(iu);

    % 1) Smooth fitting
    if exist('csaps','file') == 2
        sp = 0.90;
        y = csaps(d, e, sp, dq);
    elseif exist('fit','file') == 2
        f = fit(d, e, 'smoothingspline', 'SmoothingParam', 0.85);
        y = f(dq);
    else
        win = max(15, round(0.06 * numel(e)));
        e_sm = smoothdata(e, 'movmean', win);
        y = interp1(d, e_sm, dq, 'pchip', 'extrap');
    end

    y = max(y, 0);

    % 2) Endpoint anchor
    s = (dq - d_anchor) / max(d_taper, 1e-9);
    s = min(max(s, 0), 1);
    w = s.^2 .* (3 - 2*s);

    yq = y .* w;
    yq(dq <= d_anchor) = 0;
end
