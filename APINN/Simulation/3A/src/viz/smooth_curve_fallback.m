function [dq, yq] = smooth_curve_fallback(d, e, dq)
% Smooth and interpolate curves when toolbox fitting is unavailable.

    dmin = min(dq); 
    dmax = max(dq);

    ok = isfinite(d) & isfinite(e) & d>=dmin & d<=dmax & e>=0;
    d = d(ok); 
    e = e(ok);

    [d, idx] = sort(d);
    e = e(idx);

    if numel(d) < 20
        yq = nan(size(dq));
        return;
    end

    win = max(15, round(0.06 * numel(d)));
    e_sm = smoothdata(e, 'movmean', win);


    yq = interp1(d, e_sm, dq, 'pchip', NaN);


    first_valid = find(isfinite(yq), 1, 'first');
    last_valid  = find(isfinite(yq), 1, 'last');

    if ~isempty(first_valid)
        yq(1:first_valid-1) = yq(first_valid);
    end
    if ~isempty(last_valid)
        yq(last_valid+1:end) = yq(last_valid);
    end

    yq = max(yq, 0);
end



