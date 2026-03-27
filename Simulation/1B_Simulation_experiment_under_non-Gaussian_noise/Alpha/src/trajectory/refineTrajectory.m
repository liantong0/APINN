function p_clean = refineTrajectory(p, k_spike)
    N = size(p,1);
    if N < 6
        p_clean = p;
        return;
    end

    dp = diff(p,1,1);
    v  = sqrt(sum(dp.^2,2));
    vm = median(v);
    madv = mad(v,1);
    thr = vm + k_spike * max(madv, 1e-9);

    spike_edge = (v > thr);
    spike_pt = false(N,1);
    spike_pt(2:end) = spike_edge | [false; spike_edge(1:end-1)];

    p_clean = p;

    t = (1:N)';
    good = ~spike_pt & all(isfinite(p),2);
    if sum(good) < 4
        return;
    end

    for d = 1:3
        p_clean(~good,d) = interp1(t(good), p(good,d), t(~good), 'pchip', 'extrap');
    end
end
