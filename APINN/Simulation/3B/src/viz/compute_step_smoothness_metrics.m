function metrics = compute_step_smoothness_metrics(d, P, boundaries)
% Compute smoothness metrics around turbidity-step boundaries.
    nB = numel(boundaries);
    J0 = zeros(1, nB);
    J1 = zeros(1, nB);

    fit_half_window = 4;   %  4 ?3~6

    for k = 1:nB
        db = boundaries(k);


        [~, idx_b] = min(abs(d - db));


        idx_left  = max(1, idx_b-fit_half_window) : max(1, idx_b-1);
        idx_right = min(length(d), idx_b+1) : min(length(d), idx_b+fit_half_window);

        dL = d(idx_left);
        dR = d(idx_right);
        PL = P(idx_left, :);
        PR = P(idx_right, :);


        if numel(dL) < 2 || numel(dR) < 2
            idx_minus = max(idx_b - 1, 1);
            idx_plus  = min(idx_b + 1, length(d));

            p_minus = P(idx_minus, :);
            p_b     = P(idx_b, :);
            p_plus  = P(idx_plus, :);

            d_minus = d(idx_minus);
            d_b_use = d(idx_b);
            d_plus  = d(idx_plus);

            J0(k) = norm(p_plus - p_minus);

            v_minus = (p_b - p_minus) / max(d_b_use - d_minus, 1e-12);
            v_plus  = (p_plus - p_b) / max(d_plus - d_b_use, 1e-12);
            J1(k) = norm(v_plus - v_minus);
            continue;
        end

        pL_db = zeros(1,3);
        pR_db = zeros(1,3);
        vL    = zeros(1,3);
        vR    = zeros(1,3);

        for dim = 1:3
            cL = polyfit(dL, PL(:,dim), 1);   % cL(1)=, cL(2)=?
            cR = polyfit(dR, PR(:,dim), 1);

            pL_db(dim) = polyval(cL, db);
            pR_db(dim) = polyval(cR, db);

            vL(dim) = cL(1);
            vR(dim) = cR(1);
        end

        J0(k) = norm(pR_db - pL_db);

        J1(k) = norm(vR - vL);
    end

    metrics.J0 = J0;
    metrics.J1 = J1;
end

