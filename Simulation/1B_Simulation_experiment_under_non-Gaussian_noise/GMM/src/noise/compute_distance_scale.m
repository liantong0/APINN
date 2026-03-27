function distance_scale = compute_distance_scale(d)
% Compute the distance-dependent noise scale used in data generation.
    if d <= 0.5
        baseline_scale = 0.001;
        normalized_dist = (d - 0.1) / (0.5 - 0.1);
        normalized_dist = max(0, min(normalized_dist, 1));
        distance_scale = baseline_scale * (1 + 4 * normalized_dist);
    elseif d <= 2.0
        baseline_scale = 0.005;
        normalized_dist = (d - 0.5) / (2.0 - 0.5);
        normalized_dist = max(0, min(normalized_dist, 1));
        distance_scale = baseline_scale + normalized_dist^1.5 * (0.1 - baseline_scale);
    elseif d <= 10
        normalized_dist = (d - 2.0) / (10 - 2.0);
        normalized_dist = max(0, min(normalized_dist, 1));
        distance_scale = 0.1 + normalized_dist^2 * 0.3;
    else
        distance_scale = d / 15;
    end
end
