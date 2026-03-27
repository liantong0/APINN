function pos_sm = smooth_trajectory_3d(pos, seg_id, win, k_spike)
% Smooth a 3D trajectory segment by segment after removing spike points.
    if nargin < 3 || isempty(win), win = 21; end
    if nargin < 4 || isempty(k_spike), k_spike = 4.0; end
    if mod(win,2) == 0, win = win + 1; end

    pos_sm = zeros(size(pos));

    segs = unique(seg_id(:))';
    for s = segs
        idx = find(seg_id == s);
        p = pos(idx,:);

        % 1) Remove spikes based on velocity jumps
        p_clean = refineTrajectory(p, k_spike);

        % 2) Smooth fit
        if exist('csaps','file') == 2
            p_smooth = smooth_by_arclength_csaps(p_clean);
        else
            p_smooth = smoothdata(p_clean, 1, 'sgolay', win);
            p_smooth = smoothdata(p_smooth, 1, 'movmean', max(5, round(win/2)));
        end

        pos_sm(idx,:) = p_smooth;
    end
end
