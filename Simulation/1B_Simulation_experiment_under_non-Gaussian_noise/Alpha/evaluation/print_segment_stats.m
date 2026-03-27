function print_segment_stats(err_chan, err_data, err_fixed, err_apinn, seg_id)
% Print mean 3D localization errors for each trajectory segment.
    names = {'Far(45-10m)','Arc(10-3m)','Near(<3m)'};

    fprintf('\nTrajectory segment error statistics (3D error, m)\n');
    fprintf('%-12s | %-10s %-10s %-10s %-10s\n', 'Segment', 'Analytical', 'DataNN', 'Fixed', 'APINN');

    for s = 1:3
        idx = (seg_id==s);
        mc = mean(err_chan(idx)); md = mean(err_data(idx));
        mf = mean(err_fixed(idx)); mp = mean(err_apinn(idx));
        fprintf('%-12s | %-10.4f %-10.4f %-10.4f %-10.4f\n', names{s}, mc, md, mf, mp);
    end
    fprintf('\n');
end

