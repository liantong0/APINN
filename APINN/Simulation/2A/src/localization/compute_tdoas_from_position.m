function tdoas = compute_tdoas_from_position(pos, hydrophone_pos, c)
% Compute TDOAs from a target position and hydrophone geometry.
    distances = zeros(4, 1);
    for i = 1:4
        distances(i) = norm(pos - hydrophone_pos(i, :)');
    end

    tdoas = (distances(2:4) - distances(1)) / c;
end
