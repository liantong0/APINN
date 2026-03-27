function tdoas = compute_tdoas_from_position(pos, hydrophone_pos, c)
% Compute TDOA values from a target position.
    distances = zeros(4, 1);
    for i = 1:4
        distances(i) = norm(pos - hydrophone_pos(i, :)');
    end

    tdoas = (distances(2:4) - distances(1)) / c;
end
