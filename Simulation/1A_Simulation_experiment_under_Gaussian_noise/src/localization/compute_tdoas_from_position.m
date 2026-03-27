function tdoas = compute_tdoas_from_position(pos, hydrophone_pos, c)
% Compute TDOA values from the target position and hydrophone array geometry.

    % Compute distances from the target to each hydrophone
    distances = zeros(4, 1);
    for i = 1:4
        distances(i) = norm(pos - hydrophone_pos(i, :)');
    end
    
    % Compute time differences relative to the reference hydrophone
    tdoas = (distances(2:4) - distances(1)) / c;
end
