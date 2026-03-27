function delta_0 = compute_normalized_offset(U_I, U_II, U_III, U_IV, direction)
% Compute the normalized horizontal or vertical offset from quadrant voltages.

    U_total = U_I + U_II + U_III + U_IV;
    if U_total < 1e-6
        delta_0 = 0;
        return;
    end

    if strcmp(direction, 'horizontal')
        delta_0 = (U_I + U_IV - U_II - U_III) / U_total;
    elseif strcmp(direction, 'vertical')
        delta_0 = (U_I + U_II - U_III - U_IV) / U_total;
    else
        error('Invalid direction');
    end
end
