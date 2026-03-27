function [U_I, U_II, U_III, U_IV] = generate_quadrant_voltages(delta_m0, delta_n0, params)
% Generate quadrant detector voltages from normalized horizontal and vertical offsets.
    U_total = 1.0 + 0.1 * (2*rand - 1);
    U_total = max(U_total, 0.5);
    
    U_I   = U_total/4 * (1 + delta_m0 + delta_n0);
    U_II  = U_total/4 * (1 - delta_m0 + delta_n0);
    U_III = U_total/4 * (1 - delta_m0 - delta_n0);
    U_IV  = U_total/4 * (1 + delta_m0 - delta_n0);
    
    U_I   = max(U_I, 0);
    U_II  = max(U_II, 0);
    U_III = max(U_III, 0);
    U_IV  = max(U_IV, 0);
    
    current_sum = U_I + U_II + U_III + U_IV;
    if current_sum > 0
        U_I = U_I / current_sum * U_total;
        U_II = U_II / current_sum * U_total;
        U_III = U_III / current_sum * U_total;
        U_IV = U_IV / current_sum * U_total;
    end
end

