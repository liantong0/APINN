function P = spherical_to_cartesian_batch(Y)
% Convert batched spherical coordinates to Cartesian coordinates.
    d = Y(:,1);
    psi = Y(:,2);
    theta = Y(:,3);

    P = [d .* cos(psi) .* cos(theta), ...
         d .* sin(psi) .* cos(theta), ...
         d .* sin(theta)];
end

