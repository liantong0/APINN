function pos = spherical_to_cart(Y)
% Convert spherical state variables [d, psi, theta] to Cartesian coordinates.
    d = Y(:,1); psi = Y(:,2); theta = Y(:,3);
    x = d .* cos(theta) .* cos(psi);
    y = d .* cos(theta) .* sin(psi);
    z = d .* sin(theta);
    pos = [x,y,z];
end
