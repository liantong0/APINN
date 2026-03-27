function pos = chan_algorithm(tdoas, hydrophone_pos, c)
% Estimate target position using the Chan algorithm.
    x0 = hydrophone_pos(1, :)';
    x1 = hydrophone_pos(2, :)';
    x2 = hydrophone_pos(3, :)';
    x3 = hydrophone_pos(4, :)';

    r1 = c * tdoas(1);
    r2 = c * tdoas(2);
    r3 = c * tdoas(3);

    Ga = -2 * [
        (x1 - x0)' r1;
        (x2 - x0)' r2;
        (x3 - x0)' r3
    ];

    K1 = x1' * x1 - x0' * x0;
    K2 = x2' * x2 - x0' * x0;
    K3 = x3' * x3 - x0' * x0;

    h = [
        r1^2 - K1;
        r2^2 - K2;
        r3^2 - K3
    ];

    theta1 = pinv(Ga' * Ga) * (Ga' * h);
    pos1 = theta1(1:3) + x0;

    distances = [
        norm(pos1 - x0);
        norm(pos1 - x1);
        norm(pos1 - x2);
        norm(pos1 - x3)
    ];

    B = diag([1; distances(2:4)]);
    Q = B * B';

    h2 = [
        (pos1(1) - x0(1))^2;
        (pos1(2) - x0(2))^2;
        (pos1(3) - x0(3))^2
    ];

    Ga2 = eye(3);

    theta2 = pinv(Ga2' * pinv(Q(2:4, 2:4)) * Ga2) * (Ga2' * pinv(Q(2:4, 2:4)) * h2);

    pos = sqrt(abs(theta2)) .* sign(pos1 - x0) + x0;

    if any(isnan(pos)) || any(isinf(pos))
        pos = pos1;
    end
end
