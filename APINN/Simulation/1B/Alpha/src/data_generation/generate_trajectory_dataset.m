function [traj, X_raw, Y_true, seg_id] = generate_trajectory_dataset(traj_params, params)
% Generate a multi-segment trajectory dataset and corresponding raw measurements.
    % traj_params includes distance ranges, initial/final angles, and point counts
    rng(traj_params.seed);

    % 1) Generate three segments: d, psi, theta
    smoothstep = @(s) (3*s.^2 - 2*s.^3);

    % Far segment
    s1 = linspace(0,1,traj_params.N_far)';
    d1 = traj_params.d_far(1) + (traj_params.d_far(2)-traj_params.d_far(1))*s1;
    psi1 = traj_params.psi0 * ones(size(d1));
    theta1 = traj_params.theta0 * ones(size(d1));

    % Arc segment
    s2 = linspace(0,1,traj_params.N_arc)';
    s2s = smoothstep(s2);
    d2 = traj_params.d_arc(1) + (traj_params.d_arc(2)-traj_params.d_arc(1))*s2;

    psi_mid = traj_params.psi0 + (traj_params.psi_end - traj_params.psi0)*s2s ...
              + deg2rad(10)*sin(pi*s2);

    theta_mid = traj_params.theta0 + (deg2rad(-30) - traj_params.theta0)*s2s;

    psi2 = psi_mid;
    theta2 = theta_mid;

    % Near segment: align to target angles
    s3 = linspace(0,1,traj_params.N_near)';
    s3s = smoothstep(s3);
    d3 = traj_params.d_near(1) + (traj_params.d_near(2)-traj_params.d_near(1))*s3;

    psi3 = psi2(end) + (traj_params.psi_end - psi2(end))*s3s;
    theta3 = theta2(end) + (traj_params.theta_end - theta2(end))*s3s;

    % Concatenate
    d = [d1; d2; d3];
    psi = [psi1; psi2; psi3];
    theta = [theta1; theta2; theta3];

    if theta(1) >= 0
        theta = theta - deg2rad(5);
    end

    Y_true = [d, psi, theta];

    % seg_id: 1=far, 2=arc, 3=near
    seg_id = [ones(size(d1)); 2*ones(size(d2)); 3*ones(size(d3))];

    traj = struct();
    traj.d = d; traj.psi = psi; traj.theta = theta;
    traj.seg_id = seg_id;

    % 2) Generate X_raw from the true states
    X_raw = generate_inputs_from_states(Y_true, traj_params, params);
end
