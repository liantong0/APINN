function p_smooth = smooth_by_arclength_csaps(p)
% Smooth a 3D trajectory with a spline parameterized by arc length.
    N = size(p,1);
    if N < 6
        p_smooth = p;
        return;
    end

    ds = sqrt(sum(diff(p,1,1).^2,2));
    s  = [0; cumsum(ds)];
    s = s + (0:N-1)'*1e-9;

    sp = 0.98;

    sx = csaps(s, p(:,1), sp, s);
    sy = csaps(s, p(:,2), sp, s);
    sz = csaps(s, p(:,3), sp, s);

    p_smooth = [sx(:), sy(:), sz(:)];
end
