
function d = spd_distance(X0, X)
% Affine-invariant distance between SPD matrices X0 and X.
%
%   d(X0,X) = || log( X0^{-1/2} X X0^{-1/2} ) ||_F

    % symmetric square root and inverse
    X0_half     = sqrtm(X0);
    X0_inv_half = inv(X0_half);

    % similarity transform to tangent at X0
    M = X0_inv_half * X * X0_inv_half;
    M = (M + M')/2;       % enforce symmetry

    % ensure SPD numerically
    ev = eig(M);
    if min(ev) <= 0
        % shift inside SPD cone
        M = M + (1e-12 - min(ev)) * eye(size(M));
    end

    % matrix log
    Z = logm(M);

    % Frobenius norm
    d = norm(Z, 'fro');
end

