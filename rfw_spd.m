% -------------------------------------------------------------------------
% Localized free-energy minimization on the SPD manifold
% Simplified Riemannian Frank-Wolfe (RFW) on a ball in the SPD manifold
% -------------------------------------------------------------------------
% INPUT:
%   f        : function handle returning [f(X), grad f(X)]
%   Xbar     : center of the trust region (SPD matrix)
%   r        : trust-region radius
%   max_iter : maximum iterations
%
% OUTPUT:
%   X            : final iterate
%   obj_history  : objective values at each iteration (length max_iter+1)
% -------------------------------------------------------------------------

function [X, obj_history] = rfw_spd(f, Xbar, r, max_iter)
    % initialize
    X = Xbar;
    obj_history = zeros(max_iter + 1, 1);
    [obj_history(1), ~] = f(X);

    for k = 1:max_iter
        % ---- (1) Gradient -------------------------------------------------
        [~, Gk] = f(X);

        % ---- (2) LMO direction ------------------------------------------
        Delta_k = X * Gk;
        eta_max = r / norm(Delta_k, 'fro');

        % pick stepsize (exact-maximum on the geodesic ball)
        eta_k = eta_max;

        % ---- (3) Geodesic update ----------------------------------------
        % X_{k+1} = exp(eta_k * Delta_k) * X_k
        X = expm(eta_k * Delta_k) * X;

        [obj_history(k + 1), ~] = f(X);
    end
end
