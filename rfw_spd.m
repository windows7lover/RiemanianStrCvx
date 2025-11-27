% -------------------------------------------------------------------------
% Simplified RFW on a ball in the SPD Manifold
% -------------------------------------------------------------------------
% INPUT:
%   f        : function handle returning f(X) and grad f(X)
%   Xbar     : center of the trust region (SPD matrix)
%   r        : radius
%   max_iter : maximum iterations
%
% OUTPUT:
%   X        : final iterate
% -------------------------------------------------------------------------

function X = rfw_spd(f, Xbar, r, max_iter)

    X = Xbar;

    for k = 1:max_iter

        % ---- (1) Gradient -----------------------------------------------
        [~, Gk] = f(X);

        % ---- (2) LMO direction ------------------------------------------
        Delta_k = X * Gk;
        eta_max = r / norm(Delta_k, 'fro');

        % pick stepsize (your rule; here exact-maximum)
        eta_k = eta_max;

        % ---- (3) Geodesic update ----------------------------------------
        % X_{k+1} = exp(eta_k * Delta_k) * X_k
        X = expm(eta_k * Delta_k) * X;

    end
end
