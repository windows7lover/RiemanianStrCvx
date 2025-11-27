% -------------------------------------------------------------------------
% Localized free-energy minimization on the SPD manifold
% -------------------------------------------------------------------------
% Energy with dimension-mixing interactions:
%
%     E(X) = 1/2 * Tr(A X B X)  +  Tr(C X)
%
% where A,B,C are symmetric matrices (A,B ? 0 for stability).
%
% Gradient:
%
%     ?E(X) = 1/2 * (A X B + B X A) + C
%
% We linearize only the energy term around X0 and keep entropy S(X) exact.
%
% Solve the local trust-region subproblem:
%
%     minimize_X    < ?E(X0),  X - X0 >_X  -  T * S(X)% -------------------------------------------------------------------------
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

%     subject to    d(X, X0) ? r
%                   X ? S_{++}^n
%
% where:
%   - d(·,·) is the affine-invariant Riemannian distance,
%   - S(X)  = -Tr[ X log(X) + (I-X) log(I-X) ],
%   - r > 0 is the trust-region radius.
%
% -------------------------------------------------------------------------

% Example entropy
function val = entropy_S(X)    I = eye(size(X));
    val = -trace( X*logm(X) + (I-X)*logm(I-X) );
end

% Example energy ingredients
A = ...;   % symmetric positive definite
B = ...;   % symmetric positive definite
C = ...;   % symmetric

X0 = ...;  % current SPD matrix
T  = ...;  % temperature
r  = ...;  % trust-region radius

% Gradient of E at X0:
gradE_X0 = 0.5 * (A*X0*B + B*X0*A) + C;
