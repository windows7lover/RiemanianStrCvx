% =========================================================================
% Localized free-energy minimization on SPD manifold using RFW
%   Minimize   <?E(X0), X - X0> - T S(X)
%   s.t.       d(X, X0) ? r,  X ? 0
%
% Here:
%   E(X)  = 1/2 Tr(A X B X) + Tr(C X)
%   ?E(X) = 1/2 (A X B + B X A) + C
%   S(X)  = -Tr[ X log X + (I-X) log(I-X) ]
%
% We approximate the inner product by the Frobenius inner product so that
% the linearized energy term is simply Tr(?E(X0)^T (X - X0)), whose
% Euclidean gradient is constant: ?E(X0).
% =========================================================================

function X = solve_local_model_rfw(A, B, C, X0, T, r, max_iter)
    % A,B: SPD coupling matrices (n x n)
    % C  : symmetric matrix (n x n)
    % X0 : current SPD matrix (n x n)
    % T  : temperature
    % r  : trust-region radius in the RFW ball
    % max_iter: number of RFW iterations

    % ---- gradient of E at X0 (example energy) ---------------------------
    gradE_X0 = 0.5 * (A*X0*B + B*X0*A) + C;

    % ---- define local objective f_local(X) ------------------------------
    % f_local(X) = <gradE_X0, X - X0>_F - T S(X)
    % grad f_local(X) = gradE_X0 - T * grad S(X),
    % with grad S(X) = -log(X) + log(I - X) (matrix logs).
    f_handle = @(X) local_free_energy(X, X0, gradE_X0, T);

    % ---- run RFW on SPD ball centered at X0 ----------------------------
    X = rfw_spd(f_handle, X0, r, max_iter);
end

% -------------------------------------------------------------------------
% Local free-energy model: value and gradient
% -------------------------------------------------------------------------
function [val, G] = local_free_energy(X, X0, gradE_X0, T)
    % Linearized energy term: <gradE_X0, X - X0>_F
    lin_val = trace(gradE_X0' * (X - X0));
    lin_grad = gradE_X0;

    % Entropy term S(X)
    S_val = entropy_S(X);
    S_grad = entropy_grad(X);

    % Local model:
    %   f(X) = lin_val - T * S(X)
    % ?f(X) = lin_grad - T * S_grad
    val = lin_val - T * S_val;
    G   = lin_grad - T * S_grad;
end

% -------------------------------------------------------------------------
% Entropy S(X) = -Tr[ X log X + (I-X) log(I-X) ]
% -------------------------------------------------------------------------
function val = entropy_S(X)
    n = size(X,1);
    I = eye(n
