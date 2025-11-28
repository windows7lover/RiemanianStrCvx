function s = rfw_stepsize(A, X, V, L)
% Computes the stepsize:
%   s = argmin_{s?[0,1]}  ? s + ? s^2
% where
%   ? = <?f(X),  log_X(V)>_X
%   ? = (L/2) d^2(X, V)

    % --- gradient of f(X) = ||A X||^2 -----------------------------------
    % A is SPD ? A' = A
    G = 2 * (A * A * X);    

    % --- compute Exp^{-1}_X(V) = log_X(V) -------------------------------
    Xhalf  = sqrtm(X);
    Xihalf = inv(Xhalf);
    M = logm( Xihalf * V * Xihalf );      % M = log(X^{-1/2} V X^{-1/2})
    log_X_V = Xhalf * M * Xhalf;

    % --- Riemannian inner product <G, log_X(V)>_X ------------------------
    % <U,W>_X = trace(X^{-1} U X^{-1} W)
    Xinv = inv(X);
    alpha = trace( Xinv * G * Xinv * log_X_V );

    % --- Riemannian distance d(X,V) --------------------------------------
    d2 = norm(M, 'fro')^2;       % since d(X,V) = ||M||_F

    beta = 0.5 * L * d2;

    % --- unconstrained minimizer s* = -alpha / (2beta) -------------------
    if beta > 0
        s_uncon = -alpha / (2 * beta);
    else
        s_uncon = 1;   % degenerate case
    end

    % --- project onto [0,1] ----------------------------------------------
    s = min(1, max(0, s_uncon));
end
