% =========================================================================
% DEMO: Convergence of RFW on SPD manifold for local free-energy model
% =========================================================================
clear; clc;

n = 50;                     % matrix dimension
rng(1);

% --- build synthetic SPD matrices A,B and symmetric C --------------------
A = rand(n); A = A*A';
A = A/norm(A) + eye(n)*0.01;
% --- pick starting point X0 (SPD) ---------------------------------------
X0 = randn(n); 
X0 = X0'*X0 + eye(n);

% --- temperature and trust-region parameters -----------------------------
T = 0.5;
r = 5.5;
max_iter = 100;

% --- build objective model -----------------------------------------------
f_handle = build_local_model(A);

% --- run RFW -------------------------------------------------------------
X = X0;
vals = zeros(max_iter,1);

for k = 1:max_iter
    % evaluate f(X)
    vals(k) = f_handle(X);
    % 1 RFW step
    [~, Gk] = f_handle(X);
    Delta_k = X * Gk;
    eta_max = r / norm(Delta_k,'fro');
    Delta_k = Delta_k/eta_max;
    s   = rfw_stepsize(A, X, -Delta_k, norm(A));
    X = expm(s * Delta_k) * X;
    s
end

% --- plot convergence ----------------------------------------------------
figure;
title('Convergence of RFW on SPD manifold');
subplot(1,2,1)
loglog(1:max_iter, real(abs(vals)), 'LineWidth',2);
xlabel('Iteration','fontsize',14,'interpreter','latex');
ylabel('$f(X_k) - f^\ast$','fontsize',14,'interpreter','latex');
grid on;

subplot(1,2,2)
semilogy(1:max_iter, real(abs(vals)), 'LineWidth',2);
xlabel('Iteration','fontsize',14,'interpreter','latex');
ylabel('$f(X_k) - f^\ast$','fontsize',14,'interpreter','latex');
grid on;

