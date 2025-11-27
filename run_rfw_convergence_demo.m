% RUN_RFW_CONVERGENCE_DEMO  Demonstrate convergence of the RFW solver
% on a localized free-energy model on the SPD manifold.
%
% The demo builds small SPD matrices A and B, a symmetric coupling C, and
% an initial point X0. It then calls solve_local_model_rfw to minimize the
% localized model and plots the objective decrease.

rng(0);            % reproducibility
n = 3;             % matrix dimension
max_iter = 50;     % RFW iterations
T = 0.5;           % temperature
r = 0.1;           % trust-region radius (keeps iterates near X0)

% Build SPD matrices A and B
M = randn(n);
A = M' * M + n * eye(n);
N = randn(n);
B = N' * N + n * eye(n);

% Symmetric linear term
C = randn(n);
C = 0.5 * (C + C');

% Start from a well-conditioned SPD matrix inside (0, I)
X0 = 0.4 * eye(n);

[X_final, obj_history] = solve_local_model_rfw(A, B, C, X0, T, r, max_iter);

% Plot convergence rate using the best value seen as a surrogate optimum
best_val = min(obj_history);
suboptimality = obj_history - best_val;

figure;
semilogy(0:max_iter, suboptimality, '-o', 'LineWidth', 1.2);
xlabel('Iteration');
ylabel('Objective gap');
title('RFW convergence on localized free-energy model');
grid on;

fprintf('Final objective: %.6f\n', obj_history(end));
fprintf('Best objective : %.6f\n', best_val);
