clear; clc; close all;

% ================================================================
% PARAMETERS
% ================================================================
n        = 50;
r        = 1;      % radius of geodesic ball around X0
max_iter = 30;
L        = 10.0;      % smoothness constant for FW stepsize (take >= 1)

% ================================================================
% 1. CENTER X0
% ================================================================
A0 = randn(n);
X0 = A0'*A0 + eye(n);                 % SPD
X0 = X0 / norm(X0,'fro') * n;         % mild normalization

% ================================================================
% 2. TARGET Y (must lie OUTSIDE the ball)
% ================================================================
Y = sample_point_at_radius(X0, r, 1.001);

% check distance
fprintf("d(X0,Y) = %.6f (should be > %.6f)\n", spd_distance(X0,Y), r);

% ================================================================
% 3. INITIAL POINT INSIDE THE BALL
% ================================================================
Z = randn(n);  Z = (Z+Z')/2;
Z = 0.1*r * Z / norm(Z,'fro');          % < r
X = spd_exp(X0, Z);

fprintf("d(X0,X_init) = %.6f (should be < %.6f)\n", ...
        spd_distance(X0,X), r);

% ================================================================
% OBJECTIVE:  f(X) = 1/2 d(X,Y)^2
% ================================================================
f_handle = @(X) objective_outside(X,Y);

% ================================================================
% EXACT CONSTRAINED OPTIMUM
% ================================================================
X_star = spd_exact_solution(X0, Y, r);
[f_star,~] = objective_outside(X_star, Y);

fprintf("Exact constrained optimum computed.\n");
fprintf("d(X0,X_star) = %.6f (should be %.6f)\n", ...
    spd_distance(X0,X_star), r);

% ================================================================
% RFW LOOP
% ================================================================
history = zeros(max_iter,1);


fprintf("d(X0,X_init) = %.6f (target r = %.6f)\n", spd_distance(X0,X), r);

for k = 1:max_iter

    % evaluate f and gradient
    [fk, Gk] = f_handle(X);
    history(k) = fk - f_star;     % optimality gap

    % ---- LMO on the SPD ball ----
    V_star = spd_lmo(X0, X, Gk, r);
    
    [ftemp,~]=f_handle(V_star);
    [fk-ftemp]
    
    d_boundary = spd_distance(X0,V_star);
    fprintf("Iter %3d: d(X0,V_star) = %.6f (target r = %.6f)\n", k, d_boundary, r);

    % FW direction
    Delta = spd_log(X, V_star);

    % ---- quadratic stepsize: minimize a*s + b*s^2 ----
    a  = spd_inner(X, Gk, Delta);
    dn = spd_norm(X, Delta);
    b  = 0.5 * L * dn^2;

    if b < 1e-14
        eta = 1;
    else
        eta = min(1, max(0, -a/(2*b)));
    end
    
    if(k<10)
        eta = 1/(k+2);
    end

    % update
    X = spd_exp(X, eta * Delta);
end

% ================================================================
% PLOTTING
% ================================================================
% Data
k = 1:max_iter;
gap = history(:)';
k_switch = 10;

% Precompute y-min and y-max for safe axis control
gap_min = min(gap);
gap_max = max(gap);


figure('Position',[100 100 1100 450]);

% ================================================================
% LEFT SUBPLOT (log-log)
% ================================================================
subplot(1,2,1); hold on;

% --- blue region (1/t) ---
idx1 = 1:k_switch;
h1 = area(k(idx1), gap(idx1));
set(h1,'FaceColor',[0.6 0.8 1],'EdgeColor','none','FaceAlpha',0.7);

% --- green region (linear) ---
idx2 = k_switch:max_iter;
h2 = area(k(idx2), gap(idx2));
set(h2,'FaceColor',[0.7 0.9 0.7],'EdgeColor','none','FaceAlpha',0.7);

% --- RFW curve ---
h3 = plot(k, gap, 'k', 'LineWidth',3);

set(gca,'XScale','log','YScale','log');
xlabel('{\bf \#Gradient Oracle Calls}','Interpreter','latex','FontSize',12);
ylabel('{\bf RFW Duality Gap}','Interpreter','latex','FontSize',12);

xlim([1 max_iter])
ylim([gap_min gap_max])
grid on;


% ================================================================
% RIGHT SUBPLOT (linear-x, log-y)
% ================================================================
subplot(1,2,2); hold on;

% --- blue region (1/t) ---
h1b = area(k(idx1), gap(idx1));
set(h1b,'FaceColor',[0.6 0.8 1],'EdgeColor','none','FaceAlpha',0.7);

% --- green region (linear) ---
h2b = area(k(idx2), gap(idx2));
set(h2b,'FaceColor',[0.7 0.9 0.7],'EdgeColor','none','FaceAlpha',0.7);

% --- RFW curve ---
h3b = plot(k, gap, 'k', 'LineWidth',3);

set(gca,'YScale','log');
xlabel('{\bf \#Gradient Oracle Calls}','Interpreter','latex','FontSize',12);
% ylabel omitted for alignment

xlim([1 max_iter])
ylim([gap_min gap_max])
grid on;

% --- Legend (only on right subplot) ---
legend([h1b h2b h3b], ...
    {'$\frac{1}{t}$ convergence', 'Linear Convergence', 'RFW Dual Gap'}, ...
    'Interpreter','latex', 'FontSize',12, ...
    'Location','northeast');


% ======================================================================
%  OBJECTIVE: f(X) = 1/2 d(X,Y)^2
% ======================================================================
function [fval,G] = objective_outside(X,Y)
    Z = spd_log(Y,X);
    fval = 0.5 * norm(Z,'fro')^2;
    G = -spd_log(X,Y);      % Riemannian gradient of squared distance
end

% ======================================================================
%  EXACT CONSTRAINED SOLUTION (projection of Y onto ball)
% ======================================================================
function X_star = spd_exact_solution(X0, Y, r)
    Z  = spd_log(X0,Y);
    d0 = spd_norm(X0,Z);

    if d0 <= r
        X_star = Y;
        return;
    end

    X_star = spd_exp(X0, (r/d0)*Z);
end

% ======================================================================
%  DISTANCE d(X0,X)
% ======================================================================
function d = spd_distance(X0,X)
    X0h  = sqrtm(X0); 
    X0ih = inv(X0h);
    M = X0ih * X * X0ih;
    M = (M+M')/2;
    Z = logm(M);
    d = norm(Z,'fro');
end


% ======================================================================
%  LMO (uses robust solve_alpha)
% ======================================================================
function V_star = spd_lmo(X0, Xk, Gk, r)

    % ---- construct 2D subspace ----
    u1 = spd_log(Xk,X0);
    n1 = spd_norm(Xk,u1);
    u1 = u1 / n1;

    g_perp = Gk - spd_inner(Xk,Gk,u1)*u1;
    u2 = g_perp / spd_norm(Xk,g_perp);

    % transforms for inner product
    Xk_half     = sqrtm(Xk); 
    Xk_inv_half = inv(Xk_half);

    % reduced objective over φ
    obj = @(phi) reduced_obj(phi,X0,Xk,Xk_inv_half,u1,u2,Gk,r);
    
    opts_phi = optimset( ...
        'TolX', 1e-12, ...
        'TolFun', 1e-12, ...
        'MaxIter', 400, ...
        'MaxFunEvals', 2000, ...
        'Display', 'off' ...
    );
    phi_star = fminbnd(obj,-pi,pi,opts_phi);

    % reconstruct boundary point
    p_star     = cos(phi_star)*u1 + sin(phi_star)*u2;
    alpha_star = solve_alpha(phi_star,X0,Xk,u1,u2,r);

    V_star = spd_exp(Xk, alpha_star*p_star);
end

% reduced objective F(φ)
function Fphi = reduced_obj(phi,X0,Xk,Xk_inv_half,u1,u2,Gk,r)
    p     = cos(phi)*u1 + sin(phi)*u2;
    alpha = solve_alpha(phi,X0,Xk,u1,u2,r);
    Gp = trace( (Xk_inv_half*Gk*Xk_inv_half) * (Xk_inv_half*p*Xk_inv_half) );
    Fphi = alpha * Gp;
end

% robust alpha solver via bisection (true exp + true distance)
function alpha = solve_alpha(phi, X0, Xk, u1, u2, r)

    % search direction
    p = cos(phi)*u1 + sin(phi)*u2;

    % objective g(a) = d(X0, Exp_{Xk}(a p)) - r
    g = @(a) spd_distance(X0, spd_exp(Xk, a*p)) - r;

    % --- automatic bracketing for fzero ----------------------------
    % start from [0, aR], expand until g(aR) > 0
    aL = 0;
    aR = 1;
    while g(aR) <= 0
        aR = 2*aR;
        if aR > 1e6
            error('Could not bracket alpha — direction does not reach boundary.');
        end
    end

    % --- root find --------------------------------------------------
    alpha = fzero(g, [aL, aR]);
end

function Y = sample_point_at_radius(X0, r, c)
% Construct Y at distance c*r from X0 on the SPD manifold

    n = size(X0,1);

    % Random symmetric direction
    B = randn(n); 
    U = 0.5*(B + B');

    % Normalize in tangent norm at X0
    U = U / spd_norm(X0, U);

    % Exponential map: Y = Exp_{X0}((c*r) * U)
    Y = spd_exp(X0, (c*r) * U);
end




% ======================================================================
%  SPD GEOMETRY
% ======================================================================
function Y = spd_exp(X,U)
    Xh = sqrtm(X); Xin = inv(Xh);
    Y = Xh * expm(Xin*U*Xin) * Xh;
    Y = 0.5*(Y+Y');           % symmetrize
end

function U = spd_log(X,Y)
    Xh = sqrtm(X); Xin = inv(Xh);
    M = Xin * Y * Xin;
    M = (M+M')/2;
    U = Xh * logm(M) * Xh;
    U = 0.5*(U+U');           % symmetrize
end

function v = spd_inner(X,U,V)
    Xi = inv(X);
    v = trace((Xi*U)*(Xi*V));
end

function n = spd_norm(X,U)
    n = sqrt(spd_inner(X,U,U));
end
