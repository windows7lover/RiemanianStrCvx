function f_handle = build_local_model(A)
% -------------------------------------------------------------------------
% Simple quadratic model:
%       f(X) = ||A X||_F^2
%  grad f(X) = 2 A' (A X)
%
% Returned:
%     f_handle(X) = [value, gradient]
% -------------------------------------------------------------------------

    f_handle = @(X) quad_obj(A, X);
end


% -------------------------------------------------------------------------
% Quadratic objective and gradient
% -------------------------------------------------------------------------
function [val, G] = quad_obj(A, X)

    AX = A * X;
    val = sum(AX(:).^2);      % ||AX||_F^2

    G = 2 * (A' * AX);        % gradient
end
