function x_plus = line_search(x, y, f, k, manifold)

ye = manifold.log(x,y);
fun = @(alpha) f( manifold.exp(x, ye, alpha) );

alpha_opt = 0.001;
f_opt = fun(alpha_opt);
for alpha = (1/(k+1)):0.01:1
    fnew = fun(alpha);
    if fnew <= f_opt
        f_opt = fnew;
        alpha_opt = alpha;
    end
end

step_size = alpha_opt;
alpha_opt
% step_size = 0.001;

x_plus = manifold.exp(x, ye, step_size);
