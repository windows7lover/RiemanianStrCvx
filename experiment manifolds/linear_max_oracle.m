function v = linear_max_oracle(w, x, R, x0, manifold)

w = manifold.transp(x0, x, w);
w = w/norm(w);
v = manifold.exp(x0, w, R);