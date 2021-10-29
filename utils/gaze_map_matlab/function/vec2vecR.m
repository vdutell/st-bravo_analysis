function R = vec2vecR(a,b)

    u = a/norm(a);                      % a and b must be column vectors
    v = b/norm(b);                      % of equal length
    N = length(u);
    S = reflection( eye(N), v+u );      % S*u = -v, S*v = -u 
    R = reflection( S, v );             % v = R*u

end

function v = reflection( u, n )     % Reflection of u on hyperplane n.
    %
    % u can be a matrix. u and v must have the same number of rows.

    v = u - 2 * n * (n'*u) / (n'*n);
end