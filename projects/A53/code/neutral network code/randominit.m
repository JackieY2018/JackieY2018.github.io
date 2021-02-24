function theta = randominit(L_in, L_out)
    theta = zeros(L_out, L_in + 1);
    epsilon = sqrt(6) / sqrt(L_in + L_out);
    theta = rand(L_out, L_in + 1) * 2 * epsilon - epsilon;
end
