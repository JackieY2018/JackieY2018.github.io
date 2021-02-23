function [J, grad] = costfunction(X, y, params, input_layers, hiden_layers, num_labels, lambda)
    J = 0;
    grad = [];
    m = size(X, 2);
    Theta = params2Theta(params, input_layers, hiden_layers, num_labels);
    a = cell(length(hiden_layers) + 2, 1);
    z = cell(length(hiden_layers) + 1, 1);
    a{1} = X;
    for i = 1:length(hiden_layers) + 1
        z{i} = Theta{i} * [ones(1, m); a{i}];
        a{i+1} = sigmoid(z{i});
    end
    h = (a{end}-y).^2;
    for i = 1:length(hiden_layers) + 1
        J = J + lambda * sum(sum(Theta{i}.^2));
    end
    J = (J / 2 + sum(sum(h))) / m;
    delta = cell(length(hiden_layers) + 1, 1);
    delta{end} = 2 .* (a{end}-y) .* sigmoidgrid(z{end});
    for i = length(hiden_layers):-1:1
        delta{i} = Theta{i+1}(:,2:end)' * delta{i+1} .* sigmoidgrid(z{i});
    end
    for i = 1:length(delta)
        delta{i} = delta{i} * [ones(1, m); a{i}]';
        delta{i} = (delta{i} + lambda * [zeros(size(Theta{i}, 1), 1) Theta{i}(:, 2:end)]) / m;
        grad = [grad; delta{i}(:)];
    end
end

    
    