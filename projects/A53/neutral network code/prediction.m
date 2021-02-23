function p = prediction(X, params, input_layers, hiden_layers, num_labels)
    m = size(X, 2);
    Theta = params2Theta(params, input_layers, hiden_layers, num_labels);
    for i = 1:length(Theta)
        if i == 1
            h = sigmoid(Theta{i} * [ones(1, m); X]);
        else
            h = sigmoid(Theta{i} * [ones(1, m); h]);
        end
    end
    p = h;
    