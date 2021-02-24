function params = trainnetwork(X, y, input_layers, hiden_layers, num_labels, lambda)
    init_params = [];
    for i = 1:length(hiden_layers) + 1
        if i == 1
            theta = randominit(input_layers, hiden_layers(1));
        elseif i == length(hiden_layers) + 1
            theta = randominit(hiden_layers(end), num_labels);
        else
            theta = randominit(hiden_layers(i-1), hiden_layers(i));
        end
        init_params = [init_params; theta(:)];
    end
    costf = @(p) costfunction(X, y, p, input_layers, hiden_layers, num_labels, lambda);
    options = optimset('MaxIter', 2000, 'Display','iter','PlotFcns',@optimplotfval);
    params = fmincg(costf, init_params, options);
end
