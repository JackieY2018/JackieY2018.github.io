function Theta = params2Theta(params, input_layers, hiden_layers, num_labels)
     Theta = cell(length(hiden_layers) + 1, 1);
    Theta{1} = reshape(params(1:hiden_layers(1) * (input_layers + 1)), hiden_layers(1), input_layers + 1);
    for i = 1:length(hiden_layers)
        num_used = 0;
        for j = 1:i
            num_used = num_used + numel(Theta{j});
        end
        if i == length(hiden_layers)
            Theta{i+1} = reshape(params(num_used + 1:end), num_labels, hiden_layers(i) + 1);
        else
            Theta{i+1} = reshape(params(num_used + 1:num_used + hiden_layers(i+1) * (hiden_layers(i) + 1)), hiden_layers(i+1), hiden_layers(i) + 1);
        end
    end
end
