x = load('PV1.txt');
xx = (x-min(min(x)))/(max(max(x))-min(min(x)));
xxx = 2 * xx - 1;
X = zeros(10, 13*3);
Y = zeros(1, 10);
[m, n] = size(X);
input_layers = n;
num_labels = 1;
lambda = 0.000001;
params_lst = cell(1,13);
    hiden_layers = [10,10,10];
     for k = 1:13
         for i = 1:m
             Y(i) = xxx(k, i+3);
             X(i,1:end) = [xxx(1:end,i)' xxx(1:end,i+1)' xxx(1:end,i+2)'];
         end
        params = trainnetwork(X', Y, input_layers, hiden_layers, num_labels, lambda);
        params_lst{k} = params;
     end

s = zeros(13,11);
for k = 1:13
    for i = 1:11
        Z = [xxx(1:end,i)' xxx(1:end,i+1)' xxx(1:end,i+2)'];
        p = prediction(Z', params_lst{k}, input_layers, hiden_layers, num_labels);
        s(k,i) = (p+1)*(max(max(x))-min(min(x)))/2+ min(min(x));
    end
end