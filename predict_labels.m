function pred_labels=predict_labels(train_inputs,train_labels,test_inputs)

part = repmat(1:7,1,ceil(size(train_inputs,1)/7));
part = part(1:size(train_inputs,1));
part = part(randperm(size(train_inputs,1)));
pred_labels = zeros(size(test_inputs,1),size(train_labels,2));
C_vals = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1];
for f = 1:9
    min_idx = 0; min_cv_err = Inf;
    for i = 1:6
        C = C_vals(i);
        tot_cv_err = 0;
        for j = 1:7
            curlbls = train_labels(:,f);
            X_train = train_inputs(part~=i,:);
            y_train = curlbls(part~=i,:);
            X_test = train_inputs(part==i,:);
            y_test = curlbls(part==i,:);
            % setting up the net and training
            net = fitnet(10);
            net.layers{1}.transferFcn = 'poslin';
            net.performFcn = 'mse';
            net.performParam.regularization = C;
            net.trainFcn = 'trainlm';
            net = configure(net, X_train', y_train');
            net = train(net, X_train', y_train');
            % predicting
            y_pred = net(X_test')';
            cl_err = sum(sqrt(sum((y_test-y_pred).^2)));
            tot_cv_err = tot_cv_err + cl_err;
        end
        if tot_cv_err < min_cv_err
            min_cv_err = tot_cv_err;
            min_idx = i;
        end
    end
    C_min = C_vals(min_idx)
    X_train = train_inputs;
    y_train = train_labels(:,f);
    X_test = test_inputs;
    % setting up and training final net w optimal parameters
    net = fitnet(10);
    net.layers{1}.transferFcn = 'poslin';
    net.performFcn = 'mse';
    net.performParam.regularization = C_min;
    net.trainFcn = 'trainlm';
    net = configure(net, X_train', y_train');
    net = train(net, X_train', y_train');
    % predicting
    pred_labels(:,f) = net(X_test')';
end
% y_p_tr = sign(net(X_train')');
% fprintf('C used by final model: %f\n', C_min);
% fprintf('training error achieved by final model: %f\n', ...
%     classification_error(y_p_tr, y_train));
% fprintf('min cross validation error across all parameters: %f\n', ...
%     min_cv_err/5);
end