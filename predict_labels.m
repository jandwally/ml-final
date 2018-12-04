function pred_labels=predict_labels(train_inputs,train_labels,test_inputs)

    %{
    Health outcomes are:
        1: 'health_aamort'    -- years potential life lost, per 100000
        2: 'health_fairpoor'  -- % of adults that report fair or poor health
        3: 'health_mentunh'   -- average # of mentally unhealthy days
        4: 'health_pcdiab'    -- % of adults reported diabetic
        5: 'health_pcexcdrin' -- % of adults who drink excessively
        6: 'health_pcinact'   -- % of adults who report no leisure physical activity
        7: 'health_pcsmoker'  -- % of adults who are smokers
        8: 'health_physunh'   -- average # of physically unhealthy days
        9: 'heath_pcobese'    -- % of adults obese
    %}

rng('default')

fips_codes = train_inputs(:,1);
demographics = train_inputs(:,2:22);
topic_freqs = train_inputs(:,23:end);

test_demographics = test_inputs(:,2:22);
test_topic_freqs = test_inputs(:,23:end);

%% PCA on LDA topic data
[coeff,score,latent,tsquared,explained,mu] = pca(topic_freqs, 'Centered', false);

% Get a sufficient number of vectors
threshold = 99;
num_vectors = 1;
for i = 1:size(coeff, 1)
    curr_explained = sum(explained(1:i));
    if (curr_explained >= threshold);
        num_vectors = i;
        break
    end
end
num_vectors = 36;
fprintf('Precent explained with %d vectors: %f\n', num_vectors, sum(explained(1:num_vectors)));

train_freqs_dimred = score(:,1:num_vectors);
% test_freqs_center = (test_topic_freqs - repmat(mean(test_topic_freqs),size(test_topic_freqs,1),1));
test_freqs_dimred = test_topic_freqs * coeff(:, 1:num_vectors);

% Combine feature space
train_inputs_dimred = [demographics train_freqs_dimred];
test_inputs_dimred  = [test_demographics test_freqs_dimred];

pred_labels = zeros(size(test_inputs,1),size(train_labels,2));
C_vals = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1];
neu_vals = [1 3 5 10 15 20];

% Optimal hyperparameters
d = size(train_labels,2);
C_min_final = zeros(d,1);
neu_min_final = zeros(d,1);

%test
% C_min_final = [1e-1 1e-5 0.0001 0.1 0.0001 0.1 0.1 0 0];
% neu_min_final = [1 1 5 20 5 1 1 5 1];
C_min_final = [1e-5 1e-5 0.001 0.1 0.0001 1e-3 1e-4 1e-2 0];
neu_min_final = [1 3 10 5 5 1 1 5 1];

for f = 1:d

    X_train = train_inputs_dimred;
    y_train = train_labels(:,f);
    X_test = test_inputs_dimred;

    % Global parameters
    activation_function = 'logsig';
    training_function = 'trainlm';
    loss_function = 'mse';

    fprintf("\n\n=== Feature %d ===\n", f)
    neu_min = 0;
    C_min = 0;
    func_best = 0;
    min_cv_err = Inf;

    num_folds = 5;
    part = cross_validation(size(X_train, 1), num_folds)';

    % For each number of neurons
%{
    for h = 1:6
        neu = neu_vals(h);

        % For each regularization value
        for i = 1:7
            C = C_vals(i);
            tot_cv_err = 0;

            fprintf('testing : i=%d,h=%d ', i, h);

            % Do n-fold validation training
            for j = 1:num_folds
                fprintf('.');
                curlbls = y_train(:);
                X_train_curr = X_train(part~=j,:);
                y_train_curr = curlbls(part~=j);
                X_test_curr = X_train(part==j,:);
                y_test_curr = curlbls(part==j);

                % setting up the net and training
                net = fitnet(neu);
                net.layers{1}.transferFcn = activation_function;
                net.performFcn = loss_function;
                net.trainFcn = training_function;
                net.performParam.regularization = C;
                net = configure(net, X_train_curr', y_train_curr');
                net = train(net, X_train_curr', y_train_curr');

                % predicting
                y_pred = net(X_test_curr')';
                cl_err = sum(sqrt(sum((y_test_curr-y_pred).^2)));
                tot_cv_err = tot_cv_err + cl_err;
            end
            if tot_cv_err < min_cv_err
                min_cv_err = tot_cv_err;
                C_min = C;
                neu_min = neu;
            end
            fprintf('\n');
        end
    end
%}


    % fprintf('\nOPTIMAL:\n');
    % fprintf('C=%d, neu=%d\n', C_min, neu_min);
    % C_min_final(f) = C_min
    % neu_min_final(f) = neu_min


    %% Neural network regression

    % Setting up and training final net w optimal parameters
    net = fitnet(neu_min_final(f));
    net.layers{1}.transferFcn = activation_function;
    net.performFcn = loss_function;
    net.trainFcn = training_function;
    net.performParam.regularization = C_min_final(f);
    net = configure(net, X_train', y_train');
    net = train(net, X_train', y_train');

    % Predicting...
    pred_labels(:,f) = net(X_test')';
end

C_min_final
neu_min_final

end