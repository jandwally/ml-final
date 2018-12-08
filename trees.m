function pred_labels=trees(train_inputs,train_labels,test_inputs)

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

%rng('default')

fips_codes = train_inputs(:,1);
demographics = train_inputs(:,2:22);
topic_freqs = train_inputs(:,23:end);

test_demographics = test_inputs(:,2:22);
test_topic_freqs = test_inputs(:,23:end);

% Optimal hyperparameters
d = size(train_labels,2);

tree_ratio_final = zeros(d,1);

for f = 1:d
    y_train = train_labels(:,f);
    X_train = train_inputs(:,2:end);
    X_test = test_inputs(:,2:end);
    
    fprintf("\n\n=== Feature %d ===\n", f);
    
    numtests = 7;
    ratios = (0:numtests)*0.01/numtests;
    opt_ratio = 0;
    min_cv_err = Inf;
    num_folds = 5;
    part = cross_validation(size(X_train, 1), num_folds)';
    % For each number of neurons
    for i = 1:numtests
        ratio = ratios(i);
        tot_cv_err = 0;
        fprintf('testing : i=%d', i);
        % Do n-fold validation training
        for j = 1:num_folds
            fprintf('.');
            X_train_curr = X_train(part~=j,:);
            y_train_curr = y_train(part~=j);
            X_test_curr = X_train(part==j,:);
            y_test_curr = y_train(part==j);
            % setting up the net and training
            splt = ratio * (size(X_train_curr,1)-1);
            tree = fitrtree(X_train_curr,y_train_curr,'MaxNumSplits',splt);
            % predicting
            y_pred = predict(tree,X_test_curr);
            cl_err = sum(sqrt(sum((y_test_curr-y_pred).^2)));
            tot_cv_err = tot_cv_err + cl_err;
        end
        if tot_cv_err < min_cv_err
            min_cv_err = tot_cv_err;
            opt_ratio = ratio;
        end
        fprintf('\n');
    end
    tree_ratio_final(f) = opt_ratio;

    % fprintf('\nOPTIMAL:\n');
    % fprintf('C=%d, neu=%d\n', C_min, neu_min);
    % C_min_final(f) = C_min
    % neu_min_final(f) = neu_min


    % tree regression
    splt = opt_ratio * (size(X_train,1)-1);
    tree = fitrtree(X_train,y_train,'MaxNumSplits',splt);
    % Predicting...
    pred_labels(:,f) = predict(tree,X_test);
end
tree_ratio_final
end