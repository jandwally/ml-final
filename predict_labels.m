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
	    9: 'heath_pcobese'	  -- % of adults obese
	%}

rng('default')

%% PCA
[coeff,score,latent,tsquared,explained,mu] = pca(train_inputs);
num_vectors = 30;
train_inputs_dimred = score(:,1:num_vectors);
test_center = (test_inputs - repmat(mean(test_inputs),size(test_inputs,1),1));
test_inputs_dimred = test_center * coeff(:, 1:num_vectors);

part = repmat(1:7,1,ceil(size(train_inputs,1)/7));
part = part(1:size(train_inputs,1));
part = part(randperm(size(train_inputs,1)));
pred_labels = zeros(size(test_inputs,1),size(train_labels,2));
C_vals = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1];
neu_vals = [1 3 5 10 15 20 30];

% For summary's sake
d = size(train_labels,2);
C_min_final = zeros(d,1);
neu_min_final = zeros(d,1);

for f = 1:d

	X_train = train_inputs_dimred;
    %X_train = train_inputs;
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

	% For each number of neurons
    for h = 1:7
    	neu = neu_vals(h);

    	% For each regularization value
	    for i = 1:6
	        C = C_vals(i);
	        tot_cv_err = 0;

	        fprintf('testing : i=%d,h=%d ', i, h);

	        num_folds = 7;
	        part = cross_validation(size(X_train, 1), num_folds)';

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

	fprintf('\nOPTIMAL:\n');
	fprintf('C=%d, neu=%d\n', C_min, neu_min);
	C_min_final(f) = C_min
	neu_min_final(f) = neu_min

    %% Neural network regression

    % Setting up and training final net w optimal parameters
    net = fitnet(neu_min);
    net.layers{1}.transferFcn = activation_function;
    net.performFcn = loss_function;
    net.trainFcn = training_function;
    net.performParam.regularization = C_min;
    net = configure(net, X_train', y_train');
    net = train(net, X_train', y_train');

    % Predicting...
    pred_labels(:,f) = net(X_test')';
end

C_min_final
neu_min_final

% y_p_tr = sign(net(X_train')');
% fprintf('C used by final model: %f\n', C_min);
% fprintf('training error achieved by final model: %f\n', ...
%     classification_error(y_p_tr, y_train));
% fprintf('min cross validation error across all parameters: %f\n', ...
%     min_cv_err/5);
end