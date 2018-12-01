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

	pred_labels = zeros(size(test_inputs,1),size(train_labels,2));

	% 1: 'health_aamort'
	model1 = fitrlinear(train_inputs, train_labels(:,1));
	pred_labels(:,1) = predict(model1, test_inputs);

	% 2: 'health_fairpoor'
	% 3: 'health_mentunh'
	model3 = fitrlinear(train_inputs, train_labels(:,3));
	pred_labels(:,3) = predict(model3, test_inputs);

	% 4: 'health_pcdiab'
	% 5: 'health_pcexcdrin'
	% 6: 'health_pcinact'
	% 7: 'health_pcsmoker'
	% 8: 'health_physunh'
	model8 = fitrlinear(train_inputs, train_labels(:,8));
	pred_labels(:,8) = predict(model8, test_inputs);

	% 9: 'heath_pcobese'
	

end