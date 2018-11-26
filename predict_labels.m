function pred_labels=predict_labels(train_inputs,train_labels,test_inputs)

	pred_labels = zeros(size(test_inputs,1),size(train_labels,2));
	for i = 1:size(train_labels,2)
	    Mdl = fitrlinear(train_inputs, train_labels(:,i));
	    pred_labels(:,i) = predict(Mdl, test_inputs);
	end

end