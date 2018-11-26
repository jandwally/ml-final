
% Use this file to test if our methods work, so we don't need to modify predict_labels.m

function err = test(training_data, training_labels)

	n = size(training_data, 1)
	p = size(training_data, 2)
	d = size(training_labels, 2)

	% Get two folds to test with
	part = cross_validation(n, 2)
	actual_labels = training_labels(part==2,:);
	testing_data = training_data(part==2,:);
	training_labels = training_labels(part==1,:);
	training_data = training_data(part==1,:);

	% Call a method
	pred_labels = predict_labels(training_data, training_labels, testing_data)

	% Compute cross-validation error
	error_metric(pred_labels, actual_labels)

end