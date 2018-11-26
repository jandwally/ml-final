function [part] = cross_validation(n, n_folds)

	% cross_validation - Randomly generate cross validation partition.
	%
	% Usage:
	%
	%  PART = cross_validation(N, N_FOLDS)
	%
	% Randomly generates a partitioning for N datapoints into N_FOLDS equally
	% sized folds (or as close to equal as possible). PART is a 1 X N vector,
	% where PART(i) is a number in (1...N_FOLDS) indicating the fold assignment
	% of the i'th data point.

	% Get a partition
	part = repmat(1:n_folds,1,ceil(n/n_folds));
	part = part(1:n);
	part = part(randperm(n));

end