function [ pdf ] = gaussian_pdf(X, mu, Sigma)
%Inputs:
%X - mxd Matrix of data points
%mu - Row vector for the mean
%Sigma - Covariance matrix

% Get the vector length.
d = size(X, 2);

% Subtract the mean from every data point.
meanDiff = bsxfun(@minus, X, mu);

% Calculate the multivariate gaussian.
pdf = 1 / sqrt((2*pi)^d * det(Sigma)) * exp(-1/2 * sum((meanDiff * inv(Sigma) .* meanDiff), 2));

end