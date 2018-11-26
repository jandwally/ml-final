function [normalized_llh] = compute_nllh(X,k, mu, sigma,phi)
%Inputs:
%X - mxd Matrix of data points
%k - Number of Gaussians
%mu - matrix of all the gaussian means
%sigma - Cell array of Covariance matrices of all gaussians
%phi - array of mixing coefficients


m = size(X,1);
%Initialize matrix to store P(x_i|z_i) values
g = zeros(m,k);

%Compute P(x_i|z_i)
for j = 1:k
     g(:,j)= gaussian_pdf(X,mu(j,:),sigma{j});
end

%Compute P(x_i|z_i)*phi(z_i)
g_phi = bsxfun(@times, g, phi);

%Compute log likelihood
llh = sum(log(sum(g_phi,2)),1);

%Compute normalized log likelihood
normalized_llh = llh/m;
end