function [mu, phi, sigma] = gaussian_mixture_EM_alg(X, K)

    m = size(X,1);      % number of observations
    d = size(X,2);      % features

    % Initializing values
    mu = X(randperm(m, K), :)
    phi = (1/K) * ones(K,1);
    sigma = cell(K,1);
    for k = 1:K
        sigma{k} = eye(d);
    end
    repmat(eye(K),1,1,K);
    likelihood = compute_nllh(X,K,mu,sigma,phi');

    for t = 1:1000
        % E step
        P_z_K = [];
        for k = 1:K
            P_z_k = phi(k) .* gaussian_pdf(X,mu(k,:),sigma{k});
            P_z_K = cat(2,P_z_K,P_z_k);
        end
        P_z_K = P_z_K./sum(P_z_K,2); % normalizing (m x K matrix)

        % M step (X is m x d)
        for k = 1:K
            mu(k,:) = sum(P_z_K(:,k).*X,1) ./ sum(P_z_K(:,k));
            phi(k) = (1./m) .* sum(P_z_K(:,k));
            mndf = bsxfun(@minus, X, mu(k,:));
            upsum = 0;
            for i = 1:m
                upsum = upsum + P_z_K(i,k).*mndf(i,:)'*mndf(i,:);
            end
            sigma{k} = upsum ./ sum(P_z_K(:,k));
        end

        % Checking likelihood, for stopping condition
        curlik = compute_nllh(X,K,mu,sigma,phi');
        diff = curlik - likelihood;
        if diff < 1e-6
            break;
        end
        likelihood = curlik;
    end
end