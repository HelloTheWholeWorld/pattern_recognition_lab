function logpSum = GMMpdf( X, Alphas, Means, Sigmas )

[N,d] = size(X);
K = length(Alphas);

logp = zeros(N,K);
logpSum = zeros(N,1);

for k = 1:K
    Mean = Means(k,:);
    Sigma = squeeze(Sigmas(k,:,:));
    invSigma = inv(Sigma);
    logAlpha = log(Alphas(k));
    [v,ev] = eig(Sigma);
    logdetSigma = 0;
    for j = 1:d
        logdetSigma = logdetSigma + log(ev(j,j));
    end
    
    logp(:,k) = -0.5*d*log(2*pi) - 0.5*logdetSigma - 0.5*sum((X-repmat(Mean,N,1))*invSigma.*(X-repmat(Mean,N,1)),2) + logAlpha;
end
        
logpSum = logp(:,1);
for k = 2:K
    logpSum = logsum(logpSum,logp(:,k));
end