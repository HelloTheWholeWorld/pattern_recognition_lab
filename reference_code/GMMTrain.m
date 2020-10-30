function [Alphas, Means, Sigmas] = GMMTrain( X, K )

[N,d] = size(X);

TH = 0.0000001;              %ÊÕÁ²ãÐÖµ

Means = repmat(mean(X),K,1) + rand(K,d)*10;
for k = 1:K
    Sigmas(k,:,:) = eye(d,d)*20;
end
Alphas = ones(K)/K;

logp = zeros(N,K);

LSum = 10000;
OLSum = 0;

while( abs(LSum-OLSum) > TH )
    OLSum = LSum;
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
    
    p = exp(logp-repmat(logpSum,1,K));
    Sum = sum(p);
    LSum = sum(logpSum)/N;
    
    Alphas = mean(p);
    for k = 1:K
        Means(k,:) = sum(X.*repmat(p(:,k),1,d))./repmat(Sum(k),1,d);
        Sigmas(k,:,:) = ((X-repmat(Means(k,:),N,1)).*repmat(p(:,k),1,d))'*(X-repmat(Means(k,:),N,1))./repmat(Sum(k),d,d);
    end
    
    fprintf( '     %f\r', LSum );
end

