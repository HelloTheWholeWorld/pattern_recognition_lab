load samples.mat;

for c = 0:9
    id = find(TrainLabels==c);
    [GMM(c+1).Alpha,GMM(c+1).Mu,GMM(c+1).Sigma] = GMMTrain( TrainSamples(id,:), 10);
end

logpdf = zeros(10000,10);
for c = 0:9
    logpdf(:,c+1) = GMMpdf(TestSamples,GMM(c+1).Alpha,GMM(c+1).Mu,GMM(c+1).Sigma);
end

[y,l] = max(logpdf,[],2);
sum(TestLabels~=(l-1))

