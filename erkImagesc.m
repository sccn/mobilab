function erkImagesc(X,samplingFrequency,strTitle,showConfidenceInterval)
if nargin < 2, samplingFrequency = 1;end
if nargin < 3, strTitle = '';end
if nargin < 4, showConfidenceInterval = false;end

[n,p] = size(X);
time = linspace(-n/2,n/2,n)/samplingFrequency;
mu = mean(X,2);
figure;
subplot(211);imagesc(time,1:p,X');set(gca,'YDir','normal');title(strTitle);xlabel('Time (sec)')
subplot(212);plot(time,mu);title(['Average ' strTitle]);

if showConfidenceInterval
    sigma = std(X,[],2);
    hold on;
    plot(time,mu+sigma,'r');
    plot(time,mu-sigma,'r');
end