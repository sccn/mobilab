% See Gery Casiez, Nicolas Roussel, Daniel Vogel.  1 ? Filter:  A Simple Speed-based Low-pass Filter for
% Noisy Input in Interactive Systems.  CHI?12, the 30th Conference on Human Factors in Computing
% Systems, May 2012, Austin, United States. ACM, pp.2527-2530, 2012, <10.1145/2207676.2208639>.
% <hal-00670496>

%Example signal noisy sine wave
t = -pi:0.01:pi;
x = linspace(-pi, pi, length(t));
noisySignal = sin(x)+0.5*rand(1, length(x));

%Declare oneEuro object
a = oneEuro;
%Alter filter parameters to tune
a.mincutoff = 1.0;
a.beta = 0.0;

filteredSignal = zeros(size(noisySignal));

for i = 1:length(noisySignal)
    filteredSignal(i) = a.filter(noisySignal(i),i);
end

newfig;
plot(t, noisySignal);
hold on;
plot(t, filteredSignal);