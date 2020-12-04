clear all;
close all; clc;

driverSVDandPicard;

datasets = {'MNIST', 'CIFAR10'};

for k=1:numel(datasets)
	dataset = datasets{k};
	driverGradientFlow;
	driverWeightDecay;
	driverHybrid;
end
