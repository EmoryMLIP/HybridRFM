%
% Example for applying hybrid regularization tools to a random feature
% model using the MNIST/CIFAR dataset
%
% to run this code, you need to get a working copy of the following
% packages and add it to your path:
%
% 1) https://github.com/jnagy1/IRtools

% In this file, columns of his are training square error, testing square
% error and norm of the weights, respectively.
% clear all;
close all;
rng("default")
rng(1)

nTrain = 2^10;
nVal   = 10000;

if not(exist('dataset','var'))
    dataset = 'CIFAR10'; % 'MNIST' or 'CIFAR10';
end

sample = 'Sd';

if strcmp(dataset, 'MNIST')
    [Y,C] = setupMNIST(nTrain+nVal);
elseif strcmp(dataset, 'CIFAR10')
    [Y,C] = setupCIFAR10(nTrain+nVal);
else
    warning('no such data!')
    return
end

dim1=size(Y,1);dim2=size(Y,2);dim3=size(Y,3);
Y    = normalizeData(Y,dim1*dim2*dim3);

id = randperm(size(C,2));
idt = id(1:nTrain);
idv = id(nTrain+1:end);
Yt  = reshape(Y(:,:,:,idt),dim1*dim2*dim3,[]); Ct = C(:,idt);
Yv  = reshape(Y(:,:,:,idv),dim1*dim2*dim3,[]); Cv = C(:,idv);

ms  = 2.^(4:15);
MaxIter_list = 2.^(4:10);
his = zeros(numel(MaxIter_list),numel(ms),3);
INFO = cell(numel(MaxIter_list),numel(ms),size(C,1));

for k=1:numel(ms)
    m = ms(k);
    
    switch sample
        case 'Sd'
            K = sampleSd(dim1*dim2*dim3,m-1);
            b = sampleSd(m-1,1)';
        otherwise
            sample = 'uniform';
            K = 2*(rand(m-1,dim1*dim2*dim3)-0.5);
            b = 2*(rand(m-1,1)-0.5);
    end
    
    Zt = [max(K*Yt+b,0); ones(1,size(Yt,2))];
    Zv = [max(K*Yv+b,0); ones(1,size(Yv,2))];
    
    WOpt = zeros(size(Ct,1),m);
    
    for i=1:numel(MaxIter_list)
        if MaxIter_list(i)>m
            break
        end
        fprintf('%s\tdataset=%s, \t m=%d, maxiter=%d\n',mfilename,dataset,m,MaxIter_list(i));
        
        for j = 1:size(Ct,1)
            options.MaxIter=MaxIter_list(i);
            options.RegParam = 'gcv';
            options.NoStop = 'on';
            options.Reorth = 'on';
            options.IterBar = 'off';
            [wHyBR, info] = IRhybrid_lsqr(Zt', Ct(j,:)', options);
            WOpt(j,:) = wHyBR;
            INFO{i,k,j} = info;
        end
        
        his(i,k,1) = norm(WOpt*Zt-Ct,'fro')^2/(2*size(Zt,2));
        his(i,k,2) = norm(WOpt*Zv-Cv,'fro')^2/(2*size(Zv,2));
        his(i,k,3) = norm(WOpt,'fro');
    end
end

save(sprintf('%s_%s_%s.mat',mfilename,dataset,sample),'his','ms','MaxIter_list','INFO')

figure
hold on
for i=1:numel(MaxIter_list)
    loglog(ms(i:end),his(i,i:end,2),'linewidth',2,'MarkerSize',10,'Marker','x','DisplayName',strcat('test loss iter=',num2str(MaxIter_list(i))))
    
    title(sprintf(strcat('Test error for hybrid', dataset ,' data, n=%d'), nTrain))
    ylabel('Test Error', "FontSize", 20)
    xlabel('m, in 2^x',"FontSize", 20)
    xticks(ms)
    xticklabels(split(num2str(log2(ms))))
end
% set(gca, 'YScale', 'log')
% set(gca, 'XScale', 'log')
yL = get(gca,'YLim');
legend('Location', 'southwest')

loglog([nTrain;nTrain],yL,'-.k','LineWidth',1,'DisplayName','Number of Examples');
