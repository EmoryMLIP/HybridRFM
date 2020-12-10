%
% Example for applying hybrid regularization tools to a random feature
% model using the MNIST/CIFAR dataset
%
% to run this code, you need to get a working copy of the following
% packages and add it to your path:
% 
% 1) https://github.com/XtractOpen/Meganet.

% In this file, columns of his are training square error, testing square
% error and norm of the weights, respectively.
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
trial_no = 5;
his = zeros(numel(ms),3, trial_no);

sample = 'Sd';

for k=1:numel(ms)   
    m = ms(k);
    for trial=1:trial_no

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

        [U,S,V] = svd(Zt, 'econ');
        WOpt = (Ct*V)*(S\U');

        his(k,1,trial) = norm(WOpt*Zt-Ct,'fro')^2/(2*size(Zt,2));
        his(k,2,trial) = norm(WOpt*Zv-Cv,'fro')^2/(2*size(Zv,2));
        his(k,3,trial) = norm(WOpt,'fro');
        
    end
end

figure

hold on
loglog(ms,mean(his(:,1,:),3),'linewidth',2,'MarkerSize',10,'Marker','x','DisplayName','Training Error')
loglog(ms,mean(his(:,2,:),3),'linewidth',2,'MarkerSize',10,'Marker','x','DisplayName','Test Error')
loglog(ms,mean(his(:,3,:),3),'linewidth',2,'MarkerSize',10,'Marker','x','DisplayName','Norm of W')

title(sprintf(strcat('Double Descent for ', dataset ,' data, n=%d'), nTrain))
ylabel('Error', "FontSize", 20)
xlabel('m, in 2^x',"FontSize", 20)
xticks(ms)
xticklabels(split(num2str(log2(ms))))

legend('Location', 'southwest')
set(gca, 'YScale', 'log')
set(gca, 'XScale', 'log')
yL = get(gca,'YLim');

loglog([nTrain;nTrain],yL,'-.k','LineWidth',1,'DisplayName','Number of Examples');

save(sprintf('%s_%s_%s.mat',mfilename,dataset,sample),'his','ms')
