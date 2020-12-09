%
% Example for applying hybrid regularization tools to a random feature
% model using the MNIST/CIFAR dataset
%

% In this file, columns of his are training square error, testing square
% error and norm of the weights, respectively.
% clear all;
close all; 
rng("default")
rng(1)

doPlot = true;
nTrain = 2^10;
nVal   = 10000;

if not(exist('dataset','var'))
    dataset = 'CIFAR10'; % 'MNIST' or 'CIFAR10';
end

sample = 'Sd';

switch dataset
    case 'MNIST'
        [Y,C] = setupMNIST(nTrain+nVal);
    case 'CIFAR10'
        [Y,C] = setupCIFAR10(nTrain+nVal);
    otherwise
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
his = zeros(numel(ms),2);
tt = logspace(-6,10,100);
ftest_all = zeros(numel(ms),numel(tt));
ftrain_all = zeros(numel(ms),numel(tt));

for k=1:numel(ms)
    m = ms(k);
    fprintf('%s : \t dataset=%s, \t m=%d\n',mfilename,dataset,m);
    
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
    diagS = diag(S);
    phiS = @(alpha) diagS./(diagS.^2+nTrain*alpha^2);
    WOpt = @(alpha) (Ct*V)*(phiS(alpha).*U');
    train_error = @(alpha) norm(WOpt(alpha)*Zt-Ct,'fro')^2/(2*size(Zt,2));
    test_error = @(alpha) norm(WOpt(alpha)*Zv-Cv,'fro')^2/(2*size(Zv,2));
    
    ftest = 0*tt;
    ftrain = 0*tt;
    for j=1:numel(tt)
        ftest(j) = test_error(tt(j));
        ftrain(j) = train_error(tt(j));
    end
    ftest_all(k,:) = ftest;
    ftrain_all(k,:) = ftrain;
    
    [f0,j0] = min(ftest);
    [opt_alpha,opt_error,flag] = fminsearch(test_error,tt(j0));
    
    if doPlot
       fig = figure(); clf;
       fig.Name = sprintf('WD_%s,m-%d',dataset,m);
       loglog(tt,ftest,'LineWidth',2,'DisplayName','test error')
       hold on;
       loglog(opt_alpha,opt_error,'.r','MarkerSize',30,'DisplayName','optimal')
       legend()
       drawnow
    end

    his(k,:) = [opt_alpha, opt_error];
    if flag~=1
        warning('fminbnd did not converge');
    end
    fprintf('m=%d\topt_alpha=%1.2e\topt_error=%1.4f\n',m,opt_alpha,opt_error);
end

save(sprintf('%s_%s_%s.mat',mfilename,dataset,sample),'his','ms','tt','ftest_all','ftrain_all')
