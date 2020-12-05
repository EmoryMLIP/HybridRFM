%
% Example for applying hybrid regularization tools to a random feature
% model using the CIFAR10 dataset
%

rng('default')
rng(1)
close all; 


nTrain = 2^10;
nVal   = 10000;
[Y,C] = setupCIFAR10(nTrain+nVal);
dim1=size(Y,1);dim2=size(Y,2);dim3=size(Y,3);
Y    = normalizeData(Y,dim1*dim2*dim3);

%% Compare random feature model without regularization and with HyBR
trial_no = 5;
       
% divide into training and validation data
id = randperm(size(C,2));
idt = id(1:nTrain);
idv = id(nTrain+1:end);
Yt  = reshape(Y(:,:,:,idt),dim1*dim2*dim3,[]); Ct = C(:,idt);
Yv  = reshape(Y(:,:,:,idv),dim1*dim2*dim3,[]); Cv = C(:,idv);

m_list = [2^9,2^10,2^11];
his = zeros(numel(m_list), trial_no);

sample = 'Sd';

for i_train = 1:size(m_list,2)
    
    m = m_list(i_train);
    class_no = 1;
    rank_Z = min(m,nTrain);
    sing_vec = zeros(1, rank_Z);
    ip_vec = zeros(1, rank_Z);
    ipnorm_vec = zeros(1, rank_Z);

    for j=1:trial_no
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
        
        [U, S, V] = svd(Zt, 'econ');
        diagS = diag(S);
        sing_vec = sing_vec + diagS(1:rank_Z)'/trial_no;
        ip_vec = ip_vec + abs(Ct(class_no,:)*V(:,1:rank_Z))/trial_no;
        ipnorm_vec = ipnorm_vec + (abs(Ct(class_no,:)*V(:,1:rank_Z)))./diagS(1:rank_Z)'/trial_no;

        WOpt = (Ct*V)*(S\U');
        his(i_train,j) = norm(WOpt*Zv-Cv,'fro')^2/(2*size(Zv,2));
    end
    figure(i_train)
    semilogy(sing_vec, 'LineWidth',1.5)
    hold on 

    semilogy(ip_vec, 'LineWidth',1.5)
    semilogy(ipnorm_vec, 'LineWidth',1.5)
    title(strcat("picard, m=",num2str(m)))
    if i_train==1
        legend("\sigma_i","|c_{"+num2str(class_no)+",:}*v_i|","|c_{"+num2str(class_no)+",:}*v_i|/ \sigma_i",'fontsize', 7, 'Location', 'southeast')
    end
    fprintf("Average Test Error: %.2f, m=%d",mean(his(i_train,:)),num2str(m))
end
