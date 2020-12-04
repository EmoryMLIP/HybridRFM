function X = sampleSn(d,n)
% function X = sampleSn(d,n)
%
% create n i.i.d samples uniformly on S_{d-1}
%
% Input:
%   d - spatial dimension
%   n - number of points
%
% Output:
%   X - matrix of samples, size(X)=[n,d]

if nargin==0   
    d = 3;
    n = 2000;
    X = feval(mfilename,d,n);    
    if not(all(abs(sum(X.^2,2)-1.0)<1e-3))
        error('points not on sphere')
    end    
    if d==3
        figure(d); clf;
        plot3(X(:,1),X(:,2),X(:,3),'.r','MarkerSize',20);
    elseif d==2
        figure(d); clf;    
        plot(X(:,1),X(:,2),'.r','MarkerSize',20);
    end
    return
end

X = randn(n,d);
r = sqrt(sum(X.^2,2));
X = X./r;


