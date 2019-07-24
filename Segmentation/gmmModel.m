classdef gmmModel
   properties(GetAccess='public', SetAccess='public')
      % initialize parameters
      mu = [];
      sigma = [];
      lambda = [];
      lambdai = [];
      k;
      bound;
   end
   methods(Static = true)
       % building a gmm model
       function obj = build(x,k)
           obj = gmmModel;
           % get size of data matrix
           [n,d]=size(x);
           
           % Initialize the gmm mu by kmeans
           [idx,obj.mu] = kmeans(x,k);
           
           % At the first time,I initalize sigma by computing covariance of
           % training data x and adding a regulization term with value 0.01
           % or what,but it doesn't always work very well.So I think it
           % depends on data's.
           
           %obj.sigma = repmat(diag(diag(cov(x))+0.01)*1,1,1,k);
           obj.sigma = repmat(eye(d)*0.001,1,1,k);
           
           obj.k = k;
           
           % Initialize with equal lambda 1/k
           obj.lambda = 1/k*ones(k,1);
           
           % Start to run EM algorithm!
           % Most of the time,when I try to compute the lower bound of
           % current iteration,it falls to NaN,which I will explain in the
           % report and bound function later.Eventually,I just set the 
           % iteration 50 or 100,depending on data's complexity.
           % (Time is invaluable T.T)
           
           for i=1:100
               % E step
               [obj.lambdai] = obj.gmm_Estep(x,k,obj.mu,obj.sigma,obj.lambda);
               % M step
               [obj.lambda , obj.mu , obj.sigma] = obj.gmm_Mstep(x,k,obj.lambdai);
               %obj.bound(i,1) = obj.computeBound(x,k,obj.mu,obj.sigma,obj.lambda,obj.lambdai);
               disp(['Gmm iteration:',num2str(i)]);
           end
       end
       
       % compute gmm pdf for given data x
       function y = pdf(obj,x)
           [n,d]=size(x);
           [k,i]=size(obj.lambda);
           y=0;
           for i=1:k
               y = y+mvnpdf(x,obj.mu(i,:),obj.sigma(:,:,i))*obj.lambda(i,1);
           end
       end
   end
   
   methods(Hidden = true, Static = true)
       
       % E step for EM,computing r_ik,which I denote it as lambdai.
       function [lambdai] = gmm_Estep(x,k,mu,sigma,lambda)
           [n,d]=size(x);
           
           % compute the lambda*norm[mu_k,sigma_k] for all datas
           for i = 1:k
               lambdai(:,i) = lambda(i,1)*mvnpdf(x,mu(i,:),sigma(:,:,i));
           end
           
           % divide each lambda*norm[mu_k,sigma_k] by sum of the row.
           for i = 1:n
               lambdai(i,:) = lambdai(i,:)/sum(lambdai(i,:));
           end
       end
       
       % M step for EM,computing lambda,mu and sigma.
       % We have to keep an eye on the computing of sigma, it's likely to
       % fall into not being positive-definite in some of the task. ( MNIST
       % ) So when doing MNIST, I add a regularization term 0.1*I. It
       % prevents computing from errors!
       function [lambda , mu , sigma] = gmm_Mstep(x,k,lambdai)
           [n,d]=size(x);
           
           % update the lambda
           lambda = (sum(lambdai)/n).';
           
           % update the mu, first compute the sum_{i=1}{I}r_ik*x_i
           mu = (lambdai.')*x;
           for i = 1:k
               % divide sum_{i=1}{I}r_ik*x_i by sum of row, finishing
               % update of mu.
               mu(i,:) = mu(i,:)/sum(lambdai(:,i));
               
               % temp = x_i-mu_k.
               temp = (x.' - repmat(mu(i,:).',1,n));
               
               % update the sigma,  sum_{i=1}{I}r_ik*temp*temp^t / sum_{i=1}{I}r_ik
               % In the MNIST case, add a regularization term 0.1*I
               sigma(:,:,i) = temp*diag(lambdai(:,i))*temp.'/sum(lambdai(:,i));
%                sigma(:,:,i) = (diag(diag(temp*diag(lambdai(:,i))*temp.'))/sum(lambdai(:,i))  + eye(d)*0.1);
           end
       end
       
       % When calculating the log of pdf, there exists some values of
       % -infinite, which could cause NaN! So I didn't take this as a
       % condition to stop the iteration.
       function [bound] = computeBound(x,k,mu,sigma,lambda,lambdai)
           [n,d]=size(x);
           for i = 1:k
               lambda_norm(:,i) = lambda(i,1)*mvnpdf(x,mu(i,:),sigma(:,:,i));
           end
           bound = sum(sum(lambdai.'*log(lambda_norm)));
       end
   end
end