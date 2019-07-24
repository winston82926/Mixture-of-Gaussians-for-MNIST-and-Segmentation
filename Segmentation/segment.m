clear all; close all; clc;
%% read image data
filename = '221272.jpg';
x=imread(['Data\',filename]);

% To save the time in training, resize the image.
% x is the raw image data, x_down is the downsized image data used to train
% the GMM.
x= imresize(x,1);
[m,n]=size(x(:,:,1));
x_down = imresize(x,0.05);

% Reconstruct the image matrix to vector form and normalize it to the range 0~1.
train = double([reshape(x_down(:,:,1),[],1) reshape(x_down(:,:,2),[],1) reshape(x_down(:,:,3),[],1)])/255;
test = double([reshape(x(:,:,1),[],1) reshape(x(:,:,2),[],1) reshape(x(:,:,3),[],1)])/255;

%% GMM training
% set the mixture amount k.
k=10;
% train the GMM
gmm_seg = gmmModel.build(train,k);

%% GMM testing
% Output the probabilities, corresponding to every Gaussians, of every
% pixel.
for i=1:k
    y(:,i) = mvnpdf(test,gmm_seg.mu(i,:),gmm_seg.sigma(:,:,i))*gmm_seg.lambda(i,1);
end

% find the highest probability and label the test example.
[M,I] = max(y.');

% restore the output labeled vector to the original image matrix.
out = reshape(I,m,[]);

% plot each pixel by mu.
for i =1:m
    for j =1:n
        outim(i,j,1)=gmm_seg.mu(out(i,j),1);
        outim(i,j,2)=gmm_seg.mu(out(i,j),2);
        outim(i,j,3)=gmm_seg.mu(out(i,j),3);
    end
end

% Build the colormap, which contains all of the mu in GMM.
R = [];
G = [];
B = [];
for i = 1:k
    R = [R ;ones(50,50)*gmm_seg.mu(i,1)*255];
    G = [G ;ones(50,50)*gmm_seg.mu(i,2)*255];
    B = [B ;ones(50,50)*gmm_seg.mu(i,3)*255];
end
colorMap(:,:,1) = uint8(R);
colorMap(:,:,2) = uint8(G);
colorMap(:,:,3) = uint8(B);

% Show image and save it.
test_myGmm = uint8(outim*255);
imshow(test_myGmm);
imwrite(test_myGmm,['Data\',filename,'_myGmm_k=',num2str(k),'.png']);
% imshow(colorMap);
%% compare Original with Segmented
% subplot(1,2,1);
% imshow(test_myGmm);
% subplot(1,2,2);
% imshow(x);
