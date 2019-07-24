clear all; close all; clc;

%% load MNIST data, with training set and test set.
train = load('Train.mat');
test = load('Test.mat');
train_data = [train.Train.images.' train.Train.labels];
test_data = [test.Test.images.' test.Test.labels];

% get size of data matrix
[train_data_m , train_data_n] = size(train_data);

%% GMM training
% initialize  a gmm cell.
gmm = {};

% set the mixture amount k.
k = 5;

% train 10 GMM for 10 different digit.
for i = 1:10
    gmm{i} = gmmModelforMNIST.build(train_data(train_data(:,train_data_n) == (i-1), 1:train_data_n-1),k);
end
% I have saved this variables in gmm.mat. If teacher don't want to wait for
% training, you can just load it!

%% GMM testing
% initialize the output matrix, which stores 10 probabilities, corresponding to 10 GMM, for each test example. 
y = [];

% compute the probabilities.
for i = 1:10
    y = [y gmmModelforMNIST.pdf(gmm{i},test_data(:,1:train_data_n-1))];
end

% find the highest probability and label the test example.
[M,I] = max(y.');
I=I.'-ones(10000,1);

% compare the labels with the true answer to calculate the error rate.
[error_rate temp]=size(find(I-test.Test.labels));
error_rate=error_rate/10000;


%% show the mu and sigma for each GMM and write into image file.
% imshow(vec2mat(gmm_3.mu(1,:),20).');
% [mat,padded] = vec2mat(images(:,4),28);
% imshow((mat.')*255);
% for i=1:10
%     for j=1:5
%         imwrite(vec2mat(diag(gmm{i}.sigma(:,:,j)),20).',['Data_k=5\',num2str(i),'_',num2str(j),'.png']);
%     end
% end

%% testing for different number of k
% for j = 5:20
%     gmm = {};
%     for i = 1:10
%         gmm{i} = gmmModelforMNIST.build(train_data(train_data(:,train_data_n) == (i-1), 1:train_data_n-1),j);
%     end
%     y = [];
%     for i = 1:10
%         y = [y gmmModelforMNIST.pdf(gmm{i},test_data(:,1:train_data_n-1))];
%     end
%     [M,I] = max(y.');
%     I=I.'-ones(10000,1);
%     [error_rate temp]=size(find(I-test.Test.labels));
%     e(j) = error_rate/10000;
% end