function [Y_pred]= knn(X_train, Y_train, X_test, Y_test)

% calculate the distance between test data point and all train data points
[~,I] = pdist2(X_train,X_test, 'euclidean', 'Smallest',1);

% get the labels for the distances in order
Y_pred = Y_train(I');

end