function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
% Here is an outline for forward propagation using the vectorized method.
% This is an implementation of the formula in Figure 2 on Page 11 of
% ex3.pdf.
% 
%     Add a column of 1's to X (the first column), and it becomes 'a1'.
%     Multiply by Theta1 and you have 'z2'. Compute the sigmoid() of 'z2'
%     and add a column of 1's, and it becomes 'a2' Multiply by Theta2,
%     compute the sigmoid() and it becomes 'a3'. Now use the max(a3, [], 2)
%     function to return two vectors - one of the highest value for each
%     row, and one with its index. Ignore the highest values. Keep the
%     vector of the indexes where the highest values were found. These are
%     your predictions.
% 
% Note: When you multiply by the Theta matrices, you'll have to use
% transposition to get a result that is the correct size.
% 
% Note: the predictions must be returned as a column vector - size (m x 1).
% If you return a row vector, the script will not compute the accuracy
% correctly.
% 
% ------ dimensions of the variables ---------
% 
% a1 is (m x n), where 'n' is the number of features including the bias
% unit
% 
% Theta1 is (h x n) where 'h' is the number of hidden units
% 
% a2 is (m x (h + 1))
% 
% Theta2 is (c x (h + 1)), where 'c' is the number of labels.
% 
% a3 is (m x c)
% 
% p is a vector of size (m x 1)


% First, add a column of ones to input X
X = [ones(m,1) X];

% Now calculate z1 (Theta1 is read in with the data)
z1 = Theta1 * X';
% Implement sigmoid function on z1 as per vectorized NN procedure
z1 = sigmoid(z1);
% Now add the bias node to z1 to calculate a2
a2 = [ones(1, size(z1, 2)); z1];
% Now calculate z2 and use that to calculate a3
z2 = Theta2 * a2;
a3 = sigmoid(z2);
% Transpose the matrix and use the procedure from predictOneVsAll.m
a3 = a3';

for i=1:size(a3,1),
    [x, ix] = max(a3(i,:)); %
    p(i) = ix;
end



% =========================================================================


end
