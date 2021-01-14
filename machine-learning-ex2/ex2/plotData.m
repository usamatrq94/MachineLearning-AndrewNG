function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

m=size(X);
j=1;
k=1;

for i=1:100

if y(i) == 1
  
  N(j,1)=X(i,1);
  N(j,2)=X(i,2);
  j=j+1;
else
  
  P(k,1)=X(i,1);
  P(k,2)=X(i,2);
  k=k+1;
endif

endfor

% =========================================================================

plot(N(:,1),N(:,2),"k+", "markersize", 10,'LineWidth',2)
hold on

plot(P(:,1),P(:,2),"o",'MarkerEdgeColor','b','MarkerFaceColor','y',"markersize", 8)
hold off;

end
