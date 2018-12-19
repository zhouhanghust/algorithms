load('ShangZZS.mat');
cmd = ['-s 3 -t 2','-c 1 -g 0.7'];
model = libsvmtrain(y_train(:,1),X_train,cmd);
plot(y_test(:,1));
hold on;
plot(predicted_label);

