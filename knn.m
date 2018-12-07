function pred_labels=predict_labels2(train_inputs,train_labels,test_inputs)
part = repmat(1:2,1,ceil(size(train_inputs,1)/2));
part = part(1:size(train_inputs,1));
part = part(randperm(size(train_inputs,1)));
test_labels = train_labels(part==2,:);
test_inputs = train_inputs(part==2,:);
train_labels = train_labels(part==1,:);
train_inputs = train_inputs(part==1,:);
pred_labels = zeros(size(test_inputs,1),size(train_labels,2));
for i = 1:size(train_labels,2)
    Mdl = fitcknn(train_inputs, train_labels(:,i),...
                  'NumNeighbors',10,'Standardize',1);
    pred_labels(:,i) = predict(Mdl, test_inputs);
end
error_metric(pred_labels,test_labels)
end