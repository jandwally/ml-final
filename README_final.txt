John Wallison
Nikita Lapin
Final Submission

We have the following functions:
(generative)
pcr.m				choosing some pca vectors and doing regression on those

(discriminative)
trees.m				using decision trees with cross validation on number of splits


knn.m				knn with cross validation on k

(novelty)
predict_labels.m		neural nets with pca
-novel because only did pca on word data, and appended it to the census data, then feed
 that into the net
 
all above functions follow predict_labels.m calling conventions (train_inputs,train_labels,test_inputs)

test.m				runs tests on all 4 files, reports errors
cross_validation.m		used in above files
error_metric.m