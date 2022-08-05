### pyspark_logr_prediction
from pyspark.ml import PipelineModel

test_df = read("FreddieMac_Spark.PySpark_TestingData")
# load logr pipeline model
lr_pipeline_model = PipelineModel.load("/incorta/IncortaAnalytics/Tenants/ebs_cloud/incorta.ml/models"+ "/logr-pipeline-model2")

# predict
predictions = lr_pipeline_model.transform(test_df) # can work
df_output = predictions.select("seq_num", "label", "predictedLabel", "probability")
df_output.show(5)
save(df_output)

### check performance
import matplotlib.pyplot as plt 
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix 
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics
# Accuracy
evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %f" % accuracy)

# AUC
lr_eval = BinaryClassificationEvaluator(rawPredictionCol="probability", labelCol="indexedLabel")
lr_AUC  = lr_eval.evaluate(predictions)
print("AUC = %.2f" % lr_AUC)

# confusion matrix
# method 1
y_true = predictions.select(['label']).collect()
y_pred = predictions.select(['predictedLabel']).collect()
print("Confusion matrix:")
print(classification_report(y_true, y_pred))

# method 2
cm_lr_result = predictions.crosstab("predictedLabel", "label")
cm_lr_result = cm_lr_result.toPandas()
cm_lr_result

# method 3
y_true = predictions.select("label")
y_true = y_true.toPandas()
y_pred = predictions.select("predictedLabel")
y_pred = y_pred.toPandas()

cnf_matrix = confusion_matrix(y_true, y_pred,labels=['delinquent', 'not_delinquent'])
cnf_matrix
TP = float(cm_lr_result.iat[0,1])
FP = float(cm_lr_result.iat[0,2])
TN = float(cm_lr_result.iat[1,2])
FN = float(cm_lr_result.iat[1,1])
Accuracy = (TP+TN)/(TP+FP+TN+FN)
Recall = TP/(TP+FN)
Specificity = TN/(TN+FP)
Precision = TP/(TP+FP)
F1 = (2*Precision*Recall) / (Precision+Recall)

print ("Accuracy = %0.2f" %Accuracy )
print ("Recall = %0.2f" %Recall )
print ("Specificity = %0.2f" %Specificity )
print ("Precision = %0.2f" %Precision )
print("F1 score = %0.2f" %F1)


# ROC Grafik
# Create ROC grafik from lr_result
PredAndLabels           = predictions.select("probability", "indexedLabel")
PredAndLabels_collect   = PredAndLabels.collect()
PredAndLabels_list      = [(float(i[0][0]), 1.0-float(i[1])) for i in PredAndLabels_collect]
PredAndLabels           = sc.parallelize(PredAndLabels_list)
metrics = BinaryClassificationMetrics(PredAndLabels)
# Visualization ROC curve
FPR = dict()                     # FPR: False Positive Rate
tpr = dict()                     # TPR: True Positive Rate
roc_auc = dict()
 
y_test = [i[1] for i in PredAndLabels_list]
y_score = [i[0] for i in PredAndLabels_list]
 
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
 
plt.figure(figsize=(5,4))
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend(loc="lower right")
plt.show()
