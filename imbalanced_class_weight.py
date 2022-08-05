### Imbalanced - Set class weights
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer, StringIndexerModel
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression

# read dataframe
train_df = read("FreddieMac_Spark.PySpark_TrainingData")

# Class Weights in PySpark
balancingRatio = train_df.filter(F.col('indexedLabel') == 1.0).count() / train_df.count()
calculateWeights = F.udf(lambda x: 1 * balancingRatio if x == 0 else (1 * (1.0-balancingRatio)), DoubleType())
weightedDataset = train_df.withColumn('classWeightCol', calculateWeights('indexedLabel'))
weightedDataset.show(5)

# load label indexer
labelIndexer = StringIndexerModel.load("/incorta/IncortaAnalytics/Tenants/ebs_cloud/incorta.ml/models"+ "/string-indexer-model") 
lr = LogisticRegression(featuresCol='indexedFeatures', labelCol='indexedLabel', 
	                    maxIter=10, weightCol='classWeightCol')
# Convert indexed labels back to original labels.
labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                               labels=labelIndexer.labels)
pipeline = Pipeline(stages=[lr, labelConverter])
# Train model.  This also runs the indexers.
model = pipeline.fit(weightedDataset)


# evaluate traingData performance
train_predictions = model.transform(train_df)
train_predictions.show()

# testing data
test_df = read("FreddieMac_Spark.PySpark_TestingData")
predictions = model.transform(test_df)

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
'''
Accuracy = 0.697945
AUC = 0.72
'''

# confusion matrix
cm_lr_result = predictions.crosstab("predictedLabel", "label")
cm_lr_result = cm_lr_result.toPandas()
cm_lr_result

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

'''
Accuracy = 0.70
Recall = 0.62
Specificity = 0.70
Precision = 0.13
F1 score = 0.22
'''

# lr model ROC curve
import matplotlib.pyplot as plt

trainingSummary = model.stages[0].summary
roc = trainingSummary.roc.toPandas()
plt.plot(roc['FPR'],roc['TPR'])
plt.ylabel('False Positive Rate')
plt.xlabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
print('Training set areaUnderROC: ' + str(trainingSummary.areaUnderROC))
