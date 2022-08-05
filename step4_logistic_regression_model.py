from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer, StringIndexerModel
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression

# read dataframe
train_df = read("FreddieMac_Spark.PySpark_TrainingData")

# load label indexer
labelIndexer = StringIndexerModel.load("/incorta/IncortaAnalytics/Tenants/ebs_cloud/incorta.ml/models"+ "/string-indexer-model") 
lr = LogisticRegression(featuresCol='indexedFeatures', labelCol='indexedLabel')
# Convert indexed labels back to original labels.
labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                               labels=labelIndexer.labels)
pipeline = Pipeline(stages=[lr, labelConverter])


# Train model.  This also runs the indexers.
model = pipeline.fit(train_df)
# save logr pipeline model
model.write().overwrite().save("/incorta/IncortaAnalytics/Tenants/ebs_cloud/incorta.ml/models"+ "/logr-pipeline-model")

# evaluate traingData performance
train_predictions = model.transform(train_df)

# save prediction result
df_output = train_predictions.select("seq_num", "label", "predictedLabel", "probability")
df_output.show(5)

save(df_output)