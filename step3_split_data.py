# training data
df = read("FreddieMac_Spark.PySpark_Integrated")
train_df,test_df = df.randomSplit([0.7, 0.3])
save(train_df)

# testing data
df = read("FreddieMac_Spark.PySpark_Integrated")
df_train = read("FreddieMac_Spark.PySpark_TrainingData")
df_test = df.exceptAll(df_train)
save(df_test)