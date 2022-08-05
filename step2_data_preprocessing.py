# read dataset
df_orig = read("FreddieMac.sample_orig_files")
df_svcg = read("FreddieMac.sample_svcg_files")

# origination data
orig_col = ['credit_score', 'first_payment_date', 'first_time_buyer', 'maturity_date', 'msa_code',
                        'mi_percent', 'unit_ct', 'occupancy_status', 'comb_loan_to_value', 'debt_to_income',
                        'org_upb', 'loan_to_value', 'org_roi', 'channel', 'ppm',
                        'rate_type', 'state', 'prop_type', 'pincode', 'seq_num',
                        'loan_purpose', 'org_term', 'num_borrowers', 'seller_name', 'servicer_name',
                        'sup_conforming', 'pre_harp_seq_num', 'program_indicator', 'harp_indicator', 'valuation_method',
                        'io_indicator']
# performance data
svcg_col = ['seq_num', 'reporting_period', 'cur_upb', 'delinquency_status', 'loan_age',
                      'months_to_maturity', 'repurchased', 'modified', 'zero_bal_code', 'zero_bal_date',
                      'cur_roi', 'cur_def_upb', 'last_due_date', 'mi_recovery', 'net_sales_profit',
                      'non_mi_recovery', 'expenses', 'legal_cost', 'maintenance_cost', 'tax_insurance',
                      'misc_expenses', 'act_loss', 'modification_cost', 'step_modification', 'def_payment_plan',
                      'est_loan_to_value', 'zero_bal_removal_upb', 'delinquent_interest', 'delinquency_due_disaster',
                      'borrower_assistance_status','curr_modification_cost','int_bearing_upb']
# rename the columns
df_orig = df_orig.toDF(*orig_col)
df_svcg = df_svcg.toDF(*svcg_col)

##### data profile
from incorta_dataprep.eda import summary_table
### origination data
df_orig_summary = summary_table(df_orig, table_name='orig_table') # may take about 5 minutes
incorta.show(df_orig_summary)

## count missing value for each column. We can either use method 1 or 2 to count
# method 1
df_orig_summary.select(col("column_name"),col("missing_count"),col("missing_pct")).show() # may take about 1 minute

# method 2
from pyspark.sql.functions import col,isnan, when, count
df_miss = df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns])
incorta.show(df_miss)

## distinct value for each column
df_orig_summary.select(col("column_name"),col("unique_count"),col("unique_pct")).show()

### performance data
df_svcg_summary = summary_table(df_svcg, table_name='svcg_table') 
incorta.show(df_svcg_summary)


import numpy as np
### origination data
## whole empty columns
orig_unwanted = (np.array([26,27,29])-1).tolist()
orig_del_name = []
for i in orig_unwanted:
    orig_del_name.append(orig_col[i])

## low variance columns
# of unique count of rate_type & io_indicator = 1
# of unique count of ppm = 2

## not suitable for model training
# msa_code has 23% of missing value and 437 unique counts, not suitable for encoding
orig_del_name = orig_del_name + ['msa_code','ppm','rate_type','io_indicator']  
  
### performance data
## whole empty columns
svcg_unwanted = (np.array([8,23,24,25,26,29,30,31])-1).tolist()
svcg_del_name = []
for i in svcg_unwanted:
    svcg_del_name.append(svcg_col[i])

## drop columns
df_orig = df_orig.drop(*orig_del_name)
df_svcg = df_svcg.drop(*svcg_del_name)
### clean origination data
from pyspark.sql.functions import *
def clean_orig(df):
    df = df.withColumn("credit_score", when((df.credit_score>300) & (df.credit_score<851), df.credit_score).otherwise(-1))
    # deal with unavailable value
    # unavailable: 9 or 99 or 999 -> -1 
    df = df.withColumn("mi_percent", when(df.mi_percent==999, -1).otherwise(df.mi_percent))
    df = df.withColumn("unit_ct", when(df.unit_ct==99, -1).otherwise(df.unit_ct))
    df = df.withColumn("comb_loan_to_value", when(df.comb_loan_to_value==999, -1).otherwise(df.comb_loan_to_value))
    df = df.withColumn("debt_to_income", when(df.debt_to_income==999, -1).otherwise(df.debt_to_income))
    df = df.withColumn("loan_to_value", when(df.loan_to_value==999, -1).otherwise(df.loan_to_value))
    df = df.withColumn("num_borrowers", when(df.num_borrowers==99, -1).otherwise(df.num_borrowers))
    df = df.withColumn("valuation_method", when(df.valuation_method==9, -1).otherwise(df.valuation_method))
    
    # deal with datetime
    from pyspark.sql.functions import year, month
    from pyspark.sql.functions import to_date
    df = df.withColumn('first_payment_year', year(df.first_payment_date))
    df = df.withColumn('first_payment_month', month(df.first_payment_date))
    df = df.withColumn('maturity_year', year(df.maturity_date))
    df = df.withColumn('maturity_month', month(df.maturity_date))
    return df.drop('first_payment_date','maturity_date')

### clean performance data
def clean_perf(df):
    df = df.withColumn("net_sales_profit", when(df.net_sales_profit=='U', -1).otherwise(df.net_sales_profit))
    return df
# ### Transform label to numerical
from pyspark.sql.functions import when

def update_label(df):
    df3 = df.withColumn("delinquency_status", 
                        when((df.delinquency_status.isin(['0', '1', '2', 'R'])) &(df.zero_bal_code.isin([3, 6, 9])),1) \
            .when((df.delinquency_status.isin(['0', '1', '2', 'R']))&(~df.zero_bal_code.isin([3, 6, 9])),0) \
            .otherwise(1))
    return df3
# # Feature Engineering
# ### Aggregate Performance Features  
from pyspark.sql import functions as F
def aggregate_features(df):
    df_svcg_gp = df_svcg.groupBy('seq_num').agg(
        F.mean('cur_upb').alias('mean_cur_upb'),
        F.mean('cur_def_upb').alias('mean_cur_def_upb'),
        F.mean('mi_recovery').alias('mean_mi_recovery'),
        F.mean('net_sales_profit').alias('mean_net_sales_profit'),
        F.mean('non_mi_recovery').alias('mean_non_mi_recovery'),
        F.mean('expenses').alias('mean_expenses'),
        F.mean('legal_cost').alias('mean_legal_cost'),
        F.mean('maintenance_cost').alias('mean_maintenance_cost'),
        F.mean('tax_insurance').alias('mean_tax_insurance'),
        F.mean('misc_expenses').alias('mean_misc_expenses'),
        F.mean('act_loss').alias('mean_act_loss'),
        F.mean('zero_bal_removal_upb').alias('mean_zero_bal_removal_upb'),
        F.mean('delinquent_interest').alias('mean_delinquent_interest'),
        F.min('months_to_maturity').alias('min_months_to_maturity'),
        F.max('loan_age').alias('max_loan_age'),
        F.max('delinquency_status').alias('max_delinquency_status'),
        F.max('zero_bal_code').alias('max_zero_bal_code'),  
    )
    
    return df_svcg_gp
from pyspark.sql.types import *
def get_categoricalCols(df):
    # get string
    str_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, StringType)]
    str_cols.remove("seq_num")
    str_cols.remove("max_delinquency_status")
    str_cols.remove("seller_name")
    str_cols.remove("servicer_name")
    return str_cols
def get_continuousCols(df):
    # get double and long
    dbl_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, DoubleType)]
    long_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, LongType)]
    continu_cols = dbl_cols + long_cols
    return continu_cols
def get_dummy(df,categoricalCols, specialCols, continuousCols, labelCol):
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler
    from pyspark.sql.functions import col
    
    # first: deal with categoricalCols
    # string indexer
    indexers = [ StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c)).setHandleInvalid("keep") for c in categoricalCols]
    
    # one hot encoding
    # default setting: dropLast=True
    encoder = OneHotEncoderEstimator(inputCols=[indexer.getOutputCol() for indexer in indexers], 
                                        outputCols=["{0}_encoded".format(indexer.getOutputCol()) for indexer in indexers]).setHandleInvalid("keep")
    
    # second: string indexer for special columns
    indexers2 = [ StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c)).setHandleInvalid("keep") for c in specialCols]
    
    # vectorization
    assembler = VectorAssembler(inputCols=encoder.getOutputCols() + [indexer.getOutputCol() for indexer in indexers2] + continuousCols, 
                                outputCol="features")
    pipeline = Pipeline(stages=indexers + indexers2 + [encoder, assembler])
    model=pipeline.fit(df)

    # save prep-model
    temp_path = "/incorta/IncortaAnalytics/Tenants/ebs_cloud/incorta.ml/models"
    modelPath = temp_path + "/prep-model"
    model.write().overwrite().save(modelPath)
    
    data = model.transform(df)
    data = data.withColumn('label', col(labelCol))
    return data.select('seq_num','features','label')
### start the flow
orig = clean_orig(df_orig) 
perf = update_label(clean_perf(df_svcg))
perf = aggregate_features(perf) 
perf = perf.withColumn("max_delinquency_status", when(perf.max_delinquency_status == 1,"delinquent") \
      .otherwise("not_delinquent"))
# drop 97% misssing and similar columns 
perf2 = perf.drop('mean_mi_recovery', 'mean_net_sales_profit', 'mean_non_mi_recovery', 'mean_expenses', 'mean_legal_cost', 'mean_maintenance_cost', 'mean_tax_insurance', 'mean_misc_expenses', 'mean_act_loss','mean_zero_bal_removal_upb', 'mean_delinquent_interest')
# fill null value in max_zero_bal_code with mode
bal_code_mode = perf2.groupby("max_zero_bal_code").count().orderBy("count", ascending=False).first()[0]
perf3 = perf2.fillna(bal_code_mode)
# join data frame
df_join2 = orig.join(perf3,(orig['seq_num']==perf3['seq_num']),'right').drop(orig['seq_num'])

def get_dummy(df,categoricalCols, specialCols, continuousCols, labelCol):
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler
    from pyspark.sql.functions import col
    
    # first: deal with categoricalCols
    # string indexer
    indexers = [ StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c)).setHandleInvalid("keep") for c in categoricalCols]
    
    # one hot encoding
    # default setting: dropLast=True
    encoder = OneHotEncoderEstimator(inputCols=[indexer.getOutputCol() for indexer in indexers], 
                                        outputCols=["{0}_encoded".format(indexer.getOutputCol()) for indexer in indexers]).setHandleInvalid("keep")
    
    # second: string indexer for special columns
    indexers2 = [ StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c)).setHandleInvalid("keep") for c in specialCols]
    
    # vectorization
    assembler = VectorAssembler(inputCols=encoder.getOutputCols() + [indexer.getOutputCol() for indexer in indexers2] + continuousCols, 
                                outputCol="features")
    pipeline = Pipeline(stages=indexers + indexers2 + [encoder, assembler])
    model=pipeline.fit(df)

    # save prep-model
    temp_path = "/incorta/IncortaAnalytics/Tenants/ebs_cloud/incorta.ml/models"
    modelPath = temp_path + "/prep-model"
    model.write().overwrite().save(modelPath)
    
    data = model.transform(df)
    data = data.withColumn('label', col(labelCol))
    return data.select('seq_num','features','label')

data1 = get_dummy(df_join2,get_categoricalCols(df_join2),["seller_name","servicer_name"],get_continuousCols(df_join2),'max_delinquency_status') # can work

### Deal with Categorical Label
from pyspark.ml.feature import StringIndexer, IndexToString
labelIndexer = StringIndexer(inputCol='label',
                             outputCol='indexedLabel').setHandleInvalid("skip").fit(data1)
temp_path = "/incorta/IncortaAnalytics/Tenants/ebs_cloud/incorta.ml/models"
modelPath = temp_path + "/string-indexer-model"
labelIndexer.write().overwrite().save(modelPath) 

### Deal with Categorical Features
from pyspark.ml.feature import VectorIndexer
featureIndexer = VectorIndexer(inputCol="features", \
                                  outputCol="indexedFeatures", \
                                  maxCategories=4)
# Chain indexers in a Pipeline
from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer])
# Train model.  This also runs the indexers.
model = pipeline.fit(data1)
data2 = model.transform(data1)

data2.show(5)
df_output = data2.select("seq_num", "features", "indexedFeatures", "label", "indexedLabel")
save(df_output)