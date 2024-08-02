import random
import sys 
import numpy as np
import pandas as pd
import quinn
import findspark
findspark.init('/opt/spark/spark-3.5.1-bin-hadoop3')

#from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.sql.functions import col, desc
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, Normalizer, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression

#SparkSession
spark = SparkSession \
    .builder \
    .appName("CS643_Wine_Quality_Predictions") \
    .getOrCreate()

 #conf = pyspark.SparkConf().setAppName('winequality').setMaster('local')
 #sc = pyspark.SparkContext(conf=conf)
 #spark = SparkSession(sc)

## Load Training Dataset. Pull data, make header and 'inferSchema' so column has integer values(or appropirate values)
train_df = spark.read.format('csv').options(header='true', inferSchema='true', sep=';').load('/home/ubuntu/CS643/tavarisTrainingDataset.csv')
validation_df = spark.read.format('csv').options(header='true', inferSchema='true', sep=';').load('/home/ubuntu/CS643/tavarisValidationDataset.csv')


print("Original Data loaded from local directory on Master EC2 Instance.")
print(train_df.toPandas().head(2))
print(validation_df.toPandas().head(2))


#gathering all columns
numerical_features_lstTr = train_df.columns
numerical_features_lstTr.remove('quality')

#gathering all columns
numerical_features_lstVal = validation_df.columns
numerical_features_lstVal.remove('quality')

#show example of column in list
numerical_features_lstTr



numerical_vector_assembler = VectorAssembler(inputCols=numerical_features_lstTr,
                                       outputCol='numerical_feature_vector')

train_df = numerical_vector_assembler.transform(train_df)
validation_df = numerical_vector_assembler.transform(validation_df)


#Showing all values represented in a single column to be used for processing
train_df.select('numerical_feature_vector').take(2)


scaler = StandardScaler(inputCol='numerical_feature_vector',
                        outputCol='scaled_numerical_feature_vector',
                        withStd=True, withMean=True)

scaler = scaler.fit(train_df)

train_df = scaler.transform(train_df)
validation_df = scaler.transform(validation_df)

#Showing all rows and addition
train_df.show(3)


#Showing all values represented in a single column to be used for processing
train_df.select('scaled_numerical_feature_vector').take(3)


#Selecting the column for which the prediction must happen against
lr = LinearRegression(featuresCol='scaled_numerical_feature_vector',
                      labelCol='quality')

lr = lr.fit(train_df)
lr


pred_train_df = lr.transform(train_df).withColumnRenamed('prediction',
                                                         'predicted_wine_quality')

#Shows ALL rows and columns in the training set that were added
pred_train_df.show(5)


print("Showing 5 rows for F1 Score for LogisticRegression Model Only!")

#Showing values represented in a "single" column
pred_train_df.select('quality','predicted_wine_quality').take(10)



