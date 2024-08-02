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
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression

#SparkSession
spark = SparkSession \
    .builder \
    .appName("CS643_Wine_Quality_Predictions") \
    .getOrCreate()


## Load Training Dataset. Pull data, make header and 'inferSchema' so column has integer values(or appropirate values)
train_df = spark.read.format('csv').options(header='true', inferSchema='true', sep=';').load('/home/ubuntu/CS643-AWS-ProgAssgn-2/tavarisTrainingDataset.csv')
validation_df = spark.read.format('csv').options(header='true', inferSchema='true', sep=';').load('/home/ubuntu/CS643-AWS-ProgAssgn-2/tavarisValidationDataset.csv')

#Used at the end for Second set of Prediction values
train_RandForest = spark.read.format('csv').options(header='true', inferSchema='true', sep=';').load('/home/ubuntu/CS643-AWS-ProgAssgn-2/tavarisTrainingDataset.csv')
validation_RandForest = spark.read.format('csv').options(header='true', inferSchema='true', sep=';').load('/home/ubuntu/CS643-AWS-ProgAssgn-2/tavarisValidationDataset.csv')


print("Original Data loaded from local directory on Master EC2 Instance.")
print(train_df.toPandas().head(2))
print(validation_df.toPandas().head(2))


#gathering all columns-LinearRegression
numerical_features_lstTr = train_df.columns
numerical_features_lstTr.remove('quality')

#gathering all columns-LinearRegression
numerical_features_lstVal = validation_df.columns
numerical_features_lstVal.remove('quality')

#show example of column in list
numerical_features_lstTr



#gathering all columns-Random Forest
numerical_features_lstTrRandom = train_RandForest.columns
numerical_features_lstTrRandom.remove('quality')

#gathering all columns-Random Forest
numerical_features_lstValRandom = validation_RandForest.columns
numerical_features_lstValRandom.remove('quality')


#Setup for LinearRegression
numerical_vector_assembler = VectorAssembler(inputCols=numerical_features_lstTr,
                                       outputCol='numerical_feature_vector')
#Used for 1st set of Prediction values
train_df = numerical_vector_assembler.transform(train_df)
validation_df = numerical_vector_assembler.transform(validation_df)




#Setup for RandomForest
numerical_vector_assemblerRandom = VectorAssembler(inputCols=numerical_features_lstTrRandom,
                                       outputCol='numerical_feature_vectorRandom')
#Used for 2nd set of Prediction values
train_RandForest = numerical_vector_assemblerRandom.transform(train_RandForest)
validation_RandForest = numerical_vector_assemblerRandom.transform(validation_RandForest)



#1st Prediction Starting Point LinearRegression
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
print("Showing 5 rows for F1 Score for LogisticRegression Model #1")
pred_train_df.show(5)



print("Random Forest Model #2")
#2nd Prediction Starting Point LinearRegression
rf_classifer = RandomForestClassifier(featuresCol='numerical_feature_vectorRandom',
                                    labelCol='quality',
                                     numTrees=50).fit(train_RandForest)

rf_predictions = rf_classifer.transform(validation_RandForest)
rf_predictions.show(5)


print("Prediction1 - F1 Score Linear Regression is more accurate")
pred_train_df.select('quality','predicted_wine_quality').take(20)



