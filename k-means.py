#MIMI DO - 002263586
import pyspark
from pyspark.context import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.clustering import BisectingKMeans

# Creating a Spark Context
conf = SparkConf()
sc = SparkContext(conf = conf)
sc.setLogLevel("ERROR")


#creating spark session
spark = SparkSession \
    .builder \
    .appName("K-mean") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

# Loads data.
dataset = spark.read.format("libsvm").load("/home/mdo8/Assignment6/kmeans_input.txt")

#printing parsed data
dataset.show(200)

#printing the schema
dataset.printSchema()

#training bisecting k-means model
#set k(#clusters) to 2 and sets seed to 1
bkm = BisectingKMeans().setK(2).setSeed(1)
#this fits a model to the input dataset which we loaded in the beginning
model = bkm.fit(dataset)


#calculating the Squared errors and evaluating clustering
cost = model.computeCost(dataset)
#printing results
print("Within Set Sum of Squared Errors = " + str(cost))

# Shows the result.
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)