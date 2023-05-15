import csv
import os

import numpy as np
import pyspark.sql.functions as F
from pyspark import SparkConf, SparkContext
from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix
from pyspark.sql import SparkSession
from pyspark.sql import Window
from tqdm import tqdm

# filename = "email-Eu-core-temporal-Dept3.txt"
filename = "email-Eu-core-temporal.txt"
save_dir = "results"
os.makedirs(save_dir, exist_ok=True)

conf = SparkConf().setMaster("local").setAppName("My App")
sc = SparkContext(conf = conf)

# create a spark session
spark = SparkSession.builder.appName('katz').getOrCreate()

# read data as list
with open(filename) as csvfile:
    data = [list(map(int, row)) for row in csv.reader(csvfile, delimiter=' ')]

# make a spark dataframe
columns = ["u", "v", "t"]
df = spark.createDataFrame(data=data, schema=columns)

df.toPandas()

train_df = df.filter(F.col("t") < 20_000_000)
test_df = df.filter(F.col("t") >= 20_000_000).filter(F.col("t") < 22_000_000)

train_df.toPandas(), test_df.toPandas()


edges = train_df\
  .drop("t")\
  .groupby("u", "v")\
  .count()

maxCount = edges\
  .select(
      F.max(edges["count"]).alias("maxCount")
  ).collect()[0]["maxCount"]

print("edges")
print(edges.orderBy("u", "v").toPandas())

nodes = train_df\
  .select("u")\
  .intersect(edges.select("v"))\
  .withColumnRenamed("u", "node")\
  .orderBy("node")


nodes.write.mode("ignore").csv(os.path.join(save_dir, f"{filename.split('.')[0]}_nodes.csv"))

num_nodes = nodes.count()
print("nodes")
print(nodes.toPandas())


_adj = nodes.withColumnRenamed("node", "_u")\
  .join(nodes.withColumnRenamed("node", "_v"))\
  .orderBy("_u", "_v")


_edges = edges\
  .withColumn("value", edges["count"])\
  .drop("count")

adj = _adj\
  .join(
    _edges,
    (_adj._u == _edges.u)
      & (_adj._v == _edges.v),
    'left'
  )\
  .select("_u", "_v", "value")\
  .fillna(0)\
  .withColumnRenamed("_u", "u")\
  .withColumnRenamed("_v", "v")

print("adj")
print(adj.orderBy("u", "v").toPandas())

w = Window.partitionBy("u").orderBy("v")
_A = adj\
  .withColumn("sorted_list", F.collect_list("value").over(w))\
  .groupBy("u")\
  .agg(F.max("sorted_list").alias('row'))\
  .orderBy('u')\
  .withColumn("id", F.monotonically_increasing_id())

print("A")
print(_A.toPandas())


A = IndexedRowMatrix(_A.select("id", "row").rdd.map(lambda row: IndexedRow(*row)))\
  .toBlockMatrix(num_nodes, num_nodes)
matrix = IndexedRowMatrix(_A.select("id", "row").rdd.map(lambda row: IndexedRow(*row)))\
  .toBlockMatrix(num_nodes, num_nodes)
stationary_distribution = IndexedRowMatrix(sc.parallelize([IndexedRow(_, [0] * num_nodes) for _ in range(num_nodes)]))\
  .toBlockMatrix(num_nodes, num_nodes)
A.toLocalMatrix().toArray()

print(A.toLocalMatrix().toArray())

n = 100
_n = int(n*0.9)
for i in tqdm(range(n), total=n):
  matrix = matrix.multiply(A)
  if i >= _n:
      stationary_distribution = stationary_distribution.add(matrix)

print(stationary_distribution.toLocalMatrix().toArray())

np.save(os.path.join(save_dir, f"{filename.split('.')[0]}_stationary-distribution.npy"),
        stationary_distribution.toLocalMatrix().toArray() / (n - _n))
