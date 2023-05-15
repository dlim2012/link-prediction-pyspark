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


_katz_adj = nodes.withColumnRenamed("node", "_u")\
  .join(nodes.withColumnRenamed("node", "_v"))\
  .orderBy("_u", "_v")

beta = 1 / (1.5 * maxCount)

katz_edges = edges\
  .withColumn("value", edges["count"] * beta)\
  .drop("count")

katz_adj = _katz_adj\
  .join(
    katz_edges,
    (_katz_adj._u == katz_edges.u)
      & (_katz_adj._v == katz_edges.v),
    'left'
  )\
  .select("_u", "_v", "value")\
  .fillna(0)\
  .withColumnRenamed("_u", "u")\
  .withColumnRenamed("_v", "v")

print("katz_adj")
print(katz_adj.orderBy("u", "v").toPandas())

katz_w = Window.partitionBy("u").orderBy("v")
_katz_A = katz_adj\
  .withColumn("sorted_list", F.collect_list("value").over(katz_w))\
  .groupBy("u")\
  .agg(F.max("sorted_list").alias('row'))\
  .orderBy('u')\
  .withColumn("id", F.monotonically_increasing_id())

print("_katz_A")
print(_katz_A.toPandas())


katz_A = IndexedRowMatrix(_katz_A.select("id", "row").rdd.map(lambda row: IndexedRow(*row)))\
  .toBlockMatrix(num_nodes, num_nodes)
katz_matrix = IndexedRowMatrix(_katz_A.select("id", "row").rdd.map(lambda row: IndexedRow(*row)))\
  .toBlockMatrix(num_nodes, num_nodes)
katz_scores = IndexedRowMatrix(sc.parallelize([IndexedRow(_, [0] * num_nodes) for _ in range(num_nodes)]))\
  .toBlockMatrix(num_nodes, num_nodes)
katz_A.toLocalMatrix().toArray()

n = 100
for i in tqdm(range(n), total=n):
  katz_scores = katz_scores.add(katz_matrix)
  katz_matrix = katz_matrix.multiply(katz_A)
print(katz_scores.toLocalMatrix().toArray())

np.save(os.path.join(save_dir, f"{filename.split('.')[0]}_katz-scores.npy"), katz_scores)