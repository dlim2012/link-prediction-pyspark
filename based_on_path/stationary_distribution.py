import csv
import os

import numpy as np
import pyspark.sql.functions as F
from pyspark import SparkConf, SparkContext
from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix
from pyspark.sql import SparkSession
from pyspark.sql import Window
from tqdm import tqdm

filename = "email-Eu-core-temporal-Dept3.txt"
# filename = "email-Eu-core-temporal.txt"
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


# u, v
edges = train_df\
  .drop("t")\
  .groupby("u", "v")\
  .count()

nodes = train_df\
  .select("u")\
  .union(edges.select("v"))\
  .withColumnRenamed("u", "node")\
  .distinct().orderBy("node")

# nodes.write.mode("ignore").csv(os.path.join(save_dir, f"{filename.split('.')[0]}_nodes.csv"))
num_nodes = nodes.count()

# _u, _v
_adj = nodes.withColumnRenamed("node", "_u")\
  .join(nodes.withColumnRenamed("node", "_v"))\
  .orderBy("_u", "_v")

# __u, total
_total = edges\
  .groupBy("u")\
  .agg(F.sum(edges["count"]))\
  .withColumnRenamed("sum(count)", "total")\
  .withColumnRenamed("u", "__u")

"""
T = [
[0, 1, 0]
[0.5, 0, 0.5]
[0, 0, 0]
]

"""

# u, value
_edges = edges\
  .join(_total, edges.u == _total.__u)\
  .withColumn("value", edges["count"] / (_total["total"]))\
  .drop("count")\
  .drop("total")\
  .drop("__u")

print("_edges", _edges.toPandas())

# u, v, value
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

print("A")
print(A.toLocalMatrix().toArray())

n = 100
_n = int(n*0.9)
for i in tqdm(range(n), total=n):
  matrix = matrix.multiply(A)
  # print(matrix.toLocalMatrix().toArray())
  if i >= _n:
      stationary_distribution = stationary_distribution.add(matrix)


np.save(os.path.join(save_dir, f"{filename.split('.')[0]}_stationary-distribution.npy"),
        stationary_distribution.toLocalMatrix().toArray() / (n - _n))