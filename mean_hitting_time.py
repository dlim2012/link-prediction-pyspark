
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
SparkContext.setSystemProperty('spark.executor.memory', '10g')
spark = SparkSession.builder.appName('mean-hitting-time')\
  .config("spark.memory.fraction", 0.8) \
  .config("spark.executor.memory", "10g") \
  .config('spark.cores.max', "20") \
  .config("spark.driver.memory", "10g")\
  .config("spark.sql.shuffle.partitions" , "10") \
    .getOrCreate()
# spark = SparkSession.builder.appName('mean-hitting-time').getOrCreate()

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

counts = edges\
  .groupBy("u")\
  .sum("count")\
  .withColumnRenamed("sum(count)", "total")\
  .orderBy("u")

print("counts")
print(counts.toPandas())

_hitting_adj = nodes.withColumnRenamed("node", "_u")\
  .join(nodes.withColumnRenamed("node", "_v"))

hitting_adj = _hitting_adj\
  .join(
    edges,
    (_hitting_adj._u == edges.u)
      & (_hitting_adj._v == edges.v),
    'left'
  )\
  .select("_u", "_v", "count")\
  .fillna(0)

hitting_adj = hitting_adj\
  .join(
      counts,
      counts.u == hitting_adj._u,
      'left'
  )\
  .fillna(0)\
  .withColumn("prob", F.expr("count / (total + 1e-15)"))\
  .select("_u", "_v", "prob")\
  .withColumnRenamed("_u", "u")\
  .withColumnRenamed("_v", "v")

print("hitting_adj")
print(hitting_adj.orderBy("u", "v").toPandas())



result = np.zeros((num_nodes, num_nodes))


n = 100
w = Window.partitionBy('u').orderBy('v')
for j, row in tqdm(enumerate(nodes.collect()), total=num_nodes):

    node = row["node"]
    # print(j, node)

    hitting_T_df = hitting_adj \
        .withColumn(
        "_prob",
        F.when(hitting_adj["u"] != node, hitting_adj["prob"]) \
            .otherwise(F.when(hitting_adj["v"] != node, 0.0) \
                       .otherwise(1.0) \
                       ) \
        )

    hitting_T_df = hitting_T_df \
        .withColumn("sorted_list", F.collect_list("_prob").over(w)) \
        .groupBy("u") \
        .agg(F.max("sorted_list").alias("row")) \
        .orderBy("u") \
        .withColumn("id", F.monotonically_increasing_id())

    hitting_T = IndexedRowMatrix(hitting_T_df.select("id", "row").rdd.map(lambda row: IndexedRow(*row))) \
        .toBlockMatrix(num_nodes, num_nodes)
    hitting_matrix = IndexedRowMatrix(hitting_T_df.select("id", "row").rdd.map(lambda row: IndexedRow(*row))) \
        .toBlockMatrix(num_nodes, num_nodes)
    hitting_scores = IndexedRowMatrix(sc.parallelize([IndexedRow(_, [0] * num_nodes) for _ in range(num_nodes)], numSlices=100)) \
        .toBlockMatrix(num_nodes, num_nodes)

    # print(hitting_T.toLocalMatrix().toArray().sum(axis=1))
    for i in range(n):
        hitting_scores = hitting_scores.add(hitting_matrix)
        hitting_matrix = hitting_matrix.multiply(hitting_T)

    # print(hitting_matrix.toLocalMatrix().toArray()[:, j])
    scores = n - hitting_scores.toLocalMatrix().toArray()[:, j]
    np.save(f"{filename.split('.')[0]}_hitting-time_{j}.npy", scores)
    # print(scores)
    result[j] = scores

    np.save(os.path.join(save_dir, f"{filename.split('.')[0]}_hitting-time.npy"), result)
