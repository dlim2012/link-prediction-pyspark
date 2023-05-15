import csv
import os

import numpy as np
import pyspark.sql.functions as F
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from tqdm import tqdm

# filename = "email-Eu-core-temporal-Dept3.txt"
filename = "email-Eu-core-temporal.txt"
save_dir = "results"
os.makedirs(save_dir, exist_ok=True)

conf = SparkConf().setMaster("local").setAppName("My App")
sc = SparkContext(conf = conf)

# create a spark session

SparkContext.setSystemProperty('spark.executor.memory', '100g')
spark = SparkSession.builder.appName('shortest-path')\
  .config("spark.memory.fraction", 0.8) \
  .config("spark.executor.memory", "100g") \
  .config('spark.cores.max', "8") \
  .config("spark.driver.memory", "100g")\
  .config("spark.sql.shuffle.partitions" , "800") \
    .config("spark.driver.maxResultSize", "100g") \
    .getOrCreate()
# spark = SparkSession.builder.appName('shortest-path').getOrCreate()

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

result = np.zeros((num_nodes, num_nodes))

for j, row in tqdm(enumerate(nodes.collect()), total=num_nodes):
  node = row["node"]
  shortest_length = initial_shortest_length = nodes\
    .withColumn("shortest_length", F.when(nodes["node"] != node, F.lit(10000000)).otherwise(0))\
    .orderBy("node")
  for i in tqdm(range(shortest_length.count()), total=shortest_length.count()):
    temp = shortest_length\
      .join(edges.select("u", "v"), shortest_length["node"] == edges["u"])\
      .groupBy("v")\
      .agg(F.min("shortest_length").alias("temp"))\
      .select("v", "temp")\
      .withColumnRenamed("v", "_node") # join(-).groupBy(-).aggregate(min)
    shortest_length = shortest_length\
          .join(temp, shortest_length["node"] == temp["_node"])\
          .withColumn(
              "new_shortest_length",
              F.when(shortest_length["shortest_length"] < temp["temp"] + 1, shortest_length["shortest_length"]).otherwise(temp["temp"] + 1)
          )\
          .select("node", "new_shortest_length")\
          .withColumnRenamed("new_shortest_length", "shortest_length")
    temp.collect()
    shortest_length = spark.createDataFrame(shortest_length.collect())
  print(node)
  print(shortest_length.orderBy("node").select("shortest_length").toPandas())
  result[j] = np.array(shortest_length.orderBy("node").select("shortest_length").collect()).squeeze(1)
  print((j, node), result[j])



np.save(os.path.join(save_dir, f"{filename.split('.')[0]}_shortest-length.npy"), result)

