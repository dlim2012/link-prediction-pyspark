import numpy as np
import pandas as pd

hitting_time_filename = "results/email-Eu-core-temporal-Dept3_hitting-time.npy"
stationary_dist_filename = "results/email-Eu-core-temporal-Dept3_stationary-distribution.npy"
csv_filename = "results/email-Eu-core-temporal-Dept3_nodes_hitting-time.csv"

ht = np.load(hitting_time_filename)
sd = np.load(stationary_dist_filename)
nodes = pd.read_csv(csv_filename)



print(ht.shape)
print(sd.shape)
print(nodes)