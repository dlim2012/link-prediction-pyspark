import numpy as np
import pandas as pd

filename = "results/email-Eu-core-temporal-Dept2_shortest-length.npy"
nodes = pd.read_csv("results/email-Eu-core-temporal-Dept2_nodes_shortest_length.csv").values[:, -1]
s = np.load(filename)

print(s.shape
      )

print(np.sum(s > 10000) / 132 / 132)
rows = []
for i, node in enumerate(nodes):
    for j, node2 in enumerate(nodes):
        rows.append([node, node2, s[i][j]])

df = pd.DataFrame(rows, columns=["u", "v", "score"])
