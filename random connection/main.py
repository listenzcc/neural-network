"""
File: main.py
Author: Chuncheng Zhang
Date: 2024-11-08
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Neural network demo for random connection.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2024-11-08 ------------------------
# Requirements and constants
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from rich import print
from tqdm.auto import tqdm
from collections import defaultdict
from scipy.spatial import Delaunay


# %% ---- 2024-11-08 ------------------------
# Function and class
def sigmoid(x):
    return np.exp(x) / (1+np.exp(x))


class MyGraph(object):
    num_points = 200
    values_buf_size = 10000

    def __init__(self):
        self.generate_points_tri()
        self.find_neighbors()
        self.generate_values_array()
        self.initialize_weights()

    def initialize_weights(self):
        weights = self.normalized_distances
        self.weights = weights
        return weights

    def extending(self):
        values = np.zeros((self.num_points,))
        values[0] = self.values[0, self.i]
        for j in range(1, self.num_points):
            for k, w in self.weights[j].items():
                values[j] += self.values[k, self.i] * w

        self.values[:, self.i+1] = sigmoid(values)
        # self.values[:, self.i+1] = values

        self.step()

        return

    def step(self):
        self.i += 1
        self.i %= self.values_buf_size-1

    def set_node_value(self, node_idx: int = 0):
        t = self.times[self.i]
        omega = 5
        value = (np.sin(t * omega * np.pi * 2) + 1)*0.5
        self.values[node_idx, self.i] = value

    def generate_values_array(self):
        fs = 1000  # Hz
        i = 0
        values = np.zeros((self.num_points, self.values_buf_size))
        self.i = i
        self.values = values
        self.times = np.linspace(
            0, self.values_buf_size/fs, self.values_buf_size, endpoint=False)
        return self.values

    def generate_points_tri(self):
        points = np.random.randn(self.num_points, 2)
        tri = Delaunay(points)
        self.points = points
        self.tri = tri
        return points, tri

    def find_neighbors(self):
        # ----------------------------------------
        # ---- Find neighbors ----
        neighbors = defaultdict(set)
        for tri in tqdm(self.tri.simplices, 'Building neighbors'):
            neighbors[int(tri[0])].add(int(tri[1]))
            neighbors[int(tri[0])].add(int(tri[2]))
            neighbors[int(tri[1])].add(int(tri[0]))
            neighbors[int(tri[1])].add(int(tri[2]))
            neighbors[int(tri[2])].add(int(tri[0]))
            neighbors[int(tri[2])].add(int(tri[1]))

        # ----------------------------------------
        # ---- Compute distances of edges ----
        distances = defaultdict(dict)
        for k, nodes in neighbors.items():
            for n in nodes:
                distances[k][n] = float(np.linalg.norm(
                    self.points[k] - self.points[n]))

        # ----------------------------------------
        # ---- Normalize the distance of each node ----
        normalized_distances = defaultdict(dict)
        for k, nodes in distances.items():
            den = sum(v for v in nodes.values())
            for n in nodes:
                normalized_distances[k][n] = nodes[n] / den

        self.neighbors = neighbors
        self.distances = distances
        self.normalized_distances = normalized_distances
        return neighbors, distances, normalized_distances

    def plot_points_tri(self, scatterplot_kwargs={}, nodes_idx=[]):
        points = self.points
        tri = self.tri
        fig, ax = plt.subplots(1, 1)

        ax.triplot(points[:, 0], points[:, 1], tri.simplices, color='white')

        sns.scatterplot(x=points[:, 0], y=points[:, 1],
                        palette='RdBu_r',
                        ax=ax, **scatterplot_kwargs)

        # Not draw the text other than 0 idx
        dx = 0.2
        dy = 0.2
        for i, (x, y) in enumerate(points):
            if i in nodes_idx:
                ax.text(x+dx, y+dy, str(i))
                ax.arrow(
                    x+dx, y+dx, -dx, -dy, color='black', head_width=dx/5)

        ax.set_title('Points in delaunay triangles')
        return fig


# %% ---- 2024-11-08 ------------------------
# Play ground
sns.set_theme('paper')

mg = MyGraph()
mg.generate_points_tri()
mg.find_neighbors()
# print(mg.weights)
fig = mg.plot_points_tri()
plt.show()

# %%
nodes_idx = [0, 1, 2]
# nodes_idx = [10, 11, 12]

mg.i = 0
for i in tqdm(range(5000)):
    for node_idx in nodes_idx:
        mg.set_node_value(node_idx)
    mg.extending()

x = mg.times[10:mg.i]
y = mg.values[1:, 10:mg.i].transpose()
plt.plot(x, y)
plt.show()

values = mg.values.copy()
for i in nodes_idx:
    values[i, :] = 0

measure = np.std(values[:, :mg.i], axis=1)
scatterplot_kwargs = {
    'size': measure,
    'hue': measure,
    'sizes': (1, 100)
}
fig = mg.plot_points_tri(scatterplot_kwargs, nodes_idx)
plt.show()

# %% ---- 2024-11-08 ------------------------
# Pending

# %% ---- 2024-11-08 ------------------------
# Pending

# %%
