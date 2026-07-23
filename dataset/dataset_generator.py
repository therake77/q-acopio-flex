import json

import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

class Graph:

    def __init__(self, producers_coords, candidates_coords, markets_coords, supply_s, fixed_cost_f, d_ih, d_hj):
        self.producers_coords = producers_coords
        self.candidates_coords = candidates_coords
        self.markets_coords = markets_coords
        self.supply_s = supply_s
        self.fixed_cost_f = fixed_cost_f
        self.d_ih = d_ih
        self.d_hj = d_hj

    def write_to_file(self, filename):
        data = {
            "producers_coords": np.asarray(self.producers_coords).tolist(),
            "candidates_coords": np.asarray(self.candidates_coords).tolist(),
            "markets_coords": np.asarray(self.markets_coords).tolist(),
            "supply_s": np.asarray(self.supply_s).tolist(),
            "fixed_cost_f": np.asarray(self.fixed_cost_f).tolist(),
            "d_ih": np.asarray(self.d_ih).tolist(),
            "d_hj": np.asarray(self.d_hj).tolist(),
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    def load_from_file(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)

        fields = (
            "producers_coords",
            "candidates_coords",
            "markets_coords",
            "supply_s",
            "fixed_cost_f",
            "d_ih",
            "d_hj",
        )
        missing_fields = [field for field in fields if field not in data]
        if missing_fields:
            raise ValueError(f"Missing fields in graph file: {', '.join(missing_fields)}")

        for field in fields:
            setattr(self, field, np.asarray(data[field]))

    def show(self):
        plt.figure(figsize=(10, 8))
        plt.scatter(self.producers_coords[:, 0], self.producers_coords[:, 1], c='blue', label='Producers', s=100)
        plt.scatter(self.candidates_coords[:, 0], self.candidates_coords[:, 1], c='green', label='Candidates', s=100)
        plt.scatter(self.markets_coords[:, 0], self.markets_coords[:, 1], c='red', label='Markets', s=100)
        plt.title('Graph Visualization')
        plt.xlabel('Latitude')
        plt.ylabel('Longitude')
        plt.legend()
        plt.grid()
        plt.show()

np.random.seed(42)

num_producers = 8
num_candidates = 4      
num_markets = 2         

producers_coords = np.random.uniform(low=[4.5, -74.2], high=[4.8, -74.0], size=(num_producers, 2))
candidates_coords = np.random.uniform(low=[4.5, -74.2], high=[4.8, -74.0], size=(num_candidates, 2))
markets_coords = np.random.uniform(low=[4.5, -74.2], high=[4.8, -74.0], size=(num_markets, 2))

supply_s = np.random.randint(10, 50, size=num_producers)       
fixed_cost_f = np.random.randint(100, 300, size=num_candidates) 

d_ih = cdist(producers_coords, candidates_coords, metric='euclidean') * 100 
d_hj = cdist(candidates_coords, markets_coords, metric='euclidean') * 100 

graph = Graph(producers_coords, candidates_coords, markets_coords, supply_s, fixed_cost_f, d_ih, d_hj)
graph.write_to_file('graph_data.txt')

graph.show()

graph.load_from_file("graph_data.txt")

graph.show()