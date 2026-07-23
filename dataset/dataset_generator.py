import json

import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

class Graph:

    def __init__(self, producers_coords, consumers_coords, intermediates_coords, supply_s, fixed_cost_f, d_ih, d_hj):
        self.producers_coords = producers_coords
        self.consumers_coords = consumers_coords
        self.intermediates_coords = intermediates_coords
        self.supply_s = supply_s
        self.fixed_cost_f = fixed_cost_f
        self.d_ih = d_ih
        self.d_hj = d_hj

    def write_to_file(self, filename):
        data = {
            "producers_coords": np.asarray(self.producers_coords).tolist(),
            "consumers_coords": np.asarray(self.consumers_coords).tolist(),
            "intermediates_coords": np.asarray(self.intermediates_coords).tolist(),
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
            "consumers_coords",
            "intermediates_coords",
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

    @classmethod
    def from_file(cls, filename):
        graph = cls(None, None, None, None, None, None, None)
        graph.load_from_file(filename)
        return graph

    def show(self):
        plt.figure(figsize=(10, 8))
        plt.scatter(self.producers_coords[:, 0], self.producers_coords[:, 1], c='blue', label='Producers', s=100)
        plt.scatter(self.consumers_coords[:, 0], self.consumers_coords[:, 1], c='green', label='Consumers', s=100)
        plt.scatter(self.intermediates_coords[:, 0], self.intermediates_coords[:, 1], c='red', label='Intermediates', s=100)
        plt.title('Graph Visualization')
        plt.xlabel('Latitude')
        plt.ylabel('Longitude')
        plt.legend()
        plt.grid()
        plt.show()

if __name__ == "__main__":
    np.random.seed(42)

    num_producers = 4 
    num_consumers = 1
    num_intermediates = 4

    producers_coords = np.random.uniform(low=[4.5, -74.2], high=[4.6, -74.0], size=(num_producers, 2))
    intermediates_coords = np.random.uniform(low=[4.6, -74.2], high=[4.7, -74.0], size=(num_intermediates, 2))
    consumers_coords = np.random.uniform(low=[4.7, -74.2], high=[4.8, -74.0], size=(num_consumers, 2))

    supply_s = np.random.randint(10, 50, size=num_producers)
    fixed_cost_f = np.random.randint(100, 300, size=num_intermediates)

    d_ih = cdist(producers_coords, consumers_coords, metric='euclidean') * 100
    d_hj = cdist(consumers_coords, intermediates_coords, metric='euclidean') * 100

    graph = Graph(producers_coords, consumers_coords, intermediates_coords, supply_s, fixed_cost_f, d_ih, d_hj)
    graph.write_to_file('graph_data.txt')
    graph.show()

    loaded_graph = Graph.from_file("graph_data.txt")
    loaded_graph.show()