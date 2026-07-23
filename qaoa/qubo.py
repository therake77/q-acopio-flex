import sys
from pathlib import Path

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset.dataset_generator import Graph

"""
===============================================================================
CLASSICAL MILP FORMULATION: INTERMEDIATE FACILITY LOCATION MODEL
===============================================================================

SETS AND INDICES:
  - i in P : Producer zones
  - j in J : Intermediate facility sites

PARAMETERS:
  - f[j]        : Fixed cost to open intermediate facility j
  - d_prime[i,j]: Precomputed effective transportation cost from producer zone i
                  to intermediate facility j through a consumer routing point

DECISION VARIABLES:
  - y[j] in {0, 1}   : 1 if intermediate facility j is opened, 0 otherwise
  - z[i,j] in {0, 1} : 1 if producer i is assigned to facility j, 0 otherwise

OBJECTIVE FUNCTION:
  Minimize total infrastructure fixed costs + total transportation costs:
  
  min sum_{h in H} (f[h] * y[h]) + sum_{i in P} sum_{h in H} (d_prime[i,h] * z[i,h])

CONSTRAINTS:
  1. Single Allocation Constraint:
     Every producer i must be assigned to exactly one collector center.
     sum_{h in H} z[i,h] == 1    forall i in P

  2. Facility Linking Constraint:
     Producer i can only be assigned to center h if center h is actually open.
     z[i,h] <= y[h]              forall i in P, h in H

NOTE:
  The cardinality constraint (sum y[h] == p) is omitted. The solver automatically
  determines the optimal number of open centers by balancing fixed opening 
  costs against distance savings.
===============================================================================
"""

"""
===============================================================================
QUANTUM QUBO FORMULATION: INTERMEDIATE FACILITY LOCATION MODEL
===============================================================================

BINARY VARIABLES VECTOR:
  x = [y_j, z_ij]^T in {0, 1}^(|J| + |P|*|J|)

UNCONSTRAINED HAMILTONIAN H(y, z):
  H(y, z) = Base_Cost(y, z) + lambda_1 * Penalty_Assign(z) + lambda_2 * Penalty_Link(y, z)

1. BASE COST FUNCTION:
   C(y, z) = sum_{h in H} (f[h] * y[h]) + sum_{i in P} sum_{h in H} (d_prime[i,h] * z[i,h])

2. PENALTY TERMS (Constraint Conversions):
   a) Single Allocation Penalty (lambda_1):
      H_assign(z) = sum_{i in P} ( sum_{j in J} z[i,j] - 1 )^2
          = sum_{i in P} [ sum_{j} z[i,j] + 2 * sum_{j < j'} (z[i,j] * z[i,j']) - 2 * sum_{j} z[i,j] + 1 ]

   b) Facility Linking Penalty (lambda_2):
      H_link(y, z) = sum_{i in P} sum_{j in J} [ z[i,j] * (1 - y[j]) ]
           = sum_{i in P} sum_{j in J} [ z[i,j] - z[i,j] * y[j] ]

QUBO MATRIX COEFFICIENTS (Q):
  - Linear  y_j           : f[j]
  - Linear  z_ij          : d_prime[i,j] - lambda_1 + lambda_2
  - Quadratic z_ij * z_ij': +2 * lambda_1   (for j != j')
  - Quadratic z_ij * y_j  : -lambda_2

PENALTY WEIGHT HYPERPARAMETER RULE OF THUMB:
  lambda_1, lambda_2 > max_{i,h} ( f[h] + d_prime[i,h] )
===============================================================================
"""

class IntermediateFacilityQUBO:
  """Build a facility-location QUBO from a generated graph dataset."""

  def __init__(self, graph, alpha=1.0, penalty=200.0, name="Intermediate_Facility_Location"):
    self.graph = graph
    self.alpha = alpha
    self.penalty = penalty
    self.name = name
    self.producers = range(len(graph.producers_coords))
    self.facilities = range(len(graph.intermediates_coords))
    self.d_prime_matrix = self._build_effective_costs()
    self.d_prime = self._as_cost_dictionary(self.d_prime_matrix)
    self.fixed_costs = {
      j: float(graph.fixed_cost_f[j])
      for j in self.facilities
    }
    self.qp = None
    self.qubo_qp = None
    self.operator = None
    self.offset = None

  @classmethod
  def from_file(cls, filename, alpha=1.0, penalty=200.0, name="Intermediate_Facility_Location"):
    return cls(Graph.from_file(filename), alpha, penalty, name)

  def _build_effective_costs(self):
    return (
      self.graph.d_ih[:, :, None]
      + self.alpha * self.graph.d_hj[None, :, :]
    ).min(axis=1)

  def _as_cost_dictionary(self, cost_matrix):
    return {
      (i, j): float(cost_matrix[i, j])
      for i in self.producers
      for j in self.facilities
    }

  def build_quadratic_program(self):
    qp = QuadraticProgram(name=self.name)

    for j in self.facilities:
      qp.binary_var(name=f"y_{j}")

    for i in self.producers:
      for j in self.facilities:
        qp.binary_var(name=f"z_{i}_{j}")

    linear_objective = {
      f"y_{j}": self.fixed_costs[j]
      for j in self.facilities
    }
    linear_objective.update({
      f"z_{i}_{j}": self.d_prime[(i, j)]
      for i in self.producers
      for j in self.facilities
    })
    qp.minimize(linear=linear_objective)  #type: ignore

    for i in self.producers:
      qp.linear_constraint(
        linear={f"z_{i}_{j}": 1.0 for j in self.facilities},
        sense="==",
        rhs=1.0,
        name=f"single_alloc_{i}",
      )

    for i in self.producers:
      for j in self.facilities:
        qp.linear_constraint(
          linear={f"z_{i}_{j}": 1.0, f"y_{j}": -1.0},
          sense="<=",
          rhs=0.0,
          name=f"link_{i}_{j}",
        )

    self.qp = qp
    return qp

  def to_qubo(self):
    if self.qp is None:
      self.build_quadratic_program()
    self.qubo_qp = QuadraticProgramToQubo(penalty=self.penalty).convert(self.qp)  #type: ignore
    return self.qubo_qp

  def to_ising(self):
    if self.qubo_qp is None:
      self.to_qubo()
    self.operator, self.offset = self.qubo_qp.to_ising()  #type: ignore
    return self.operator, self.offset


def main():
  model = IntermediateFacilityQUBO.from_file(
    "graph_data.txt",
    alpha=1.0,
    penalty=200.0,
  )
  qp = model.build_quadratic_program()
  qubo_qp = model.to_qubo()
  operator, offset = model.to_ising()

  print("=== Standard Quadratic Program ===")
  print(qp.prettyprint())
  print("\n=== QUBO Formulation ===")
  print(qubo_qp.prettyprint())
  print("\n=== Ising Hamiltonian Details ===")
  print(f"Number of Qubits Required: {operator.num_qubits}")
  print(f"Energy Offset: {offset}")
  print("\nFirst 3 Pauli Terms in Hamiltonian:")
  print(operator[:3])


if __name__ == "__main__":
  main()