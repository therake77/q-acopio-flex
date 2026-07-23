import sys
from pathlib import Path

import matplotlib.pyplot as plt

from qiskit.primitives import StatevectorSampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset.dataset_generator import Graph
from dataset.real_dataset_generator import build_real_graph, load_labels, write_labels

# --- Datos de la instancia ---------------------------------------------------
# Los actores provienen de OpenStreetMap: fabricas (productores), almacenes
# (centros/hubs) y distritos (puntos de ruteo) de una region real.

GRAPH_FILE = str(Path(__file__).resolve().parents[1] / "graph_data.txt")
REGION = "Lima, Peru"

# El QAOA se simula con statevector exacto, asi que mantenemos la instancia
# chica: n_qubits ~ |z| + |y| + holguras. 2 productores x 2 hubs ~ 10 qubits.
MAX_PRODUCTORES = 2
MAX_CENTROS = 2

# Parametros economicos.
c = 1.2         # Costo base por t-km (camion chico)
alpha = 0.65    # Factor de descuento del tramo troncal (camion pesado de 30t)

# Cargar el grafo real; si no existe, generarlo en vivo desde OpenStreetMap.
try:
    graph = Graph.from_file(GRAPH_FILE)
    labels = load_labels(GRAPH_FILE)
    print(f"Instancia real cargada desde {GRAPH_FILE}")
except FileNotFoundError:
    print(f"{GRAPH_FILE} no encontrado; generando desde OpenStreetMap ({REGION})...")
    graph = build_real_graph(place=REGION, num_producers=MAX_PRODUCTORES,
                             num_intermediates=MAX_CENTROS, num_consumers=3)
    graph.write_to_file(GRAPH_FILE)
    write_labels(graph, GRAPH_FILE)
    labels = getattr(graph, "labels", None)

# Recortar a un tamano tratable para QAOA exacto.
n_prod = min(MAX_PRODUCTORES, len(graph.producers_coords))
n_cent = min(MAX_CENTROS, len(graph.intermediates_coords))


def _names(kind, n, prefix):
    if labels and labels.get(kind):
        return [labels[kind][k] for k in range(n)]
    return [f"{prefix}_{k}" for k in range(n)]


productores = _names("producers", n_prod, "Productor")
centros = _names("intermediates", n_cent, "Hub")

# Oferta real-derivada por productor (toneladas, s_i) y costo fijo por hub (f_h).
oferta = {productores[i]: int(graph.supply_s[i]) for i in range(n_prod)}
costo_fijo = {centros[h]: int(graph.fixed_cost_f[h]) for h in range(n_cent)}

# Distancia efectiva productor -> hub usando distancias geograficas REALES
# (haversine, km): min sobre puntos de ruteo de d_ih(prod->ruteo) + alpha*d_hj(ruteo->hub).
eff_dist = (
    graph.d_ih[:n_prod, :, None] + alpha * graph.d_hj[None, :, :n_cent]
).min(axis=1)  # forma (n_prod, n_cent), en km

# Costo consolidado C_ih = oferta_i * c * distancia_efectiva.
C_ih = {}
for i, prod in enumerate(productores):
    for h, cen in enumerate(centros):
        C_ih[(prod, cen)] = float(oferta[prod] * c * eff_dist[i, h])

print(f"Instancia cargada: {len(productores)} productores y {len(centros)} centros candidatos.")

# --- Resumen de la instancia -------------------------------------------------
print("-" * 72)
print(f"{'Productor':22s} | {'Centro':22s} | {'Dist (km)':>9s} | {'Flete (S/.)':>12s}")
print("-" * 72)
for i, prod in enumerate(productores):
    for h, cen in enumerate(centros):
        print(f"  {prod:20s} | {cen:20s} | {eff_dist[i, h]:9.1f} | {C_ih[(prod, cen)]:12,.2f}")
print("-" * 72)
print(f"Oferta por productor (t):    {oferta}")
print(f"Costo fijo de apertura (f_h): {costo_fijo}")

qp = QuadraticProgram(name="Centros_de_Acopio_Hubs")

# variables binarias: z_ih asigna productor i a hub h, y_h activa el hub h
for i in productores:
    for h in centros:
        qp.binary_var(name=f"z_{i}_{h}")

for h in centros:
    qp.binary_var(name=f"y_{h}")

# objetivo: minimizar costo fijo de hubs abiertos + flete de las asignaciones
linear_obj = {}
for h in centros:
    linear_obj[f"y_{h}"] = costo_fijo[h]

for i in productores:
    for h in centros:
        linear_obj[f"z_{i}_{h}"] = C_ih[(i, h)]

qp.minimize(linear=linear_obj)

# cada productor se asigna a exactamente un hub
for i in productores:
    qp.linear_constraint(
        linear={f"z_{i}_{h}": 1 for h in centros},
        sense="==",
        rhs=1,
        name=f"asignacion_{i}"
    )

# solo se puede asignar a un hub que este abierto (z_ih <= y_h)
for i in productores:
    for h in centros:
        qp.linear_constraint(
            linear={f"z_{i}_{h}": 1, f"y_{h}": -1},
            sense="<=",
            rhs=0,
            name=f"activacion_{i}_{h}"
        )

print("-" * 60)
print("Programa cuadratico")
print("-" * 60)
print(qp.prettyprint())


# --- Conversion a QUBO y conteo de qubits ------------------------------------
converter = QuadraticProgramToQubo()
qubo = converter.convert(qp)

print("-" * 60)
print("Formulacion QUBO")
print("-" * 60)
print(qubo.prettyprint())

n_asignacion = len(productores) * len(centros)
n_activacion = len(centros)
total_vars = qubo.get_num_vars()
n_slack = total_vars - (n_asignacion + n_activacion)

print("\nEstructura de variables:")
print(f"  asignacion (z_ih): {n_asignacion}")
print(f"  activacion (y_h):  {n_activacion}")
print(f"  slack (holgura):   {n_slack}")
print(f"  total de qubits:   {total_vars}")

# --- Hamiltoniano de Ising ---------------------------------------------------
ising_op, offset = qubo.to_ising()

print("-" * 60)
print("Hamiltoniano de Ising (H_C)")
print("-" * 60)
print(f"Offset (constante de energia): {offset:.4f}")
print(f"Qubits requeridos: {ising_op.num_qubits}")
print("Primeros terminos del operador de Pauli:")

terms = str(ising_op).split('\n')
for term in terms[:15]:
    print(f"  {term}")
if len(terms) > 15:
    print(f"  ... ({len(terms) - 15} terminos mas)")

# --- Configuracion del QAOA --------------------------------------------------
sampler = StatevectorSampler()          # simulacion statevector exacta
optimizer = COBYLA(maxiter=200)         # optimizador clasico de angulos
repsqaoa = 2                            # p = 2 capas variacionales

qaoa = QAOA(
    sampler=sampler,
    optimizer=optimizer,
    reps=repsqaoa
)

print("Configuracion de QAOA:")
print(f"  Sampler:     StatevectorSampler (simulacion local)")
print(f"  Optimizador: COBYLA (max 200 iteraciones)")
print(f"  Capas (p):   {repsqaoa}")
print(f"  Parametros:  {2 * repsqaoa} ({repsqaoa} gamma + {repsqaoa} beta)")

# --- Ejecucion del QAOA ------------------------------------------------------
meo = MinimumEigenOptimizer(qaoa)  #type: ignore

print("Ejecutando QAOA (puede tomar unos segundos)...")
result = meo.solve(qp)

print("\n" + "-" * 60)
print("Resultado de la optimizacion QAOA")
print("-" * 60)
print(result.prettyprint())

# --- Interpretacion de la solucion -------------------------------------------
sol_dict = result.variables_dict

print("\n" + "-" * 60)
print("Decision de centros de acopio (y_h)")
print("-" * 60)

costo_fijo_total = 0
for h in centros:
    val = int(sol_dict[f"y_{h}"])
    estado = "ABIERTO" if val == 1 else "CERRADO"
    costo_f = costo_fijo[h] if val == 1 else 0
    costo_fijo_total += costo_f
    print(f"  [{estado}] Centro {h:24s} costo fijo: S/. {costo_fijo[h]:,d}")

print("\n" + "-" * 60)
print("Asignacion de productores (z_ih)")
print("-" * 60)

flete_total = 0
for i in productores:
    for h in centros:
        if int(sol_dict[f"z_{i}_{h}"]) == 1:
            costo_fl = C_ih[(i, h)]
            flete_total += costo_fl
            print(f"  {i:20s} -> {h:24s} flete: S/. {costo_fl:,.2f}")

print("-" * 60)
print(f"  Costo fijo total:  S/. {costo_fijo_total:,.2f}")
print(f"  Flete total:       S/. {flete_total:,.2f}")
print(f"  Costo total (QAOA):S/. {result.fval:,.2f}")
print("-" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Grafica 1: desglose de costos (fijo vs. flete)
categorias = ['Costo Fijo\n(Apertura)', 'Costo Flete\n(Transporte)', 'Costo Total']
montos = [costo_fijo_total, flete_total, result.fval]
colores = ['#3B82F6', '#10B981', '#F59E0B']

bars1 = axes[0].bar(categorias, montos, color=colores, edgecolor='black', linewidth=1.2)
axes[0].set_ylabel('Costo en Soles (S/.)', fontsize=11)
axes[0].set_title('Desglose de Costos Totales - QAOA', fontweight='bold', fontsize=12)
axes[0].set_ylim(0, max(montos) * 1.25)

for bar, val in zip(bars1, montos):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(montos)*0.02),
                 f"S/. {val:,.2f}", ha='center', fontweight='bold', fontsize=10)

# Grafica 2: flete asignado por productor
fletes_productores = []
etiquetas_productores = []

for i in productores:
    for h in centros:
        if int(sol_dict[f"z_{i}_{h}"]) == 1:
            fletes_productores.append(C_ih[(i, h)])
            etiquetas_productores.append(f"{i}\n-> {h}")

bars2 = axes[1].bar(etiquetas_productores, fletes_productores, color='#6366F1', edgecolor='black', linewidth=1.2)
axes[1].set_ylabel('Costo de Flete (S/.)', fontsize=11)
axes[1].set_title('Flete de Transporte por Productor Asignado', fontweight='bold', fontsize=12)
axes[1].set_ylim(0, max(fletes_productores) * 1.25 if fletes_productores else 100)

for bar, val in zip(bars2, fletes_productores):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(fletes_productores)*0.02),
                 f"S/. {val:,.2f}", ha='center', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.show()
