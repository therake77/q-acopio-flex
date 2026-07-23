# ------------------------------------------------------------------------------
# QAOA avanzado: modelo de ruteo 3D (productor -> hub -> consumidor)
# Cada productor elige exactamente una ruta (hub, consumidor), de modo que TODOS
# los consumidores son destinos reales del modelo (no un promedio ni un unico
# punto de ruteo). Los datos provienen de OpenStreetMap.
# ------------------------------------------------------------------------------
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
from dataset.real_dataset_generator import (
    build_real_graph,
    haversine_matrix,
    load_labels,
    write_labels,
)

# --- Datos de la instancia ---------------------------------------------------
# Actores reales (OpenStreetMap): fabricas (productores), almacenes (hubs) y
# distritos (consumidores / demanda final).

GRAPH_FILE = str(Path(__file__).resolve().parents[1] / "graph_data.txt")
REGION = "Lima, Peru"

# El modelo 3D usa |I|*|H|*|J| variables z mas |H| variables y, y el QAOA se
# simula con statevector exacto. Mantenemos la instancia chica (2x2x2 ~ 18 qubits).
MAX_PRODUCTORES = 2
MAX_CENTROS = 2
MAX_CONSUMIDORES = 2

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
                             num_intermediates=MAX_CENTROS, num_consumers=MAX_CONSUMIDORES)
    graph.write_to_file(GRAPH_FILE)
    write_labels(graph, GRAPH_FILE)
    labels = getattr(graph, "labels", None)

# Recortar a un tamano tratable para QAOA exacto.
n_prod = min(MAX_PRODUCTORES, len(graph.producers_coords))
n_cent = min(MAX_CENTROS, len(graph.intermediates_coords))
n_cons = min(MAX_CONSUMIDORES, len(graph.consumers_coords))

prod_xy = graph.producers_coords[:n_prod]
hub_xy = graph.intermediates_coords[:n_cent]
cons_xy = graph.consumers_coords[:n_cons]


def _names(kind, n, prefix):
    if labels and labels.get(kind):
        return [labels[kind][k] for k in range(n)]
    return [f"{prefix}_{k}" for k in range(n)]


productores = _names("producers", n_prod, "Productor")
centros = _names("intermediates", n_cent, "Hub")
consumidores = _names("consumers", n_cons, "Consumidor")

# Oferta por productor (toneladas, s_i) y costo fijo por hub (f_h).
oferta = {productores[i]: int(graph.supply_s[i]) for i in range(n_prod)}
costo_fijo = {centros[h]: int(graph.fixed_cost_f[h]) for h in range(n_cent)}

# Distancias geograficas REALES (haversine, km) para la topologia del modelo 3D:
#   d_rec[i, h]     = productor -> hub (recoleccion capilar)
#   d_troncal[h, j] = hub -> consumidor (tramo troncal)
d_rec = haversine_matrix(prod_xy, hub_xy)          # (n_prod, n_cent)
d_troncal = haversine_matrix(hub_xy, cons_xy)      # (n_cent, n_cons)

# Costo consolidado de la ruta completa C_ihj = oferta_i * c * (d_rec + alpha*d_troncal).
C_ihj = {}
for i, prod in enumerate(productores):
    for h, cen in enumerate(centros):
        for j, con in enumerate(consumidores):
            C_ihj[(prod, cen, con)] = float(
                oferta[prod] * c * (d_rec[i, h] + alpha * d_troncal[h, j])
            )

# --- Resumen de la instancia -------------------------------------------------
print("-" * 92)
print(f"{'Productor':20s} | {'Hub':18s} | {'Consumidor':18s} | {'Flete C_ihj (S/.)':>18s}")
print("-" * 92)
for prod in productores:
    for cen in centros:
        for con in consumidores:
            print(f"  {prod:18s} | {cen:16s} | {con:16s} | {C_ihj[(prod, cen, con)]:18,.2f}")
print("-" * 92)
print(f"Oferta por productor (t):    {oferta}")
print(f"Costo fijo de apertura (f_h): {costo_fijo}")

# --- Programa cuadratico -----------------------------------------------------
qp = QuadraticProgram(name="Red_Logistica_3D")

# variables binarias: z_ihj activa la ruta productor->hub->consumidor, y_h abre el hub
for i in productores:
    for h in centros:
        for j in consumidores:
            qp.binary_var(name=f"z_{i}_{h}_{j}")

for h in centros:
    qp.binary_var(name=f"y_{h}")

# objetivo: minimizar costo fijo de hubs abiertos + flete de las rutas activas
linear_obj = {}
for h in centros:
    linear_obj[f"y_{h}"] = costo_fijo[h]
for i in productores:
    for h in centros:
        for j in consumidores:
            linear_obj[f"z_{i}_{h}_{j}"] = C_ihj[(i, h, j)]
qp.minimize(linear=linear_obj)

# cada productor toma exactamente una ruta (hub, consumidor)
for i in productores:
    qp.linear_constraint(
        linear={f"z_{i}_{h}_{j}": 1 for h in centros for j in consumidores},
        sense="==",
        rhs=1,
        name=f"asignacion_{i}",
    )

# una ruta solo existe si su hub esta abierto (z_ihj <= y_h)
for i in productores:
    for h in centros:
        for j in consumidores:
            qp.linear_constraint(
                linear={f"z_{i}_{h}_{j}": 1, f"y_{h}": -1},
                sense="<=",
                rhs=0,
                name=f"activacion_{i}_{h}_{j}",
            )

# --- Conversion a QUBO (para conocer el numero de qubits) --------------------
qubo = QuadraticProgramToQubo().convert(qp)
print(f"Qubits necesarios: {qubo.get_num_vars()}")

# --- Ejecucion del QAOA ------------------------------------------------------
sampler = StatevectorSampler()          # simulacion statevector exacta
optimizer = COBYLA(maxiter=200)         # optimizador clasico de angulos
qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=2)  # p = 2 capas

meo = MinimumEigenOptimizer(qaoa)  #type: ignore
print("Ejecutando QAOA (puede tomar hasta un par de minutos)...")
result = meo.solve(qp)

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
print("Asignacion de rutas (z_ihj: productor -> hub -> consumidor)")
print("-" * 60)

flete_total = 0
for i in productores:
    for h in centros:
        for j in consumidores:
            if int(sol_dict[f"z_{i}_{h}_{j}"]) == 1:
                costo_fl = C_ihj[(i, h, j)]
                flete_total += costo_fl
                print(f"  {i:18s} -> {h:16s} -> {j:16s} flete: S/. {costo_fl:,.2f}")

print("-" * 60)
print(f"  Costo fijo total:  S/. {costo_fijo_total:,.2f}")
print(f"  Flete total:       S/. {flete_total:,.2f}")
print(f"  Costo total (QAOA):S/. {result.fval:,.2f}")
print("-" * 60)

# --- Mapa de la red logistica (grafo geografico real) ------------------------
# Dibuja la ciudad usada por la instancia y la ruta 3D elegida por QAOA:
# productor -> hub asignado -> consumidor.
from matplotlib.lines import Line2D

fig_map, axm = plt.subplots(figsize=(10, 8))
label_bbox = dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.65)

# consumidores (distritos de demanda)
axm.scatter(cons_xy[:, 1], cons_xy[:, 0], c='#10B981', marker='s', s=90,
            edgecolor='black', linewidth=0.6, zorder=3)
for j, name in enumerate(consumidores):
    axm.annotate(name, (cons_xy[j, 1], cons_xy[j, 0]), fontsize=8, color='#065F46',
                 xytext=(5, 5), textcoords='offset points', bbox=label_bbox, zorder=6)

# productores (fabricas)
axm.scatter(prod_xy[:, 1], prod_xy[:, 0], c='#3B82F6', marker='^', s=170,
            edgecolor='black', linewidth=0.8, zorder=4)
for i, name in enumerate(productores):
    dy = 8 if i % 2 == 0 else 16
    axm.annotate(name, (prod_xy[i, 1], prod_xy[i, 0]), fontsize=8, fontweight='bold',
                 color='#1E3A8A', xytext=(7, dy), textcoords='offset points',
                 bbox=label_bbox, zorder=6)

# hubs (estrella llena = abierto, hueca = cerrado)
for h in range(n_cent):
    is_open = int(sol_dict[f"y_{centros[h]}"]) == 1
    axm.scatter(hub_xy[h, 1], hub_xy[h, 0], marker='*',
                c='#F59E0B' if is_open else 'white', s=460 if is_open else 280,
                edgecolor='black', linewidth=1.0, zorder=5)
    axm.annotate(f"{centros[h]}\n({'ABIERTO' if is_open else 'cerrado'})",
                 (hub_xy[h, 1], hub_xy[h, 0]), fontsize=8, fontweight='bold',
                 color='#92400E' if is_open else '#6B7280',
                 xytext=(7, -18), textcoords='offset points', bbox=label_bbox, zorder=6)

# ruta activa: productor -> hub -> consumidor
for i in range(n_prod):
    for h in range(n_cent):
        for j in range(n_cons):
            if int(sol_dict[f"z_{productores[i]}_{centros[h]}_{consumidores[j]}"]) != 1:
                continue
            axm.plot([prod_xy[i, 1], hub_xy[h, 1], cons_xy[j, 1]],
                     [prod_xy[i, 0], hub_xy[h, 0], cons_xy[j, 0]],
                     '-', color='#6366F1', linewidth=1.8, alpha=0.85, zorder=2)

legend_handles = [
    Line2D([0], [0], marker='^', color='w', markerfacecolor='#3B82F6',
           markeredgecolor='black', markersize=12, label='Productor'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#10B981',
           markeredgecolor='black', markersize=10, label='Consumidor'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='#F59E0B',
           markeredgecolor='black', markersize=16, label='Hub abierto'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='white',
           markeredgecolor='black', markersize=13, label='Hub cerrado'),
    Line2D([0], [0], color='#6366F1', lw=2, label='Ruta i -> h -> j'),
]
axm.legend(handles=legend_handles, loc='best', fontsize=9)
axm.set_xlabel('Longitud')
axm.set_ylabel('Latitud')
axm.set_title(f'Red logistica 3D sobre {REGION}', fontweight='bold', fontsize=13)
axm.grid(True, alpha=0.3)
axm.margins(0.12)
fig_map.tight_layout()

# --- Graficas de costos ------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

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

# Grafica 2: flete por ruta completa (productor -> hub -> consumidor)
fletes_rutas = []
etiquetas_rutas = []
for i in productores:
    for h in centros:
        for j in consumidores:
            if int(sol_dict[f"z_{i}_{h}_{j}"]) == 1:
                fletes_rutas.append(C_ihj[(i, h, j)])
                etiquetas_rutas.append(f"{i}\n-> {h}\n-> {j}")

bars2 = axes[1].bar(etiquetas_rutas, fletes_rutas, color='#6366F1', edgecolor='black', linewidth=1.2)
axes[1].set_ylabel('Costo de Flete (S/.)', fontsize=11)
axes[1].set_title('Flete por Ruta Completa (i -> h -> j)', fontweight='bold', fontsize=12)
axes[1].set_ylim(0, max(fletes_rutas) * 1.25 if fletes_rutas else 100)

for bar, val in zip(bars2, fletes_rutas):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(fletes_rutas)*0.02),
                 f"S/. {val:,.2f}", ha='center', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.show()
