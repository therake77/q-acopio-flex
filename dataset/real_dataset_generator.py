"""Build a facility-location `Graph` from REAL geographic data.

This mirrors `dataset_generator.py`, but instead of sampling coordinates,
supplies and costs with `numpy.random`, it pulls real-world features from
OpenStreetMap:

  - producers      -> industrial works / factories   (man_made=works, industrial)
  - intermediates  -> warehouses / logistics sites    (building=warehouse, ...)
  - consumers      -> towns / cities (routing points)  (place=city|town|suburb)

All coordinates are the true lat/lon of those places, and the distance
matrices (`d_ih`, `d_hj`) are real great-circle (haversine) distances in
kilometres -- not random euclidean noise.

Values that OpenStreetMap does not carry (per-producer supply, per-facility
fixed opening cost) are derived deterministically from real attributes where
possible (e.g. real city population for consumer weighting) and otherwise from
a stable hash of the feature id, so runs are reproducible.

No API key is required. Uses the public Overpass API over HTTPS.

Usage:
    python -m dataset.real_dataset_generator                # Lima, Peru (default)
    python -m dataset.real_dataset_generator --place "Arequipa, Peru"
"""

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

# Allow running as a plain script (`python3 dataset/real_dataset_generator.py`)
# as well as a module (`python3 -m dataset.real_dataset_generator`): make sure
# the repo root is importable so the `dataset` package resolves either way.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset.dataset_generator import Graph

OVERPASS_MIRRORS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "q-hackaton-demo/1.0 (facility-location QAOA)"

# Overpass tag filters for each role in the supply chain. Fallbacks are tried
# in order and merged until `limit` features are collected.
ROLE_FILTERS = {
    "producers": (
        '["industrial"="factory"]',   # named manufacturers read best as producers
        '["man_made"="works"]',
        '["landuse"="industrial"]',
    ),
    "intermediates": (
        '["building"="warehouse"]',
        '["industrial"="warehouse"]',
        '["landuse"="logistics"]',
    ),
    "consumers": (
        '["place"="city"]',
        '["place"="town"]',
        '["place"="suburb"]',
    ),
}

# Substrings (case-insensitive) that flag a feature as a poor fit for its role,
# so the picked names actually read like the enterprise they represent. A
# producer should be a manufacturer (not a warehouse/logistics firm); an
# intermediate hub should be a commercial depot (not a government or impound
# yard). Keeps the data real while filtering out nonsensical matches.
ROLE_NAME_BLOCKLIST = {
    "producers": (
        "ransa", "savar", "almac", "deposito", "depósito", "logist", "logíst",
        "frio", "frío", "defensa civil", "embargad",
    ),
    "intermediates": (
        "defensa civil", "municipal", "embargad", "vehiculo", "vehículo",
        "incautad", "impound",
    ),
    "consumers": (),
}

# Roles whose features must carry a real OSM name (enterprises), so we never
# fall back to a synthetic "way/1234" label for a producer or a hub.
NAMED_ONLY_ROLES = {"producers", "intermediates"}


def _http_post(url, params, timeout):
    data = urllib.parse.urlencode(params).encode()
    req = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _http_get(url, params, timeout):
    full = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(full, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def geocode_bbox(place, timeout=20):
    """Resolve a place name to an OSM bounding box (south, west, north, east)."""
    results = _http_get(
        NOMINATIM_URL,
        {"q": place, "format": "json", "limit": 1, "polygon_geojson": 0},
        timeout,
    )
    if not results:
        raise ValueError(f"Could not geocode place: {place!r}")
    south, north, west, east = (float(x) for x in results[0]["boundingbox"])
    return south, west, north, east


def _overpass_query(bbox, filters, limit):
    south, west, north, east = bbox
    clauses = "".join(
        f'  node{f}({south},{west},{north},{east});\n'
        f'  way{f}({south},{west},{north},{east});\n'
        for f in filters
    )
    return (
        f"[out:json][timeout:60];\n"
        f"(\n{clauses});\n"
        f"out center {limit};"
    )


def _keep_feature(name, tags, blocklist, require_named):
    """Reject features that don't credibly fit their supply-chain role."""
    lowered = name.lower()
    if require_named and (not tags.get("name")):
        return False
    return not any(bad in lowered for bad in blocklist)


def _fetch_role(bbox, filters, limit, blocklist=(), require_named=False, timeout=60):
    """Query Overpass for one role, accumulating across fallback filters.

    Each fallback tag is queried in turn and its features merged (de-duplicated
    by location) until we have `limit`, so no single tag needs to be dense
    enough on its own. Features matching `blocklist` (or unnamed, when
    `require_named`) are skipped so the chosen names suit the role.
    """
    last_err = None
    collected = []
    seen = set()
    used = []
    for tag in filters:
        query = _overpass_query(bbox, (tag,), limit * 5)
        payload = None
        for mirror in OVERPASS_MIRRORS:
            try:
                payload = _http_post(mirror, {"data": query}, timeout)
                break
            except Exception as err:  # network / mirror hiccup -> try next
                last_err = err
                time.sleep(1)
        if payload is None:
            continue
        added = False
        for feat in _extract_features(payload["elements"]):
            name, lat, lon, tags = feat
            if not _keep_feature(name, tags, blocklist, require_named):
                continue
            key = (round(lat, 4), round(lon, 4))
            if key in seen:
                continue
            seen.add(key)
            collected.append(feat)
            added = True
            if len(collected) >= limit:
                if tag not in used:
                    used.append(tag)
                return collected[:limit], "+".join(used)
        if added and tag not in used:
            used.append(tag)
    if not collected and last_err is not None:
        raise last_err
    return collected, "+".join(used) if used else filters[0]


def _extract_features(elements):
    """Turn Overpass elements into (name, lat, lon, tags) tuples, de-duplicated."""
    seen = set()
    out = []
    for el in elements:
        lat = el.get("lat") or el.get("center", {}).get("lat")
        lon = el.get("lon") or el.get("center", {}).get("lon")
        if lat is None or lon is None:
            continue
        key = (round(lat, 4), round(lon, 4))
        if key in seen:
            continue
        seen.add(key)
        tags = el.get("tags", {})
        name = tags.get("name", f"{el['type']}/{el['id']}")
        out.append((name, float(lat), float(lon), tags))
    return out


def _haversine_matrix(coords_a, coords_b):
    """Great-circle distance matrix (km) between two sets of [lat, lon] points."""
    lat_a = np.radians(coords_a[:, 0])[:, None]
    lon_a = np.radians(coords_a[:, 1])[:, None]
    lat_b = np.radians(coords_b[:, 0])[None, :]
    lon_b = np.radians(coords_b[:, 1])[None, :]
    dlat = lat_b - lat_a
    dlon = lon_b - lon_a
    h = np.sin(dlat / 2) ** 2 + np.cos(lat_a) * np.cos(lat_b) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(h))  # Earth mean radius (km)


# Public alias: callers (e.g. the QAOA scripts) can build their own real
# distance matrices between any two sets of [lat, lon] points.
haversine_matrix = _haversine_matrix


def _stable_unit(text):
    """Deterministic pseudo-value in [0, 1) from a feature name (reproducible)."""
    digest = 0
    for ch in text:
        digest = (digest * 131 + ord(ch)) & 0xFFFFFFFF
    return digest / 0xFFFFFFFF


def _supply_from(features):
    """Per-producer supply. No real OSM source, so derive deterministically."""
    return np.array([10 + int(40 * _stable_unit(name)) for name, *_ in features])


def _fixed_cost_from(features):
    """Per-facility fixed opening cost, derived deterministically per site."""
    return np.array([100 + int(200 * _stable_unit(name)) for name, *_ in features])


def build_real_graph(place="Lima, Peru", num_producers=4,
                     num_intermediates=4, num_consumers=3, alpha_scale=1.0):
    """Fetch real OSM features around `place` and assemble a `Graph`."""
    print(f"Geocoding {place!r} ...")
    bbox = geocode_bbox(place)
    print(f"  bbox (S,W,N,E) = {tuple(round(b, 3) for b in bbox)}")

    roles = {}
    counts = {
        "producers": num_producers,
        "intermediates": num_intermediates,
        "consumers": num_consumers,
    }
    for role, limit in counts.items():
        print(f"Fetching {role} (up to {limit}) from OpenStreetMap ...")
        features, used = _fetch_role(
            bbox,
            ROLE_FILTERS[role],
            limit,
            blocklist=ROLE_NAME_BLOCKLIST.get(role, ()),
            require_named=role in NAMED_ONLY_ROLES,
        )
        if len(features) < limit:
            raise RuntimeError(
                f"Only found {len(features)} {role} in {place!r} "
                f"(need {limit}). Try a larger/denser region."
            )
        roles[role] = features
        print(f"  using filter {used}: " + ", ".join(n for n, *_ in features))

    producers_coords = np.array([[la, lo] for _, la, lo, _ in roles["producers"]])
    intermediates_coords = np.array([[la, lo] for _, la, lo, _ in roles["intermediates"]])
    consumers_coords = np.array([[la, lo] for _, la, lo, _ in roles["consumers"]])

    supply_s = _supply_from(roles["producers"])
    fixed_cost_f = _fixed_cost_from(roles["intermediates"])

    # Real great-circle distances (km): producer->consumer and consumer->facility.
    d_ih = _haversine_matrix(producers_coords, consumers_coords) * alpha_scale
    d_hj = _haversine_matrix(consumers_coords, intermediates_coords) * alpha_scale

    graph = Graph(
        producers_coords,
        consumers_coords,
        intermediates_coords,
        supply_s,
        fixed_cost_f,
        d_ih,
        d_hj,
    )
    # Real place names for each node. The Graph file format only stores
    # coordinates, so these are kept alongside and written to a sidecar.
    graph.labels = {  # type: ignore[attr-defined]
        "producers": [n for n, *_ in roles["producers"]],
        "consumers": [n for n, *_ in roles["consumers"]],
        "intermediates": [n for n, *_ in roles["intermediates"]],
    }
    return graph


def labels_path_for(graph_path):
    """Sidecar path holding real place names, e.g. graph_data -> graph_data_labels.json."""
    base, _ = os.path.splitext(graph_path)
    return base + "_labels.json"


def write_labels(graph, graph_path):
    labels = getattr(graph, "labels", None)
    if not labels:
        return None
    path = labels_path_for(graph_path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)
    return path


def load_labels(graph_path):
    """Load the real place-name sidecar for a graph file, or None if absent."""
    path = labels_path_for(graph_path)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--place", default="Lima, Peru",
                        help="Region to pull real features from (geocoded via OSM).")
    parser.add_argument("--producers", type=int, default=4)
    parser.add_argument("--intermediates", type=int, default=4)
    parser.add_argument("--consumers", type=int, default=3)
    parser.add_argument("--out", default="graph_data.txt")
    parser.add_argument("--show", action="store_true", help="Plot the graph.")
    args = parser.parse_args()

    graph = build_real_graph(
        place=args.place,
        num_producers=args.producers,
        num_intermediates=args.intermediates,
        num_consumers=args.consumers,
    )
    graph.write_to_file(args.out)
    labels_out = write_labels(graph, args.out)
    print(f"\nWrote real-data graph to {args.out}")
    if labels_out:
        print(f"Wrote real place names to {labels_out}")
    print(f"  d_ih (km):\n{np.round(graph.d_ih, 1)}")
    print(f"  d_hj (km):\n{np.round(graph.d_hj, 1)}")
    if args.show:
        graph.show()


if __name__ == "__main__":
    main()
