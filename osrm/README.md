# OSRM — standing up the routing service

Spec Sections 2.2 and 2.3. OSRM is BSD-2 and runs as a **separate process behind HTTP**, so
it is not linked into the product and raises no licence question (Section 2.4).

> **This environment cannot run any of the below.** Docker is not available, so
> `osrm-routed` cannot be started here. Everything in the engine routes through the
> `TravelTimeBackend` protocol in `ufe/layers/routing.py`; offline runs and the whole test
> suite use `HaversineBackend`, the deterministic non-network fallback. Tests that need a
> live service are marked `needs_osrm` and additionally require `UFE_OSRM_ENDPOINT` to be
> set. Nothing here is on the simulation path — matrices are precomputed and persisted, and
> `ufe.layers.l1_accessibility` never calls a backend (CONTRACT.md rule 3).

## 0. Layout

```
osrm/
  andhra-pradesh-latest.osm.pbf     # downloaded, gitignored
  profiles/
    twowheeler.lua                  # committed (Section 2.3)
    twowheeler_speeds.lua           # GENERATED from accessibility.yaml, gitignored
  README.md
```

## 1. Get the extract

```bash
cd osrm
curl -LO https://download.geofabrik.de/asia/india/andhra-pradesh-latest.osm.pbf
```

Geofabrik extracts are OpenStreetMap data under ODbL. Routing output is a Produced Work;
raw OSM-derived columns must not be exposed through the API (Section 22.1, `ufe/rights.py`).

## 2. Generate the two-wheeler speed constants

`osrm-extract` runs inside a container with no YAML parser, and CONTRACT.md rule 1 forbids
writing speeds into a `.lua` file. So `twowheeler.lua` `require`s a generated pure-data
module and errors out if it is missing:

```bash
python -c "from ufe.params import load_params; \
  from ufe.layers.routing import write_twowheeler_constants as w; \
  w(load_params('vizag'), 'osrm/profiles/twowheeler_speeds.lua')"
```

The generated file is stamped with the params hash so a matrix can be traced to the
parameter version that produced it. **Regenerate it and re-run `osrm-extract` whenever
`config/params/accessibility.yaml` changes** — Section 21's "stale cache across param
versions" applies to the routing graph too, not only to our own caches.

> **Known gap.** `modes.two_wheeler.speed_factor`, which Section 2.3 names as the source of
> the scaling, is **not present** in `config/params/accessibility.yaml`. The generator
> raises `MissingParameter` naming the path rather than assuming the `0.85` printed in the
> spec prose. Add the leaf (with `conf` and `scope`), or pass `speed_factor=` explicitly,
> before extracting with this profile.

## 3. Build the graph, per profile

`{profile}` is one of `car`, `foot` (both ship with `osrm/osrm-backend` at
`/opt/{profile}.lua`) and `twowheeler` (ours, at `/data/profiles/twowheeler.lua`).

```bash
# car and foot use the stock profiles
docker run -t -v "${PWD}/osrm:/data" osrm/osrm-backend \
  osrm-extract -p /opt/car.lua /data/andhra-pradesh-latest.osm.pbf

# the two-wheeler profile is ours; it lives under the mounted /data
docker run -t -v "${PWD}/osrm:/data" osrm/osrm-backend \
  osrm-extract -p /data/profiles/twowheeler.lua /data/andhra-pradesh-latest.osm.pbf

docker run -t -v "${PWD}/osrm:/data" osrm/osrm-backend \
  osrm-partition /data/andhra-pradesh-latest.osrm
docker run -t -v "${PWD}/osrm:/data" osrm/osrm-backend \
  osrm-customize /data/andhra-pradesh-latest.osrm
```

`osrm-extract` overwrites the `.osrm*` files in place, so **each profile needs its own
output directory**. Give each profile a copy of the `.pbf` in its own subdirectory, or run
the three profiles sequentially into `osrm/car/`, `osrm/foot/`, `osrm/twowheeler/`.

`twowheeler.lua` does `require('car')` and `require('twowheeler_speeds')`; both must be on
the Lua path. The stock image puts `car.lua` on `/opt`, and the profile's own directory is
searched, so mounting `osrm/profiles` as `/data/profiles` is enough for `twowheeler_speeds`.

## 4. Run the service

```bash
docker run -d -p 5000:5000 -v "${PWD}/osrm/car:/data" osrm/osrm-backend \
  osrm-routed --algorithm mld --max-table-size 10000 /data/andhra-pradesh-latest.osrm
```

**The default `--max-table-size` is 100 and will reject every matrix request we make.** It
must be raised. `OSRMBackend` chunks requests at `accessibility.matrix.chunk_size` (500),
so 10,000 leaves generous headroom; if the service still answers `NoTable`, lower
`chunk_size` in YAML rather than editing Python.

One container per profile, on separate ports:

| Profile | Port | Serves |
|---|---|---|
| `car` | 5000 | mode `car` |
| `twowheeler` | 5001 | mode `two_wheeler` |
| `foot` | 5002 | mode `walk`, and the walk-network distances of Section 8.4 |

`OSRMBackend` accepts either a single endpoint or a `{profile: endpoint}` mapping:

```python
from ufe.layers.routing import OSRMBackend, precompute_matrices
from ufe.params import load_params

params = load_params("vizag")
backend = OSRMBackend(
    {"car": "http://localhost:5000",
     "twowheeler": "http://localhost:5001",
     "foot": "http://localhost:5002"},
    params,
)
matrices = precompute_matrices(
    cells, params, backend, network_state=frozenset(), cache_dir="data/cache"
)
```

## 5. Check it

```bash
curl "http://localhost:5000/table/v1/car/83.30,17.70;83.32,17.72?annotations=duration"
UFE_OSRM_ENDPOINT=http://localhost:5000 pytest -m needs_osrm
```

A healthy response is `{"code":"Ok","durations":[[0,...],[...]]}`, durations in **seconds**;
`OSRMBackend` converts to minutes and maps `null` to `inf`.

## 6. Where the matrices go

`precompute_matrices` writes `float32` arrays to
`{cache_dir}/ttm/{mode}/{key}.npy` with an index at `{cache_dir}/ttm_index.json`
(Section 5.2). The key hashes the params hash, the network-state hash and the
origin/destination sets, so a parameter change invalidates the cache and a repeat call with
the same network state issues **zero** OSRM requests. Simulation reads these files through
`load_matrices` and never contacts the service.
