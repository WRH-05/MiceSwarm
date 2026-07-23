# MiceSwarm — Multi-Agent Swarm Mapping Simulator

A lightweight, pure-Python multi-agent swarm mapping simulator with
real-time Foxglove Studio telemetry.  Four autonomous robots ("mice")
collaboratively map a procedurally generated maze using P2P gossip
synchronisation and frontier-based exploration.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the simulation
python -m swarm_sim.main --seed 42

# 3. Open Foxglove Studio → "Open connection" → ws://localhost:8765
```

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--num-agents` | 4 | Number of agents |
| `--maze-size` | 128 | Maze grid size (must be even, multiple of 16) |
| `--comm-range` | 15.0 | Communication range in cells |
| `--port` | 8765 | Foxglove WebSocket port |
| `--seed` | *random* | Random seed for reproducible runs |
| `--noise` | 0.05 | Odometry noise standard deviation |
| `--tick-rate` | 10.0 | Simulation tick rate in Hz |
| `--map-publish-rate` | 2 | Publish maps every N ticks |
| `--host` | 0.0.0.0 | Foxglove WebSocket host |

## Architecture

```
swarm_sim/
├── main.py            # Entrypoint, asyncio loop, CLI
├── maze.py            # DFS maze generator + DDA raycasting
├── map_grid.py        # Log-odds occupancy grid, spatial chunking
├── agent.py           # Robot sensing, movement, wall-following
├── planner.py         # Frontier detection, utility-discounted selection
├── comms.py           # P2P gossip sync with spatial delta exchange
├── foxglove_bridge.py # Foxglove WebSocket server + JSON schemas
├── requirements.txt   # Python dependencies
└── README.md
```

## How It Works

### Maze
A 128×128 perfect maze is generated via iterative DFS recursive backtracker.
Values: `0` = free space, `100` = wall.  Agents start at random interior
free cells with evenly distributed headings.

### Sensing
Each agent casts 8 rays (cardinal + intercardinal) up to 10 cells using
DDA raycasting against the ground-truth maze.  Traversed cells are marked
as free; wall hits as occupied.

### Local Mapping (Log-Odds)
Each agent maintains a log-odds occupancy grid (`np.float64`).  Updates
use `L_free = -0.85`, `L_occupied = +0.85`.  The grid is partitioned into
16×16 chunks (8×8 total) with CRC32 hashing for efficient P2P delta
detection.

### Frontier Exploration
Frontier cells (free cells adjacent to unknown) are clustered via
`scipy.ndimage.label`.  Utility-based target selection:
`U(F) = 1/distance - peer_penalty`, with target persistence to prevent
oscillation.  When no target is available, agents switch to wall-following
(right-hand rule) exploration.

### P2P Communication
Every tick, pairwise Euclidean distances are checked.  Agents within
`comm_range` (default 15 cells) exchange 8×8 chunk CRC32 hashes.  Only
differing chunks are transferred.  Both sides perform Bayesian log-odds
fusion: `L_new = L_A + L_B - L_0` (with `L_0 = 0`).

### Foxglove Telemetry
Channels published over WebSocket (JSON encoding, JSON Schema):

| Channel | Schema | Rate |
|---------|--------|------|
| `/mouse_{id}/pose` | `foxglove.PoseInFrame` | 10 Hz |
| `/mouse_{id}/local_map` | `nav_msgs/OccupancyGrid` | ~5 Hz |
| `/swarm/comms_links` | `foxglove.LinePrimitive` | 10 Hz |
| `/swarm/global_merged_map` | `nav_msgs/OccupancyGrid` | ~5 Hz |

## Foxglove Studio Setup

1. [Download Foxglove Studio](https://foxglove.dev/download)
2. Open → "Open connection" → WebSocket → `ws://localhost:8765`
3. Add panels:
   - **Plot** panel for `/mouse_{id}/pose` to see agent trajectories
   - **3D** or **Grid** panel for `/swarm/global_merged_map`
   - **Raw Messages** to inspect `/swarm/comms_links`

## Dependencies

- Python 3.10+
- `numpy` — grid operations
- `scipy` — connected-component labelling
- `foxglove-websocket` — WebSocket server for Foxglove Studio

## Notes

- `foxglove-websocket` is deprecated upstream (replaced by `foxglove-sdk`);
  it is used here as specified in the project requirements and still works.
- Maze generation is deterministic with `--seed`.
- Exploration is slow by design — a 128×128 DFS maze has ~8,000 free cells
  and narrow corridors.  Full exploration takes tens of minutes at 10 Hz.
- The global merged map shows the union of all agents' knowledge via
  Bayesian fusion — useful for visual validation of P2P sync correctness.

## License

MIT
