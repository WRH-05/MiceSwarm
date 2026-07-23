#!/usr/bin/env python3
"""Multi-Agent Swarm Mapping Simulation — entry point.

Launches 4 autonomous agents ("mice") in a procedurally generated maze.
Agents collaboratively explore and map using P2P gossip synchronisation
and frontier-based planning.  Real-time telemetry is streamed to
Foxglove Studio via WebSocket.

Usage
-----
python -m swarm_sim.main --num-agents 4 --maze-size 128 --comm-range 15 --port 8765
"""

import argparse
import asyncio
import logging
import math
import sys
import time

from foxglove_websocket import run_cancellable

from .maze import Maze
from .agent import Agent
from .planner import FrontierPlanner
from .comms import CommsManager
from .foxglove_bridge import FoxgloveBridge

logger = logging.getLogger("swarm_sim")


# ======================================================================
# CLI
# ======================================================================

def parse_args(argv: list | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-Agent Swarm Mapping Simulator (Foxglove Telemetry)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--num-agents", type=int, default=4,
        help="Number of agents (default: 4)",
    )
    p.add_argument(
        "--maze-size", type=int, default=128,
        help="Maze grid size, must be even (default: 128)",
    )
    p.add_argument(
        "--comm-range", type=float, default=15.0,
        help="Communication range in cells (default: 15)",
    )
    p.add_argument(
        "--port", type=int, default=8765,
        help="Foxglove WebSocket port (default: 8765)",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )
    p.add_argument(
        "--noise", type=float, default=0.05,
        help="Odometry noise std dev (default: 0.05)",
    )
    p.add_argument(
        "--tick-rate", type=float, default=10.0,
        help="Simulation tick rate in Hz (default: 10)",
    )
    p.add_argument(
        "--map-publish-rate", type=int, default=2,
        help="Publish maps every N ticks (default: 2 → 5 Hz at 10 Hz tick)",
    )
    p.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Foxglove WebSocket host (default: 0.0.0.0)",
    )
    return p.parse_args(argv)


# ======================================================================
# Main simulation
# ======================================================================

async def run_simulation(args: argparse.Namespace) -> None:
    """Set up the world and run the main tick loop."""

    # ------------------------------------------------------------------
    # 1. Generate maze
    # ------------------------------------------------------------------
    logger.info("Generating %dx%d maze (seed=%s) ...", args.maze_size, args.maze_size, args.seed)
    maze = Maze(args.maze_size, seed=args.seed)
    free_count = int((maze.grid == Maze.FREE).sum())
    logger.info("Maze ready: %d free cells, %d walls", free_count, int((maze.grid == Maze.WALL).sum()))

    # ------------------------------------------------------------------
    # 2. Place agents
    # ------------------------------------------------------------------
    starts = maze.get_random_free_cells(args.num_agents)
    agents = []
    for i, (sx, sy) in enumerate(starts):
        theta = (i * 2 * math.pi / args.num_agents)  # evenly distributed headings
        a = Agent(
            agent_id=i,
            start_x=sx,
            start_y=sy,
            start_theta=theta,
            maze=maze,
            noise_std=args.noise,
        )
        agents.append(a)
        logger.info(
            "Agent %d: start=(%.1f, %.1f)  heading=%.2f rad",
            i, sx, sy, theta,
        )

    # ------------------------------------------------------------------
    # 3. Create simulation components
    # ------------------------------------------------------------------
    planner = FrontierPlanner()
    comms = CommsManager(comm_range=args.comm_range)

    # ------------------------------------------------------------------
    # 4. Start Foxglove bridge
    # ------------------------------------------------------------------
    bridge = FoxgloveBridge(host=args.host, port=args.port)
    await bridge.start()

    # Register channels for each agent plus swarm-level channels
    for a in agents:
        await bridge.register_agent_channels(a.agent_id)
    await bridge.register_swarm_channels()
    logger.info("Foxglove channels registered (%d agent pairs + 2 swarm)", args.num_agents)

    # ------------------------------------------------------------------
    # 5. Main loop
    # ------------------------------------------------------------------
    dt = 1.0 / args.tick_rate
    tick = 0
    maps_published_this_tick = False

    logger.info(
        "Simulation running at %.1f Hz (dt=%.3f s). "
        "Connect Foxglove Studio to ws://%s:%d",
        args.tick_rate, dt, args.host, args.port,
    )

    try:
        while True:
            t_start = time.monotonic()
            tick += 1
            now_ns = time.time_ns()

            # --- PHASE 1: Sense ---
            for a in agents:
                a.sense()

            # --- PHASE 2: Plan ---
            # Build peer info for utility discounting: (x, y, target)
            peer_info = [
                (ag.x, ag.y, ag.target_frontier)
                for ag in agents
            ]

            for i, a in enumerate(agents):
                occ = a.local_map.get_occupancy_grid()
                frontiers = planner.find_frontiers(occ)
                # Exclude self from peers list
                peers = peer_info[:i] + peer_info[i + 1:]
                target = planner.select_target(
                    a.x, a.y, frontiers, peers, args.comm_range,
                    current_target=a.target_frontier,
                )
                a.set_target(target)

            # --- PHASE 3: Move ---
            for a in agents:
                a.move(dt)

            # --- PHASE 4: Communicate ---
            links = comms.tick(agents)

            # --- PHASE 5: Publish telemetry ---
            # Poses — every tick
            await bridge.publish_poses(agents, now_ns)

            # Comms links — every tick
            await bridge.publish_comms_links(links, agents, now_ns)

            # Maps — every N ticks (rate-limited)
            if tick % args.map_publish_rate == 0:
                await bridge.publish_local_maps(agents, now_ns)
                await bridge.publish_global_map(agents, now_ns)
                maps_published_this_tick = True
            else:
                maps_published_this_tick = False

            # Broadcast server time for synchronization
            if bridge.server is not None:
                await bridge.server.broadcast_time(now_ns)

            # --- Progress logging ---
            if tick % 50 == 0:
                known_fracs = [a.local_map.known_fraction for a in agents]
                avg_known = sum(known_fracs) / len(known_fracs) * 100
                logger.info(
                    "Tick %5d | map %.1f%% known | %d links | %d syncs (%d chunks)",
                    tick, avg_known, len(links),
                    comms.total_syncs, comms.total_chunks_exchanged,
                )

                # Check convergence
                if avg_known > 99.5 and all(
                    a.target_frontier is None for a in agents
                ):
                    logger.info("Exploration complete! Map > 99.5%% known.")
                    break

            # --- Sleep ---
            elapsed = time.monotonic() - t_start
            sleep_time = max(0.0, dt - elapsed)
            if elapsed > dt * 1.5:
                logger.warning(
                    "Tick %d overran: elapsed=%.3f ms (budget=%.3f ms)",
                    tick, elapsed * 1000, dt * 1000,
                )
            await asyncio.sleep(sleep_time)

    except asyncio.CancelledError:
        logger.info("Simulation cancelled.")
    finally:
        logger.info("Shutting down Foxglove bridge ...")
        await bridge.stop()

    # Final stats
    logger.info("=== Simulation Complete ===")
    for a in agents:
        logger.info(
            "Agent %d: pos=(%.1f, %.1f)  map %.1f%% known",
            a.agent_id, a.x, a.y, a.local_map.known_fraction * 100,
        )
    logger.info("Total syncs: %d  Chunks exchanged: %d", comms.total_syncs, comms.total_chunks_exchanged)


# ======================================================================
# Entry point
# ======================================================================

def main(argv: list | None = None) -> None:
    """Parse args, configure logging, and launch the asyncio loop."""
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Windows: use SelectorEventLoop for compatibility
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    run_cancellable(run_simulation(args))


if __name__ == "__main__":
    main()
