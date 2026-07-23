"""Foxglove WebSocket telemetry bridge.

Launches a ``FoxgloveServer`` and publishes standard channels so
Foxglove Studio can visualise agent poses, occupancy maps, comms links,
and a global merged map — all in real time.

All channels use JSON encoding with JSON Schema definitions.
"""

import json
import math
import time
import numpy as np
from foxglove_websocket.server import FoxgloveServer

from .map_grid import OccupancyGrid


class FoxgloveBridge:
    """Manages a Foxglove WebSocket server and channel publishing."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self.host = host
        self.port = port
        self.server: FoxgloveServer | None = None

        # Channel IDs keyed by topic name
        self._pose_channels: dict[int, int] = {}
        self._local_map_channels: dict[int, int] = {}
        self._comms_channel: int | None = None
        self._global_map_channel: int | None = None

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Create the FoxgloveServer, open it, and register all channels."""
        self.server = FoxgloveServer(
            self.host,
            self.port,
            "MiceSwarm",
            supported_encodings=["json"],
        )
        self.server.start()
        await self.server.wait_opened()
        print(f"[FoxgloveBridge] Listening on ws://{self.host}:{self.port}")

    async def stop(self) -> None:
        """Gracefully shut down the server."""
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
            print("[FoxgloveBridge] Server stopped.")

    # ------------------------------------------------------------------
    # Channel registration (call after start(), before ticking)
    # ------------------------------------------------------------------

    async def register_agent_channels(self, agent_id: int) -> None:
        """Register pose and local-map channels for one agent."""
        if self.server is None:
            raise RuntimeError("Call start() before registering channels.")

        # --- Pose channel ---
        sn, sc = _make_pose_schema()
        chan_id = await self.server.add_channel({
            "topic": f"/mouse_{agent_id}/pose",
            "encoding": "json",
            "schemaName": sn,
            "schema": sc,
            "schemaEncoding": "jsonschema",
        })
        self._pose_channels[agent_id] = chan_id

        # --- Local map channel ---
        sn, sc = _make_occupancy_grid_schema()
        chan_id = await self.server.add_channel({
            "topic": f"/mouse_{agent_id}/local_map",
            "encoding": "json",
            "schemaName": sn,
            "schema": sc,
            "schemaEncoding": "jsonschema",
        })
        self._local_map_channels[agent_id] = chan_id

    async def register_swarm_channels(self) -> None:
        """Register the two swarm-level channels (comms links + global map)."""
        if self.server is None:
            raise RuntimeError("Call start() before registering channels.")

        # --- Comms links ---
        sn, sc = _make_line_primitive_schema()
        self._comms_channel = await self.server.add_channel({
            "topic": "/swarm/comms_links",
            "encoding": "json",
            "schemaName": sn,
            "schema": sc,
            "schemaEncoding": "jsonschema",
        })

        # --- Global merged map ---
        sn, sc = _make_occupancy_grid_schema()
        self._global_map_channel = await self.server.add_channel({
            "topic": "/swarm/global_merged_map",
            "encoding": "json",
            "schemaName": sn,
            "schema": sc,
            "schemaEncoding": "jsonschema",
        })

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    async def publish_poses(self, agents: list, timestamp_ns: int) -> None:
        """Publish the pose of every agent."""
        for agent in agents:
            cid = self._pose_channels.get(agent.agent_id)
            if cid is None:
                continue
            msg = _build_pose_msg(agent, timestamp_ns)
            await self.server.send_message(
                cid, timestamp_ns, json.dumps(msg).encode("utf-8")
            )

    async def publish_local_maps(self, agents: list, timestamp_ns: int) -> None:
        """Publish each agent's local occupancy grid."""
        for agent in agents:
            cid = self._local_map_channels.get(agent.agent_id)
            if cid is None:
                continue
            msg = _build_occ_grid_msg(agent.local_map, timestamp_ns)
            await self.server.send_message(
                cid, timestamp_ns, json.dumps(msg).encode("utf-8")
            )

    async def publish_comms_links(
        self, links: list, agents: list, timestamp_ns: int
    ) -> None:
        """Publish active comms links as line primitives."""
        if self._comms_channel is None:
            return

        points = []
        for link in links:
            a = agents[link.agent_a_id]
            b = agents[link.agent_b_id]
            points.append({"x": a.x, "y": a.y, "z": 0.0})
            points.append({"x": b.x, "y": b.y, "z": 0.0})

        msg = {
            "type": 1,  # LINE_STRIP = 1 (each pair is its own strip)
            "points": points,
            "color": {"r": 0.2, "g": 0.8, "b": 0.2, "a": 0.7},
            "thickness": 0.15,
        }
        await self.server.send_message(
            self._comms_channel, timestamp_ns, json.dumps(msg).encode("utf-8")
        )

    async def publish_global_map(
        self, agents: list, timestamp_ns: int
    ) -> None:
        """Merge all agent maps and publish the combined occupancy grid."""
        if self._global_map_channel is None or not agents:
            return

        # Sum all agents' log-odds (Bayesian fusion)
        merged_log_odds = np.zeros(
            (agents[0].local_map.size, agents[0].local_map.size),
            dtype=np.float64,
        )
        for agent in agents:
            merged_log_odds += agent.local_map.log_odds

        # Clamp
        merged_log_odds = np.clip(
            merged_log_odds, OccupancyGrid.CLAMP_MIN, OccupancyGrid.CLAMP_MAX
        )

        # Build a temporary grid for export
        tmp = OccupancyGrid(agents[0].local_map.size)
        tmp.log_odds = merged_log_odds
        msg = _build_occ_grid_msg(tmp, timestamp_ns)
        await self.server.send_message(
            self._global_map_channel, timestamp_ns, json.dumps(msg).encode("utf-8")
        )


# ======================================================================
# Schema builders
# ======================================================================

def _make_pose_schema() -> tuple:
    """Return (schemaName, schema_json) for ``foxglove.PoseInFrame``."""
    name = "foxglove.PoseInFrame"
    schema = json.dumps({
        "type": "object",
        "properties": {
            "timestamp": {
                "type": "object",
                "properties": {
                    "sec": {"type": "integer"},
                    "nsec": {"type": "integer"},
                },
            },
            "frame_id": {"type": "string"},
            "pose": {
                "type": "object",
                "properties": {
                    "position": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                            "z": {"type": "number"},
                        },
                    },
                    "orientation": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                            "z": {"type": "number"},
                            "w": {"type": "number"},
                        },
                    },
                },
            },
        },
    })
    return name, schema


def _make_occupancy_grid_schema() -> tuple:
    """Return (schemaName, schema_json) for ``nav_msgs.OccupancyGrid``."""
    name = "nav_msgs/OccupancyGrid"
    schema = json.dumps({
        "type": "object",
        "properties": {
            "header": {
                "type": "object",
                "properties": {
                    "seq": {"type": "integer"},
                    "stamp": {
                        "type": "object",
                        "properties": {
                            "sec": {"type": "integer"},
                            "nsec": {"type": "integer"},
                        },
                    },
                    "frame_id": {"type": "string"},
                },
            },
            "info": {
                "type": "object",
                "properties": {
                    "map_load_time": {
                        "type": "object",
                        "properties": {
                            "sec": {"type": "integer"},
                            "nsec": {"type": "integer"},
                        },
                    },
                    "resolution": {"type": "number"},
                    "width": {"type": "integer"},
                    "height": {"type": "integer"},
                    "origin": {
                        "type": "object",
                        "properties": {
                            "position": {
                                "type": "object",
                                "properties": {
                                    "x": {"type": "number"},
                                    "y": {"type": "number"},
                                    "z": {"type": "number"},
                                },
                            },
                            "orientation": {
                                "type": "object",
                                "properties": {
                                    "x": {"type": "number"},
                                    "y": {"type": "number"},
                                    "z": {"type": "number"},
                                    "w": {"type": "number"},
                                },
                            },
                        },
                    },
                },
            },
            "data": {
                "type": "array",
                "items": {"type": "integer"},
            },
        },
    })
    return name, schema


def _make_line_primitive_schema() -> tuple:
    """Return (schemaName, schema_json) for ``foxglove.LinePrimitive``."""
    name = "foxglove.LinePrimitive"
    schema = json.dumps({
        "type": "object",
        "properties": {
            "type": {
                "type": "integer",
                "enum": [0, 1, 2],
            },
            "points": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "z": {"type": "number"},
                    },
                },
            },
            "color": {
                "type": "object",
                "properties": {
                    "r": {"type": "number"},
                    "g": {"type": "number"},
                    "b": {"type": "number"},
                    "a": {"type": "number"},
                },
            },
            "thickness": {"type": "number"},
        },
    })
    return name, schema


# ======================================================================
# Message builders
# ======================================================================

def _split_ns(timestamp_ns: int) -> tuple:
    """Split nanosecond timestamp into (sec, nsec)."""
    return (int(timestamp_ns // 1_000_000_000), int(timestamp_ns % 1_000_000_000))


def _build_pose_msg(agent, timestamp_ns: int) -> dict:
    """Build a PoseInFrame JSON message for *agent*."""
    sec, nsec = _split_ns(timestamp_ns)
    # Convert heading (theta) to quaternion (rotation around Z)
    half_theta = agent.theta / 2.0
    return {
        "timestamp": {"sec": sec, "nsec": nsec},
        "frame_id": "map",
        "pose": {
            "position": {"x": agent.x, "y": agent.y, "z": 0.0},
            "orientation": {
                "x": 0.0,
                "y": 0.0,
                "z": math.sin(half_theta),
                "w": math.cos(half_theta),
            },
        },
    }


def _build_occ_grid_msg(grid: OccupancyGrid, timestamp_ns: int) -> dict:
    """Build an OccupancyGrid JSON message for *grid*."""
    sec, nsec = _split_ns(timestamp_ns)
    occ = grid.get_occupancy_grid()
    return {
        "header": {
            "seq": 0,
            "stamp": {"sec": sec, "nsec": nsec},
            "frame_id": "map",
        },
        "info": {
            "map_load_time": {"sec": sec, "nsec": nsec},
            "resolution": 1.0,
            "width": grid.size,
            "height": grid.size,
            "origin": {
                "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
        },
        "data": occ.flatten().tolist(),
    }


# ------------------------------------------------------------------
# Quick smoke test (requires a running event loop)
# ------------------------------------------------------------------
if __name__ == "__main__":
    import asyncio

    async def _test():
        bridge = FoxgloveBridge("127.0.0.1", 18765)
        await bridge.start()
        await bridge.register_swarm_channels()
        await bridge.register_agent_channels(0)

        # Publish a single pose
        class _FakeAgent:
            agent_id = 0
            x = 5.0
            y = 5.0
            theta = 0.0

        ts = time.time_ns()
        await bridge.publish_poses([_FakeAgent()], ts)
        print("Pose published: OK")

        # Publish an empty map
        grid = OccupancyGrid(32)
        msg = _build_occ_grid_msg(grid, ts)
        assert len(msg["data"]) == 32 * 32
        print(f"OccGrid message: {len(msg['data'])} cells — OK")

        await bridge.stop()
        print("\nfoxglove_bridge.py: all checks passed.")

    asyncio.run(_test())
