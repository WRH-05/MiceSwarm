"""Autonomous robot agent ("mouse") with sensing, odometry, and movement.

Each agent holds a continuous pose, a local log-odds map, and a target
frontier.  Sensing uses 8-ray raycasting against the ground-truth maze.
Movement uses a proportional controller with cumulative Gaussian noise.
"""

import math
import numpy as np
from typing import Optional

from .maze import Maze
from .map_grid import OccupancyGrid


class Agent:
    """A single autonomous mapping robot."""

    SENSOR_RANGE: int = 10
    LINEAR_VEL: float = 2.0  # cells per second
    ANGULAR_VEL: float = 3.0  # rad per second

    # The 8 ray directions (cardinal + intercardinal) relative to heading
    RAY_ANGLES: list = [
        0.0,
        math.pi / 4,
        math.pi / 2,
        3 * math.pi / 4,
        math.pi,
        5 * math.pi / 4,
        3 * math.pi / 2,
        7 * math.pi / 4,
    ]

    def __init__(
        self,
        agent_id: int,
        start_x: float,
        start_y: float,
        start_theta: float,
        maze: Maze,
        noise_std: float = 0.05,
    ):
        self.agent_id = agent_id
        self.x = float(start_x)
        self.y = float(start_y)
        self.theta = float(start_theta)
        self.maze = maze
        self.noise_std = noise_std

        self.local_map = OccupancyGrid(maze.size)
        self.target_frontier: Optional[tuple] = None

        # Per-agent RNG for reproducible noise
        self._rng = np.random.RandomState(agent_id * 12345 + hash(int(start_x * 1000)))

        # Track stuck state
        self._prev_position: Optional[tuple] = None
        self._stuck_ticks: int = 0
        self._ticks_since_target_change: int = 0

    # ------------------------------------------------------------------
    # Sensing
    # ------------------------------------------------------------------

    def sense(self) -> None:
        """Cast 8 rays and update the local log-odds map."""
        for rel_angle in self.RAY_ANGLES:
            theta = self.theta + rel_angle
            cells = self.maze.get_cells_along_ray(
                self.x, self.y, theta, self.SENSOR_RANGE
            )
            for cx, cy, is_last, is_wall in cells:
                if is_last and is_wall:
                    self.local_map.update_cell(cx, cy, occupied=True)
                else:
                    self.local_map.update_cell(cx, cy, occupied=False)

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    def set_target(self, target: Optional[tuple]) -> None:
        """Assign a new frontier target."""
        if target != self.target_frontier:
            self.target_frontier = target
            self._ticks_since_target_change = 0

    def move(self, dt: float) -> None:
        """Move toward *target_frontier*, or wall-follow if no target is set.

        Two modes:
        - **Targeted**: move directly toward the frontier centroid.
        - **Exploring**: wall-follow (right-hand rule) to discover new areas
          when no frontier target is available or the agent is stuck.
        """
        self._ticks_since_target_change += 1

        if self.target_frontier is not None:
            self._move_toward_target(dt)
        else:
            self._move_explore(dt)

        # Odometry noise
        self.x += self._rng.normal(0.0, self.noise_std * dt)
        self.y += self._rng.normal(0.0, self.noise_std * dt)
        self.theta += self._rng.normal(0.0, self.noise_std * 0.5 * dt)

        # Clamp to grid bounds
        self.x = max(0.5, min(self.maze.size - 1.5, self.x))
        self.y = max(0.5, min(self.maze.size - 1.5, self.y))

        # Detect prolonged stagnation
        current = (self.x, self.y)
        if self._prev_position is not None:
            d = math.sqrt(
                (current[0] - self._prev_position[0]) ** 2
                + (current[1] - self._prev_position[1]) ** 2
            )
            if d < 0.005:
                self._stuck_ticks += 1
            else:
                self._stuck_ticks = max(0, self._stuck_ticks - 1)
        self._prev_position = current

        # Force re-plan when stuck for too long
        if self._stuck_ticks > 40:  # 4 seconds at 10 Hz
            self.target_frontier = None
            self._stuck_ticks = 0
            # Random heading perturbation to break out of loops
            self.theta += self._rng.uniform(-math.pi / 2, math.pi / 2)

    # ------------------------------------------------------------------
    # Movement sub-modes
    # ------------------------------------------------------------------

    def _move_toward_target(self, dt: float) -> None:
        """Direct movement toward the current frontier target."""
        tx, ty = self.target_frontier
        dx = tx - self.x
        dy = ty - self.y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < 0.5:  # arrived — clear target so planner picks a new one
            self.target_frontier = None
            return

        # Normalise direction
        udx = dx / dist
        udy = dy / dist

        speed = self.LINEAR_VEL * dt
        new_x = self.x + udx * speed
        new_y = self.y + udy * speed

        if self.maze.is_free(new_x, new_y):
            self.x = new_x
            self.y = new_y
            self.theta = math.atan2(udy, udx)
        else:
            # Try axis-aligned sliding
            slid = False
            if self.maze.is_free(new_x, self.y):
                self.x = new_x
                slid = True
            if self.maze.is_free(self.x, new_y):
                self.y = new_y
                slid = True
            if slid:
                self.theta = math.atan2(udy, udx)
            else:
                # Blocked — try turning toward a free direction
                self.target_frontier = None  # give up on this target
                self._turn_to_free()

    def _move_explore(self, dt: float) -> None:
        """Wall-following exploration when no target is assigned.

        Uses a right-hand-rule heuristic: prefer moving forward, then
        try turning right, then left, then reverse.  This naturally
        follows corridors and explores new areas.
        """
        speed = self.LINEAR_VEL * dt

        # Candidate moves in order of preference: forward, right, left, back
        candidates = [
            (math.cos(self.theta), math.sin(self.theta)),                     # forward
            (math.cos(self.theta + math.pi / 2), math.sin(self.theta + math.pi / 2)),   # right
            (math.cos(self.theta - math.pi / 2), math.sin(self.theta - math.pi / 2)),   # left
            (-math.cos(self.theta), -math.sin(self.theta)),                  # back
        ]
        heading_updates = [self.theta, self.theta + math.pi / 2,
                           self.theta - math.pi / 2, self.theta + math.pi]

        for (cdx, cdy), new_heading in zip(candidates, heading_updates):
            nx = self.x + cdx * speed
            ny = self.y + cdy * speed
            if self.maze.is_free(nx, ny):
                self.x = nx
                self.y = ny
                self.theta = math.atan2(
                    math.sin(new_heading), math.cos(new_heading)
                )
                return

        # Completely surrounded — shouldn't happen in a valid maze
        self.theta += self._rng.uniform(-math.pi, math.pi)

    def _turn_to_free(self) -> None:
        """Turn toward the first free direction (scanning clockwise)."""
        for angle_offset in [0, math.pi / 2, -math.pi / 2, math.pi]:
            test_theta = self.theta + angle_offset
            tx = self.x + math.cos(test_theta)
            ty = self.y + math.sin(test_theta)
            if self.maze.is_free(tx, ty):
                self.theta = math.atan2(
                    math.sin(test_theta), math.cos(test_theta)
                )
                return

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def distance_to(self, other: "Agent") -> float:
        """Euclidean distance to another agent."""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    @property
    def position(self) -> tuple:
        """Return (x, y) float tuple."""
        return (self.x, self.y)


# ------------------------------------------------------------------
# Quick smoke test
# ------------------------------------------------------------------
if __name__ == "__main__":
    from .maze import Maze

    maze = Maze(128, seed=42)
    agent = Agent(0, 1.5, 1.5, 0.0, maze, noise_std=0.02)

    # Should start in a free cell
    assert maze.is_free(agent.x, agent.y)
    print("Start position: OK (in free cell)")

    # Sense — should populate local map
    assert agent.local_map.known_fraction == 0.0
    agent.sense()
    assert agent.local_map.known_fraction > 0.0
    print(f"Sensing:        OK (known fraction: {agent.local_map.known_fraction:.4f})")

    # Move toward a target
    agent.set_target((10.5, 10.5))
    for _ in range(10):
        agent.move(0.1)
    moved = math.sqrt((agent.x - 1.5) ** 2 + (agent.y - 1.5) ** 2)
    assert moved > 0.1, f"Agent barely moved: {moved}"
    print(f"Movement:       OK (moved {moved:.2f} cells in 1s)")

    print("\nagent.py: all checks passed.")
