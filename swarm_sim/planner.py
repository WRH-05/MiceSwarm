"""Frontier-based exploration with utility discounting.

Identifies frontier cells (free cells adjacent to unknown), clusters them
into contiguous regions, and selects the best target using a utility
function that discounts frontiers already targeted by nearby peers.
"""

import math
import numpy as np
from scipy import ndimage
from typing import Optional


class FrontierCluster:
    """A contiguous cluster of frontier cells."""

    def __init__(self, cells: list):
        self.cells = cells  # list of (x, y)
        ys = [c[1] for c in cells]
        xs = [c[0] for c in cells]
        self.centroid = (float(np.mean(xs)), float(np.mean(ys)))
        self.size = len(cells)

    @property
    def center(self) -> tuple:
        """(x, y) centroid of the cluster."""
        return self.centroid


class FrontierPlanner:
    """Stateless frontier detection and utility-based target selection."""

    # Penalty applied when a nearby peer is targeting the same frontier
    PENALTY: float = 5.0

    # Ticks to remember a visited location before allowing re-selection
    VISIT_MEMORY: int = 30

    def __init__(self, epsilon: float = 1e-3):
        self.epsilon = epsilon
        # Per-agent visited centroids: {agent_id: [(cx, cy, remaining_ticks), ...]}
        self._visited: dict[int, list] = {}
        # Track which agent is assigned to which frontier to avoid double-booking
        self._assignments: dict[int, tuple] = {}  # agent_id → (cx, cy)

    # ------------------------------------------------------------------
    # Frontier detection
    # ------------------------------------------------------------------

    def find_frontiers(self, occ_grid: np.ndarray) -> list:
        """Find and cluster frontier cells in an occupancy grid.

        A frontier cell is a free cell (0) that is adjacent (8-connectivity)
        to at least one unknown cell (-1).

        Parameters
        ----------
        occ_grid : np.ndarray (int8), shape (H, W)
            Occupancy grid with values -1 (unknown), 0 (free), 100 (occupied).

        Returns
        -------
        list[FrontierCluster]
        """
        # Cells adjacent to unknown
        unknown_mask = occ_grid == -1
        free_mask = occ_grid == 0

        if not np.any(unknown_mask) or not np.any(free_mask):
            return []

        # Dilate unknown to find cells adjacent to unknown
        structure = np.ones((3, 3), dtype=bool)
        adj_to_unknown = ndimage.binary_dilation(unknown_mask, structure=structure)

        # Frontier = free AND adjacent to unknown
        frontier_mask = free_mask & adj_to_unknown

        if not np.any(frontier_mask):
            return []

        # Cluster connected components
        labeled, num_labels = ndimage.label(frontier_mask, structure=structure)

        clusters = []
        for lbl in range(1, num_labels + 1):
            ys, xs = np.where(labeled == lbl)
            cells = [(int(x), int(y)) for x, y in zip(xs, ys)]
            clusters.append(FrontierCluster(cells))

        return clusters

    # ------------------------------------------------------------------
    # Utility computation
    # ------------------------------------------------------------------

    def compute_utility(
        self,
        agent_x: float,
        agent_y: float,
        frontier: FrontierCluster,
        peers: list,
        comm_range: float,
    ) -> float:
        """Compute the discounted utility of a frontier for an agent.

        U(F_k) = 1 / (dist(agent, F_k) + ε)  -  Σ penalty_j

        where penalty_j is applied for each peer within *comm_range*
        that is already targeting a point near this frontier's centroid.
        """
        cx, cy = frontier.center
        dist = math.sqrt((agent_x - cx) ** 2 + (agent_y - cy) ** 2)
        utility = 1.0 / (dist + self.epsilon)

        # Penalise if nearby peers are already targeting this frontier
        for peer in peers:
            if peer is None:
                continue
            px, py, ptarget = peer
            d_peer = math.sqrt((px - cx) ** 2 + (py - cy) ** 2)
            if d_peer <= comm_range and ptarget is not None:
                # Peer is close to this frontier and has a target —
                # check if peer's target is within 5 cells of centroid
                tx, ty = ptarget
                if math.sqrt((tx - cx) ** 2 + (ty - cy) ** 2) < 5.0:
                    utility -= self.PENALTY

        return utility

    def select_target(
        self,
        agent_id: int,
        agent_x: float,
        agent_y: float,
        frontiers: list,
        peers: list,
        comm_range: float,
        current_target: Optional[tuple] = None,
    ) -> Optional[tuple]:
        """Return the (x, y) centre of the best frontier, or None.

        *peers* is a list of ``(px, py, target)`` tuples for all other
        agents (target is ``(tx, ty)`` or ``None``).

        Tracks recently visited centroids to avoid re-selecting
        unreachable targets, and assigns one agent per frontier.
        """
        if not frontiers:
            return None

        # Age-out visited entries
        self._tick_visited()

        best_utility = -float("inf")
        best_target = None

        for f in frontiers:
            cx, cy = f.center

            # Skip if this agent recently visited this centroid
            if self._is_recently_visited(agent_id, cx, cy):
                continue

            # Skip if another agent is already assigned to this frontier
            if self._is_assigned_to_other(agent_id, cx, cy):
                continue

            u = self.compute_utility(agent_x, agent_y, f, peers, comm_range)
            # Bonus for sticking with current target (avoids oscillation)
            if current_target is not None:
                d = math.sqrt(
                    (cx - current_target[0]) ** 2 + (cy - current_target[1]) ** 2
                )
                if d < 3.0:
                    u += 2.0
            if u > best_utility:
                best_utility = u
                best_target = (cx, cy)

        # If all utilities are deeply negative or no valid target
        if best_target is None or best_utility <= -self.PENALTY:
            return None

        # If the agent already has a target and it's different from the new
        # best, mark the old one as visited
        if current_target is not None and current_target != best_target:
            self._mark_visited(agent_id, current_target[0], current_target[1])

        # Record assignment
        self._assignments[agent_id] = best_target

        return best_target

    # ------------------------------------------------------------------
    # Visited-tracking helpers
    # ------------------------------------------------------------------

    def _tick_visited(self) -> None:
        """Decrement TTL on all visited entries."""
        for agent_id in list(self._visited.keys()):
            self._visited[agent_id] = [
                (cx, cy, ttl - 1)
                for cx, cy, ttl in self._visited[agent_id]
                if ttl > 1
            ]
            if not self._visited[agent_id]:
                del self._visited[agent_id]

    def _mark_visited(self, agent_id: int, cx: float, cy: float) -> None:
        """Record that *agent_id* has attempted to reach (cx, cy)."""
        if agent_id not in self._visited:
            self._visited[agent_id] = []
        self._visited[agent_id].append((cx, cy, self.VISIT_MEMORY))

    def _is_recently_visited(self, agent_id: int, cx: float, cy: float) -> bool:
        """Return True if (cx, cy) was recently visited by *agent_id*."""
        for vx, vy, _ in self._visited.get(agent_id, []):
            if math.sqrt((cx - vx) ** 2 + (cy - vy) ** 2) < 1.0:
                return True
        return False

    def _is_assigned_to_other(self, agent_id: int, cx: float, cy: float) -> bool:
        """Return True if another agent is already targeting this frontier."""
        for aid, (ax, ay) in self._assignments.items():
            if aid == agent_id:
                continue
            if math.sqrt((cx - ax) ** 2 + (cy - ay) ** 2) < 3.0:
                return True
        return False


# ------------------------------------------------------------------
# Quick smoke test
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny artificial occupancy grid
    # 0=free, 100=occupied, -1=unknown
    occ = np.full((10, 10), -1, dtype=np.int8)
    occ[3:8, 3:8] = 0  # known free region in the middle
    occ[5, 5] = 100  # a wall in the middle of the free area
    # This should create frontiers around the border of the free region

    planner = FrontierPlanner()
    frontiers = planner.find_frontiers(occ)
    print(f"Found {len(frontiers)} frontier clusters")
    for i, f in enumerate(frontiers):
        print(f"  Cluster {i}: {f.size} cells, center=({f.center[0]:.1f}, {f.center[1]:.1f})")
    assert len(frontiers) > 0, "Should find at least one frontier"
    print("Frontier detection: OK")

    # Test utility
    agent_x, agent_y = 5.0, 7.0
    peers = []  # no peers
    u = planner.compute_utility(agent_x, agent_y, frontiers[0], peers, comm_range=5.0)
    print(f"Utility (no peers): {u:.3f}")
    assert u > 0, f"Utility should be positive, got {u}"

    # Test with a peer targeting the same frontier
    cx, cy = frontiers[0].center
    # Place a peer close to the frontier, already targeting it
    peers_with_target = [(cx + 1.0, cy + 1.0, (cx, cy))]
    u2 = planner.compute_utility(agent_x, agent_y, frontiers[0], peers_with_target, comm_range=5.0)
    print(f"Utility (with competing peer): {u2:.3f}")
    assert u2 < u, "Utility should decrease when peer is targeting same frontier"
    print("Utility discounting: OK")

    # Test selection
    target = planner.select_target(0, agent_x, agent_y, frontiers, peers, comm_range=5.0)
    assert target is not None
    print(f"Selected target: ({target[0]:.1f}, {target[1]:.1f})")
    print("Target selection: OK")

    # Test no frontiers
    occ_all_known = np.zeros((10, 10), dtype=np.int8)  # all free, no unknown
    assert planner.find_frontiers(occ_all_known) == []
    print("Empty frontier:   OK")

    print("\nplanner.py: all checks passed.")
