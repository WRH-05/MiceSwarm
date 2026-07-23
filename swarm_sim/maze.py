"""Ground truth maze generator using iterative DFS recursive backtracker.

Produces a perfect maze: exactly one path between any two free cells.
The maze is a 128x128 grid where 0 = free space, 100 = wall.
"""

import numpy as np
from typing import Optional


class Maze:
    """128x128 discrete grid maze with DFS generation and DDA raycasting."""

    FREE: int = 0
    WALL: int = 100

    def __init__(self, size: int = 128, seed: Optional[int] = None):
        if size < 3 or size % 2 != 0:
            raise ValueError(f"Maze size must be an even number >= 4, got {size}")
        self.size = size
        self.grid = np.full((size, size), self.WALL, dtype=np.int8)
        self._rng = np.random.RandomState(seed)
        self._generate()

    # ------------------------------------------------------------------
    # Maze generation (iterative DFS — no recursion depth issues)
    # ------------------------------------------------------------------

    def _generate(self) -> None:
        """Iterative DFS recursive backtracker on the logical cell grid.

        Logical cells sit at (2*i+1, 2*j+1).  The walls between them are
        the even-indexed rows/cols.  We start at (1,1), carve passages
        by removing the wall between the current cell and a random
        unvisited neighbour, then move to that neighbour.
        """
        H = self.size // 2  # number of logical cells per dimension (64)

        # visited mask on the *logical* grid
        visited = np.zeros((H, H), dtype=bool)

        # Start at logical (0, 0) → pixel (1, 1)
        stack = [(0, 0)]
        visited[0, 0] = True
        self.grid[1, 1] = self.FREE

        # Pre-compute neighbour offsets: (dr, dc, wall_dr, wall_dc)
        neighbours = [
            (-1, 0, -1, 0),  # up
            (1, 0, 1, 0),  # down
            (0, -1, 0, -1),  # left
            (0, 1, 0, 1),  # right
        ]

        while stack:
            cr, cc = stack[-1]  # current logical cell

            # Collect unvisited neighbours
            unvisited = []
            for dr, dc, wdr, wdc in neighbours:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < H and 0 <= nc < H and not visited[nr, nc]:
                    unvisited.append((nr, nc, wdr, wdc))

            if unvisited:
                nr, nc, wdr, wdc = tuple(unvisited[self._rng.randint(len(unvisited))])

                # Pixel coords
                py_cur = 2 * cr + 1
                px_cur = 2 * cc + 1
                py_next = 2 * nr + 1
                px_next = 2 * nc + 1

                # Remove the wall between them
                wall_y = py_cur + wdr
                wall_x = px_cur + wdc
                self.grid[wall_y, wall_x] = self.FREE

                # Carve the destination cell
                self.grid[py_next, px_next] = self.FREE

                visited[nr, nc] = True
                stack.append((nr, nc))
            else:
                stack.pop()  # backtrack

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def is_free(self, x: float, y: float) -> bool:
        """Return True if the cell at (int(x), int(y)) is free and in bounds."""
        ix, iy = int(x), int(y)
        if 0 <= ix < self.size and 0 <= iy < self.size:
            return self.grid[iy, ix] == self.FREE
        return False

    def get_random_free_cells(self, n: int, min_border_dist: int = 3) -> list:
        """Return *n* random free-cell centres as (x, y) float tuples.

        Cells within *min_border_dist* of the grid edge are excluded so
        agents don't start trapped against the outer wall.
        """
        free_ys, free_xs = np.where(self.grid == self.FREE)
        # Filter out cells too close to the border
        mask = (
            (free_xs >= min_border_dist)
            & (free_xs < self.size - min_border_dist)
            & (free_ys >= min_border_dist)
            & (free_ys < self.size - min_border_dist)
        )
        free_ys = free_ys[mask]
        free_xs = free_xs[mask]
        if len(free_ys) < n:
            raise RuntimeError(
                f"Only {len(free_ys)} free cells away from border, need {n}"
            )
        indices = self._rng.choice(len(free_ys), size=n, replace=False)
        return [
            (float(free_xs[i]) + 0.5, float(free_ys[i]) + 0.5)
            for i in indices
        ]

    # ------------------------------------------------------------------
    # DDA Raycasting
    # ------------------------------------------------------------------

    def raycast(
        self,
        x0: float,
        y0: float,
        theta: float,
        max_range: int = 10,
    ) -> tuple:
        """Cast a ray from (x0, y0) in direction *theta*.

        Uses the Digital Differential Analyzer (DDA) algorithm to step
        through grid cells.

        Returns
        -------
        (distance, hit_wall) : (float, bool)
            Distance to the first wall cell (or *max_range* if no hit),
            and whether a wall was struck.
        """
        dx = np.cos(theta)
        dy = np.sin(theta)

        # Avoid degenerate rays
        if abs(dx) < 1e-12:
            dx = 0.0
        if abs(dy) < 1e-12:
            dy = 0.0
        if dx == 0.0 and dy == 0.0:
            return (float(max_range), False)

        # Current cell
        cx, cy = int(x0), int(y0)

        # Step direction
        step_x = 1 if dx > 0 else -1 if dx < 0 else 0
        step_y = 1 if dy > 0 else -1 if dy < 0 else 0

        # Distance to next cell boundary
        if dx > 0:
            t_max_x = (cx + 1 - x0) / dx
        elif dx < 0:
            t_max_x = (cx - x0) / dx
        else:
            t_max_x = float("inf")

        if dy > 0:
            t_max_y = (cy + 1 - y0) / dy
        elif dy < 0:
            t_max_y = (cy - y0) / dy
        else:
            t_max_y = float("inf")

        # Step delta
        t_delta_x = abs(1.0 / dx) if dx != 0 else float("inf")
        t_delta_y = abs(1.0 / dy) if dy != 0 else float("inf")

        travelled = 0.0

        for _ in range(max_range * 2):  # safety cap
            # Check if current cell is out of bounds
            if not (0 <= cx < self.size and 0 <= cy < self.size):
                return (float(max_range), False)

            # Hit a wall?
            if self.grid[cy, cx] == self.WALL:
                return (min(travelled, float(max_range)), True)

            # Advance to next cell
            if t_max_x < t_max_y:
                travelled = t_max_x
                if travelled >= max_range:
                    return (float(max_range), False)
                cx += step_x
                t_max_x += t_delta_x
            else:
                travelled = t_max_y
                if travelled >= max_range:
                    return (float(max_range), False)
                cy += step_y
                t_max_y += t_delta_y

        return (float(max_range), False)

    # ------------------------------------------------------------------
    # Cell enumeration along a ray  (for occupancy updates)
    # ------------------------------------------------------------------

    def get_cells_along_ray(
        self,
        x0: float,
        y0: float,
        theta: float,
        max_range: int = 10,
    ) -> list:
        """Return all cells traversed by the ray, plus hit info.

        Returns a list of ``(x, y, is_last, is_wall)`` tuples.
        All cells except the last one are free space; the last cell is
        either the wall that stopped the ray or the final free cell at
        max_range.
        """
        dx = np.cos(theta)
        dy = np.sin(theta)
        if abs(dx) < 1e-12:
            dx = 0.0
        if abs(dy) < 1e-12:
            dy = 0.0
        if dx == 0.0 and dy == 0.0:
            return []

        cx, cy = int(x0), int(y0)
        step_x = 1 if dx > 0 else -1 if dx < 0 else 0
        step_y = 1 if dy > 0 else -1 if dy < 0 else 0

        if dx > 0:
            t_max_x = (cx + 1 - x0) / dx
        elif dx < 0:
            t_max_x = (cx - x0) / dx
        else:
            t_max_x = float("inf")

        if dy > 0:
            t_max_y = (cy + 1 - y0) / dy
        elif dy < 0:
            t_max_y = (cy - y0) / dy
        else:
            t_max_y = float("inf")

        t_delta_x = abs(1.0 / dx) if dx != 0 else float("inf")
        t_delta_y = abs(1.0 / dy) if dy != 0 else float("inf")

        cells = []

        for _ in range(max_range * 2):
            if not (0 <= cx < self.size and 0 <= cy < self.size):
                break

            is_wall = bool(self.grid[cy, cx] == self.WALL)

            if t_max_x < t_max_y:
                dist = t_max_x
            else:
                dist = t_max_y

            if dist >= max_range:
                # Final cell at max_range (free)
                if 0 <= cx < self.size and 0 <= cy < self.size:
                    cells.append((cx, cy, True, False))
                break

            if is_wall:
                cells.append((cx, cy, True, True))
                break

            cells.append((cx, cy, False, False))

            if t_max_x < t_max_y:
                cx += step_x
                t_max_x += t_delta_x
            else:
                cy += step_y
                t_max_y += t_delta_y

        return cells


# ------------------------------------------------------------------
# Quick smoke test
# ------------------------------------------------------------------
if __name__ == "__main__":
    m = Maze(128, seed=42)
    print(f"Maze shape: {m.grid.shape}")
    print(f"Free cells:  {np.sum(m.grid == Maze.FREE)}")
    print(f"Wall cells:  {np.sum(m.grid == Maze.WALL)}")
    print(f"Border all walls: {np.all(m.grid[0, :] == Maze.WALL)}")

    # Test raycasting
    dist, hit = m.raycast(1.5, 1.5, 0.0, max_range=10)
    print(f"Ray east from (1.5, 1.5): dist={dist:.2f}, hit={hit}")

    cells = m.get_cells_along_ray(1.5, 1.5, 0.0, max_range=10)
    print(f"Cells along ray: {len(cells)} cells, last is wall={cells[-1][3] if cells else 'N/A'}")

    # Test random free cells
    starts = m.get_random_free_cells(4)
    print(f"Random starts: {starts}")

    print("maze.py: all checks passed.")
