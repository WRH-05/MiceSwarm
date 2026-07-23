"""Log-Odds Occupancy Grid with spatial chunking and Bayesian merge.

Internal representation uses log-odds (float64).  Public export converts
to the standard Occupancy Grid triplet: -1 = Unknown, 0 = Free, 100 = Occupied.

Spatial chunking partitions the grid into 16x16 chunks (8x8 total for a
128x128 grid).  Each chunk carries a CRC32 hash for efficient P2P
delta detection.
"""

import zlib
import numpy as np
from typing import Optional


class OccupancyGrid:
    """Log-odds occupancy grid with spatial chunking & CRC32 versioning."""

    # Log-odds update deltas (moderate confidence per observation)
    L_OCCUPIED: float = 0.85
    L_FREE: float = -0.85
    L0: float = 0.0  # prior: p=0.5 → log(1)=0

    # Clamping to prevent overflow
    CLAMP_MIN: float = -50.0
    CLAMP_MAX: float = 50.0

    def __init__(self, size: int = 128, chunk_size: int = 16):
        if size % chunk_size != 0:
            raise ValueError(
                f"Grid size {size} must be a multiple of chunk_size {chunk_size}"
            )
        self.size = size
        self.chunk_size = chunk_size
        self.num_chunks = size // chunk_size  # 8 for 128/16

        # Core data: log-odds, initialised to prior L0 = 0
        self.log_odds = np.full((size, size), self.L0, dtype=np.float64)

    # ------------------------------------------------------------------
    # Cell updates
    # ------------------------------------------------------------------

    def update_cell(self, x: int, y: int, occupied: bool) -> None:
        """Bayesian update for a single cell."""
        if 0 <= x < self.size and 0 <= y < self.size:
            delta = self.L_OCCUPIED if occupied else self.L_FREE
            self.log_odds[y, x] = np.clip(
                self.log_odds[y, x] + delta, self.CLAMP_MIN, self.CLAMP_MAX
            )

    def update_cells(self, cells: list, occupied: bool) -> None:
        """Batch-update multiple cells at once."""
        delta = self.L_OCCUPIED if occupied else self.L_FREE
        for x, y in cells:
            if 0 <= x < self.size and 0 <= y < self.size:
                self.log_odds[y, x] = np.clip(
                    self.log_odds[y, x] + delta, self.CLAMP_MIN, self.CLAMP_MAX
                )

    # ------------------------------------------------------------------
    # Occupancy-grid export
    # ------------------------------------------------------------------

    def get_occupancy_grid(self) -> np.ndarray:
        """Convert log-odds to the standard -1/0/100 representation.

        Returns
        -------
        np.ndarray (int8), shape (size, size)
            -1 = Unknown (log-odds == 0)
             0 = Free    (log-odds < 0)
           100 = Occupied (log-odds > 0)
        """
        occ = np.full((self.size, self.size), -1, dtype=np.int8)
        occ[self.log_odds < 0.0] = 0
        occ[self.log_odds > 0.0] = 100
        return occ

    # ------------------------------------------------------------------
    # Spatial chunking
    # ------------------------------------------------------------------

    def _chunk_slice(self, cr: int, cc: int):
        """Return the (y_slice, x_slice) for chunk (cr, cc)."""
        r0 = cr * self.chunk_size
        c0 = cc * self.chunk_size
        return (
            slice(r0, r0 + self.chunk_size),
            slice(c0, c0 + self.chunk_size),
        )

    def get_chunk_hash(self, cr: int, cc: int) -> int:
        """CRC32 of the log-odds data in chunk (cr, cc)."""
        rs, cs = self._chunk_slice(cr, cc)
        data = self.log_odds[rs, cs]
        return zlib.crc32(data.tobytes()) & 0xFFFFFFFF

    def get_all_chunk_hashes(self) -> np.ndarray:
        """Return an (8, 8) uint32 array of chunk CRC32 hashes."""
        hashes = np.zeros((self.num_chunks, self.num_chunks), dtype=np.uint32)
        for cr in range(self.num_chunks):
            for cc in range(self.num_chunks):
                hashes[cr, cc] = self.get_chunk_hash(cr, cc)
        return hashes

    def get_chunk_data(self, cr: int, cc: int) -> np.ndarray:
        """Return a **copy** of the log-odds data for chunk (cr, cc)."""
        rs, cs = self._chunk_slice(cr, cc)
        return self.log_odds[rs, cs].copy()

    def set_chunk_data(self, cr: int, cc: int, data: np.ndarray) -> None:
        """Overwrite chunk (cr, cc) with *data*."""
        rs, cs = self._chunk_slice(cr, cc)
        self.log_odds[rs, cs] = data

    # ------------------------------------------------------------------
    # Bayesian merge
    # ------------------------------------------------------------------

    def merge_chunk(self, cr: int, cc: int, peer_data: np.ndarray) -> None:
        """Bayesian merge for one chunk.

        L_new = L_self + L_peer - L0  (L0 = 0, so simple addition).
        The caller is responsible for snapshotting *peer_data* before
        merging to avoid double-counting in bidirectional syncs.
        """
        rs, cs = self._chunk_slice(cr, cc)
        merged = self.log_odds[rs, cs] + peer_data - self.L0
        merged = np.clip(merged, self.CLAMP_MIN, self.CLAMP_MAX)
        self.log_odds[rs, cs] = merged

    def merge_full(self, other: "OccupancyGrid") -> None:
        """Full-grid Bayesian merge with another OccupancyGrid."""
        merged = self.log_odds + other.log_odds - self.L0
        self.log_odds = np.clip(merged, self.CLAMP_MIN, self.CLAMP_MAX)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def known_fraction(self) -> float:
        """Fraction of cells with a non-zero log-odds (i.e. observed)."""
        return float(np.count_nonzero(self.log_odds)) / (self.size * self.size)


# ------------------------------------------------------------------
# Quick smoke test
# ------------------------------------------------------------------
if __name__ == "__main__":
    g = OccupancyGrid(128, chunk_size=16)
    assert g.size == 128
    assert g.num_chunks == 8
    assert np.all(g.log_odds == OccupancyGrid.L0)
    print("Initialisation: OK")

    # Update a single cell
    g.update_cell(10, 10, occupied=True)
    assert g.log_odds[10, 10] == OccupancyGrid.L_OCCUPIED
    g.update_cell(10, 10, occupied=False)
    assert g.log_odds[10, 10] == 0.0  # back to prior
    print("Cell update:   OK")

    # Clamping
    for _ in range(100):
        g.update_cell(5, 5, occupied=True)
    assert g.log_odds[5, 5] == OccupancyGrid.CLAMP_MAX
    print("Clamping:      OK")

    # Occupancy grid export
    occ = g.get_occupancy_grid()
    assert occ[10, 10] == -1  # unknown
    assert occ[5, 5] == 100  # occupied
    g.update_cell(6, 6, occupied=False)
    assert g.get_occupancy_grid()[6, 6] == 0  # free
    print("Occ export:    OK")

    # Chunk hashing
    h1 = g.get_chunk_hash(0, 0)
    h2 = g.get_chunk_hash(0, 0)
    assert h1 == h2
    assert h1 != 0  # should be non-zero after updates
    print("Chunk hash:    OK (deterministic, non-zero)")

    # Hash grid
    hg = g.get_all_chunk_hashes()
    assert hg.shape == (8, 8)
    assert hg.dtype == np.uint32
    print("Hash grid:     OK")

    # Chunk get/set
    data = g.get_chunk_data(0, 0)
    assert data.shape == (16, 16)
    data2 = data.copy()
    data2[0, 0] = 99.0
    g.set_chunk_data(0, 0, data2)
    assert g.log_odds[0, 0] == 99.0
    g.set_chunk_data(0, 0, data)  # restore
    print("Chunk get/set: OK")

    # Bayesian merge — use fresh grid so we know the exact state
    ga = OccupancyGrid(128, chunk_size=16)
    gb = OccupancyGrid(128, chunk_size=16)
    ga.update_cell(10, 10, occupied=True)   # ga: +0.85
    ga.update_cell(10, 10, occupied=True)   # ga: +0.85 → 1.70
    gb.update_cell(10, 10, occupied=True)   # gb: +0.85
    peer_data = gb.get_chunk_data(0, 0)
    ga.merge_chunk(0, 0, peer_data)
    # ga had 2 occ obs (1.70), gb had 1 (0.85) → merged = 2.55
    assert abs(ga.log_odds[10, 10] - 2.55) < 1e-12, f"{ga.log_odds[10,10]} != 2.55"
    print("Bayesian merge: OK")

    # Known fraction
    print(f"Known fraction: {g.known_fraction:.4f}")

    print("\nmap_grid.py: all checks passed.")
