"""Serverless P2P gossip synchronisation via spatial chunking.

Every tick the CommsManager checks pairwise Euclidean distances between
all agents.  Pairs within communication range exchange their 8x8 chunk
CRC32 hashes.  Only chunks whose hashes differ are transferred, and both
sides perform a Bayesian log-odds merge.

Snapshot-before-merge prevents double-counting: each side keeps a copy
of the *other's* original chunk data before applying its own merge.
"""

import math
from typing import Optional


class CommLink:
    """Represents an active communication link between two agents."""

    __slots__ = ("agent_a_id", "agent_b_id")

    def __init__(self, a_id: int, b_id: int):
        self.agent_a_id = a_id
        self.agent_b_id = b_id


class CommsManager:
    """Manages P2P gossip syncs across the swarm each tick."""

    def __init__(self, comm_range: float = 15.0):
        self.comm_range = comm_range
        self.active_links: list = []

        # Stats
        self.total_syncs: int = 0
        self.total_chunks_exchanged: int = 0

    # ------------------------------------------------------------------
    # Tick entry point
    # ------------------------------------------------------------------

    def tick(self, agents: list) -> list:
        """Check all agent pairs and synchronise those within range.

        Returns the list of active ``CommLink`` objects for visualisation.
        """
        self.active_links.clear()
        n = len(agents)

        for i in range(n):
            for j in range(i + 1, n):
                a, b = agents[i], agents[j]
                dist = math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)
                if dist <= self.comm_range:
                    link = self._sync(a, b)
                    if link is not None:
                        self.active_links.append(link)

        return self.active_links

    # ------------------------------------------------------------------
    # Pairwise sync protocol
    # ------------------------------------------------------------------

    def _sync(self, a: "Agent", b: "Agent") -> Optional[CommLink]:
        """Execute the full spatial-delta sync between two agents.

        1. Exchange 8x8 chunk hash grids.
        2. For each chunk whose hashes differ, snapshot data from both
           sides, then each side merges the *other's* snapshot.
        3. Return a ``CommLink`` for visualisation.
        """
        hashes_a = a.local_map.get_all_chunk_hashes()
        hashes_b = b.local_map.get_all_chunk_hashes()

        diff_mask = hashes_a != hashes_b
        if not np_any(diff_mask):
            return CommLink(a.agent_id, b.agent_id)  # in range but identical maps

        n_chunks = a.local_map.num_chunks
        chunks_xfer = 0

        for cr in range(n_chunks):
            for cc in range(n_chunks):
                if not diff_mask[cr, cc]:
                    continue

                # Snapshot BEFORE merging — prevents double-counting
                data_a = a.local_map.get_chunk_data(cr, cc)
                data_b = b.local_map.get_chunk_data(cr, cc)

                # Each side merges the *other's* original data
                a.local_map.merge_chunk(cr, cc, data_b)
                b.local_map.merge_chunk(cr, cc, data_a)

                chunks_xfer += 1

        self.total_syncs += 1
        self.total_chunks_exchanged += chunks_xfer

        return CommLink(a.agent_id, b.agent_id)


# ------------------------------------------------------------------
# Tiny np.any helper to avoid importing numpy at module level in
# the type stub (it's already imported transitively via Agent)
# ------------------------------------------------------------------
def np_any(arr) -> bool:
    """Return True if any element of *arr* is truthy."""
    return bool(arr.any())


# ------------------------------------------------------------------
# Quick smoke test
# ------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np
    from .maze import Maze
    from .agent import Agent

    maze = Maze(128, seed=42)

    # Two agents close together — should sync
    a = Agent(0, 1.5, 1.5, 0.0, maze)
    b = Agent(1, 3.5, 1.5, 0.0, maze)

    # Give agent a some map data
    a.sense()

    # Verify they start with different maps
    ha = a.local_map.get_all_chunk_hashes()
    hb = b.local_map.get_all_chunk_hashes()
    assert np.any(ha != hb), "Agents should start with different maps"
    print("Initial hash diff: OK")

    # They're 2 cells apart (within default comm_range=15)
    mgr = CommsManager(comm_range=15.0)
    dist = a.distance_to(b)
    print(f"Distance: {dist:.1f} cells")
    assert dist <= mgr.comm_range

    links = mgr.tick([a, b])
    assert len(links) == 1, f"Expected 1 link, got {len(links)}"
    assert links[0].agent_a_id == 0
    assert links[0].agent_b_id == 1
    print("Comms link created: OK")

    # After sync, their maps should agree on previously differing chunks
    ha2 = a.local_map.get_all_chunk_hashes()
    hb2 = b.local_map.get_all_chunk_hashes()
    assert np.array_equal(ha2, hb2), "Maps should be identical after sync"
    print("Post-sync hash match: OK")

    # Agent b should now have the map data that agent a sensed
    assert b.local_map.known_fraction > 0.0
    print(f"Knowledge propagated: OK (b known fraction: {b.local_map.known_fraction:.4f})")

    # Two agents far apart — no sync
    c = Agent(2, 120.5, 120.5, 0.0, maze)
    links = mgr.tick([a, c])
    assert len(links) == 0, "Far-apart agents should not sync"
    print("Distance gating: OK")

    print(f"\nStats: {mgr.total_syncs} syncs, {mgr.total_chunks_exchanged} chunks exchanged")
    print("\ncomms.py: all checks passed.")
