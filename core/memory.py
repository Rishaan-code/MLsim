from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import math


class MemLevel(Enum):
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
    HBM = "HBM"
    DRAM = "DRAM"


@dataclass
class CacheConfig:
    size_kb: float
    line_bytes: int
    associativity: int
    hit_latency_cycles: int
    bandwidth_gb_s: float
    miss_penalty_cycles: int

    @property
    def size_bytes(self) -> int:
        return int(self.size_kb * 1024)

    @property
    def num_sets(self) -> int:
        return max(1, self.size_bytes // (self.line_bytes * self.associativity))


@dataclass
class MemoryHierarchy:
    name: str
    levels: dict[MemLevel, CacheConfig]
    freq_ghz: float

    def latency_ns(self, level: MemLevel) -> float:
        cycles = self.levels[level].hit_latency_cycles
        return cycles / self.freq_ghz

    def effective_bandwidth(self, level: MemLevel) -> float:
        return self.levels[level].bandwidth_gb_s


@dataclass
class AccessPattern:
    """Describes how a workload accesses memory — stride, reuse, working set."""
    working_set_bytes: int
    stride_bytes: int
    reuse_distance: int        # how many unique bytes accessed between reuses
    sequential_fraction: float # 0=random, 1=fully sequential
    read_write_ratio: float    # fraction that are reads


@dataclass
class MemoryAccessResult:
    level: MemLevel
    latency_cycles: float
    bytes_transferred: int
    hit_rate: float
    bandwidth_utilization: float
    stall_cycles: float


class MemorySimulator:
    def __init__(self, hierarchy: MemoryHierarchy):
        self.h = hierarchy
        self._access_counts: dict[MemLevel, int] = {l: 0 for l in MemLevel}
        self._hit_counts:    dict[MemLevel, int] = {l: 0 for l in MemLevel}
        self._total_bytes:   dict[MemLevel, int] = {l: 0 for l in MemLevel}
        self._stall_cycles: float = 0.0

    def _level_order(self) -> list[MemLevel]:
        # Sort by bandwidth descending — fastest (L1) first, slowest (DRAM) last.
        # This avoids hardcoding a fixed order and works for any hierarchy config.
        return sorted(
            self.h.levels.keys(),
            key=lambda l: self.h.levels[l].bandwidth_gb_s,
            reverse=True,
        )

    def _cache_hit_probability(
        self,
        level: MemLevel,
        pattern: AccessPattern
    ) -> float:
        cfg = self.h.levels[level]
        ws  = pattern.working_set_bytes
        cap = cfg.size_bytes

        # Basic capacity model: if working set fits, high hit rate.
        # Partial fit uses a log-based degradation. Random access penalizes more.
        if ws <= 0:
            return 1.0

        capacity_ratio = cap / ws
        if capacity_ratio >= 1.0:
            base_hit = 0.97
        else:
            base_hit = 0.97 * (1 - math.exp(-3.0 * capacity_ratio))

        # Sequential access pattern improves effective hit rate.
        spatial_bonus = pattern.sequential_fraction * 0.03
        hit_prob = min(0.99, base_hit + spatial_bonus)

        # Conflict misses via simple set-associativity penalty.
        if cfg.associativity < 8 and capacity_ratio > 0.7:
            conflict_penalty = 0.02 * (1.0 / cfg.associativity) * (1 - pattern.sequential_fraction)
            hit_prob = max(0.0, hit_prob - conflict_penalty)

        return hit_prob

    def simulate_access(
        self,
        num_bytes: int,
        pattern: AccessPattern,
        vectorized: bool = True
    ) -> MemoryAccessResult:
        levels = self._level_order()
        remaining = num_bytes
        total_latency = 0.0
        total_stall   = 0.0
        served_level  = levels[-1]  # default to last level

        for level in levels:
            cfg = self.h.levels[level]
            hit_prob  = self._cache_hit_probability(level, pattern)
            hits      = int(remaining * hit_prob)
            misses    = remaining - hits

            self._access_counts[level] += 1
            self._hit_counts[level]    += hits
            self._total_bytes[level]   += hits

            if hits > 0:
                latency = cfg.hit_latency_cycles * (hits / num_bytes)
                total_latency += latency
                served_level   = level

            remaining = misses
            if remaining <= 0:
                break

        # Any bytes not served by the cache hierarchy come from the last level.
        if remaining > 0:
            last = levels[-1]
            cfg  = self.h.levels[last]
            total_latency += cfg.hit_latency_cycles
            total_stall   += cfg.miss_penalty_cycles * (remaining / cfg.line_bytes)
            self._total_bytes[last] += remaining

        # Bandwidth utilization at served level.
        cfg_served = self.h.levels[served_level]
        transfer_time_ns = (num_bytes / 1e9) / cfg_served.bandwidth_gb_s * 1e9
        transfer_cycles  = transfer_time_ns * self.h.freq_ghz
        bw_util = min(1.0, total_latency / max(1.0, transfer_cycles))

        self._stall_cycles += total_stall

        overall_hit_rate = 1.0 - (remaining / max(1, num_bytes))

        return MemoryAccessResult(
            level=served_level,
            latency_cycles=total_latency,
            bytes_transferred=num_bytes,
            hit_rate=overall_hit_rate,
            bandwidth_utilization=bw_util,
            stall_cycles=total_stall,
        )

    def summary(self) -> dict:
        out = {}
        for level in self._level_order():
            acc = self._access_counts[level]
            hit = self._hit_counts[level]
            out[level.value] = {
                "accesses":    acc,
                "hits":        hit,
                "hit_rate":    hit / acc if acc > 0 else 0.0,
                "bytes_total": self._total_bytes[level],
            }
        out["total_stall_cycles"] = self._stall_cycles
        return out

    def reset(self):
        for l in MemLevel:
            self._access_counts[l] = 0
            self._hit_counts[l]    = 0
            self._total_bytes[l]   = 0
        self._stall_cycles = 0.0


# ---------------------------------------------------------------------------
# Pre-built hardware configs
# ---------------------------------------------------------------------------

def cpu_desktop_hierarchy(freq_ghz: float = 3.5) -> MemoryHierarchy:
    """Approximates a modern x86 desktop (e.g. Ryzen 9 / Core i9 class)."""
    return MemoryHierarchy(
        name="CPU (x86 desktop)",
        freq_ghz=freq_ghz,
        levels={
            MemLevel.L1: CacheConfig(
                size_kb=48, line_bytes=64, associativity=8,
                hit_latency_cycles=4, bandwidth_gb_s=2000,
                miss_penalty_cycles=12,
            ),
            MemLevel.L2: CacheConfig(
                size_kb=512, line_bytes=64, associativity=8,
                hit_latency_cycles=12, bandwidth_gb_s=800,
                miss_penalty_cycles=40,
            ),
            MemLevel.L3: CacheConfig(
                size_kb=32768, line_bytes=64, associativity=16,
                hit_latency_cycles=40, bandwidth_gb_s=300,
                miss_penalty_cycles=150,
            ),
            MemLevel.DRAM: CacheConfig(
                size_kb=16 * 1024 * 1024, line_bytes=64, associativity=1,
                hit_latency_cycles=200, bandwidth_gb_s=50,
                miss_penalty_cycles=0,
            ),
        },
    )


def npu_hierarchy(freq_ghz: float = 1.25) -> MemoryHierarchy:
    """
    Approximates an NPU/TPU-style accelerator with large scratchpad SRAM
    and High Bandwidth Memory (HBM).
    """
    return MemoryHierarchy(
        name="NPU (systolic array + HBM)",
        freq_ghz=freq_ghz,
        levels={
            MemLevel.L1: CacheConfig(
                size_kb=256, line_bytes=128, associativity=4,
                hit_latency_cycles=2, bandwidth_gb_s=10000,
                miss_penalty_cycles=8,
            ),
            MemLevel.L2: CacheConfig(
                size_kb=8192, line_bytes=128, associativity=8,
                hit_latency_cycles=8, bandwidth_gb_s=4000,
                miss_penalty_cycles=24,
            ),
            MemLevel.HBM: CacheConfig(
                size_kb=16 * 1024 * 1024, line_bytes=128, associativity=1,
                hit_latency_cycles=50, bandwidth_gb_s=900,
                miss_penalty_cycles=0,
            ),
        },
    )


def mobile_hierarchy(freq_ghz: float = 2.4) -> MemoryHierarchy:
    """Approximates an ARM mobile SoC (e.g. A-series or Snapdragon class)."""
    return MemoryHierarchy(
        name="Mobile SoC (ARM)",
        freq_ghz=freq_ghz,
        levels={
            MemLevel.L1: CacheConfig(
                size_kb=64, line_bytes=64, associativity=4,
                hit_latency_cycles=3, bandwidth_gb_s=800,
                miss_penalty_cycles=10,
            ),
            MemLevel.L2: CacheConfig(
                size_kb=1024, line_bytes=64, associativity=8,
                hit_latency_cycles=15, bandwidth_gb_s=200,
                miss_penalty_cycles=50,
            ),
            MemLevel.DRAM: CacheConfig(
                size_kb=8 * 1024 * 1024, line_bytes=64, associativity=1,
                hit_latency_cycles=150, bandwidth_gb_s=34,
                miss_penalty_cycles=0,
            ),
        },
    )
