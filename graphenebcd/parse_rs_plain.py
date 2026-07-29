#!/usr/bin/env python3
"""
Simple parser for FHI-aims output_rs_matrices plain output.

Reads:
  - rs_indices.out
  - rs_hamiltonian.out
  - rs_overlap.out

Reconstructs dense H(R) and S(R) matrices for each R cell index and saves:
  parsed_rs_matrices.npz
with arrays:
  - cell_indices: (n_cells, 3), integer lattice vectors
  - H: (n_cells, n_basis, n_basis)
  - S: (n_cells, n_basis, n_basis)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class RSIndexData:
    n_values: int
    n_cells: int
    n_basis: int
    cell_indices: np.ndarray
    start_idx: np.ndarray
    end_idx: np.ndarray
    col_idx: np.ndarray


def _read_nonempty_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def _parse_tagged_int(line: str, expected_tag: str) -> int:
    if ":" not in line:
        raise ValueError(f"Expected '{expected_tag}: ...' but got: {line}")
    tag, value = line.split(":", 1)
    if tag.strip() != expected_tag:
        raise ValueError(f"Expected tag '{expected_tag}', got '{tag.strip()}'")
    return int(value.strip())


def parse_rs_indices(path: Path) -> RSIndexData:
    lines = _read_nonempty_lines(path)
    i = 0

    n_values = _parse_tagged_int(lines[i], "n_hamiltonian_matrix_size")
    i += 1
    n_cells_raw = _parse_tagged_int(lines[i], "n_cells_in_hamiltonian")
    i += 1
    n_basis = _parse_tagged_int(lines[i], "n_basis")
    i += 1

    if lines[i] != "cell_index":
        raise ValueError(f"Expected 'cell_index', got: {lines[i]}")
    i += 1

    cell_list: list[list[int]] = []
    sentinel = [999999999, 999999999, 999999999]
    while i < len(lines):
        if lines[i] == "index_hamiltonian(1,:,:)":
            break
        parts = lines[i].split()
        if len(parts) == 3:
            nums = [int(x) for x in parts]
            if nums != sentinel:
                cell_list.append(nums)
            i += 1
            continue
        raise ValueError(f"Unexpected line while reading cell_index block: {lines[i]}")

    if len(cell_list) not in (n_cells_raw, n_cells_raw - 1):
        raise ValueError(
            f"Parsed {len(cell_list)} cell indices, expected {n_cells_raw} "
            f"or {n_cells_raw - 1} (if sentinel row is counted in header)"
        )
    cell_indices = np.array(cell_list, dtype=int)
    n_cells = len(cell_list)

    if lines[i] != "index_hamiltonian(1,:,:)":
        raise ValueError(f"Expected index_hamiltonian(1,:,:), got: {lines[i]}")
    i += 1

    start_idx_raw = np.zeros((n_cells_raw, n_basis), dtype=int)
    for c in range(n_cells_raw):
        parts = lines[i].split()
        if len(parts) != n_basis:
            raise ValueError(f"Expected {n_basis} entries in start_idx row, got: {len(parts)}")
        start_idx_raw[c] = [int(x) for x in parts]
        i += 1

    if lines[i] != "index_hamiltonian(2,:,:)":
        raise ValueError(f"Expected index_hamiltonian(2,:,:), got: {lines[i]}")
    i += 1

    end_idx_raw = np.zeros((n_cells_raw, n_basis), dtype=int)
    for c in range(n_cells_raw):
        parts = lines[i].split()
        if len(parts) != n_basis:
            raise ValueError(f"Expected {n_basis} entries in end_idx row, got: {len(parts)}")
        end_idx_raw[c] = [int(x) for x in parts]
        i += 1
    if n_cells_raw == n_cells + 1:
        # Some files count the 999999999 sentinel as an extra "cell" in headers.
        # In that case, the final index row is a dummy (zeros/-1) and is discarded.
        start_idx = start_idx_raw[:n_cells, :]
        end_idx = end_idx_raw[:n_cells, :]
    elif n_cells_raw == n_cells:
        start_idx = start_idx_raw
        end_idx = end_idx_raw
    else:
        raise ValueError(
            f"Inconsistent cell counts: n_cells_raw={n_cells_raw}, parsed={n_cells}"
        )


    if lines[i] != "column_index_hamiltonian":
        raise ValueError(f"Expected column_index_hamiltonian, got: {lines[i]}")
    i += 1

    col_idx = np.array([int(x) for x in lines[i:]], dtype=int)
    if len(col_idx) != n_values:
        raise ValueError(
            f"column_index_hamiltonian length ({len(col_idx)}) != "
            f"n_hamiltonian_matrix_size ({n_values})"
        )

    return RSIndexData(
        n_values=n_values,
        n_cells=n_cells,
        n_basis=n_basis,
        cell_indices=cell_indices,
        start_idx=start_idx,
        end_idx=end_idx,
        col_idx=col_idx,
    )


def parse_values(path: Path, n_expected: int) -> np.ndarray:
    lines = _read_nonempty_lines(path)
    values = np.array([float(x) for x in lines], dtype=float)
    if len(values) != n_expected:
        raise ValueError(f"{path.name} has {len(values)} values, expected {n_expected}")
    return values


def build_dense_matrices(index_data: RSIndexData, values: np.ndarray) -> np.ndarray:
    n_cells, n_basis = index_data.n_cells, index_data.n_basis
    mats = np.zeros((n_cells, n_basis, n_basis), dtype=float)

    for c in range(n_cells):
        for row in range(n_basis):
            start = index_data.start_idx[c, row]
            end = index_data.end_idx[c, row]
            if start <= 0 or end <= 0:
                continue

            # indices in file are 1-based and inclusive on both ends
            for p in range(start - 1, end):
                col = index_data.col_idx[p] - 1
                mats[c, row, col] = values[p]

    return mats


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse FHI-aims rs_* plain matrix files.")
    parser.add_argument("--indices", default="rs_indices.out", help="Path to rs_indices.out")
    parser.add_argument("--hamiltonian", default="rs_hamiltonian.out", help="Path to rs_hamiltonian.out")
    parser.add_argument("--overlap", default="rs_overlap.out", help="Path to rs_overlap.out")
    parser.add_argument("--output", default="parsed_rs_matrices.npz", help="Output .npz filename")
    args = parser.parse_args()

    idx = parse_rs_indices(Path(args.indices))
    h_values = parse_values(Path(args.hamiltonian), idx.n_values)
    s_values = parse_values(Path(args.overlap), idx.n_values)

    h_mats = build_dense_matrices(idx, h_values)
    s_mats = build_dense_matrices(idx, s_values)

    np.savez(
        args.output,
        cell_indices=idx.cell_indices,
        H=h_mats,
        S=s_mats,
    )

    print("Parsed real-space matrices successfully.")
    print(f"  n_cells  = {idx.n_cells}")
    print(f"  n_basis  = {idx.n_basis}")
    print(f"  n_values = {idx.n_values}")
    print(f"  output   = {args.output}")
    print("Array shapes:")
    print(f"  cell_indices: {idx.cell_indices.shape}")
    print(f"  H: {h_mats.shape}")
    print(f"  S: {s_mats.shape}")


if __name__ == "__main__":
    main()
