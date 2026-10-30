"""
Colony: One - Level 1 Safe Pathfinding Solver
Finds valid 2-action sequences for Cyan Kernel avoiding Magenta threats.
"""

import argparse
from typing import Dict, List, Optional, Set, Tuple

Point = Tuple[int, int]

COL_NAMES: Dict[int, str] = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H'}
COL_INDEX: Dict[str, int] = {v: k for k, v in COL_NAMES.items()}


def parse_coord(s: str) -> Point:
    s = s.strip().upper()
    return (COL_INDEX[s[0]], int(s[1]))


def format_coord(pos: Point) -> str:
    return f"{COL_NAMES[pos[0]]}{pos[1]}"


def in_bounds(pos: Point) -> bool:
    return 1 <= pos[0] <= 8 and 1 <= pos[1] <= 8


def get_kernel_steps(pos: Point) -> List[Point]:
    steps = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nxt = (pos[0] + dx, pos[1] + dy)
            if in_bounds(nxt):
                steps.append(nxt)
    return steps


def ping_has_clear_diagonal(ping_pos: Point, target: Point, occupied: Set[Point]) -> bool:
    dx = target[0] - ping_pos[0]
    dy = target[1] - ping_pos[1]
    dist = abs(dx)
    if dist != abs(dy) or dist < 1 or dist > 2:
        return False

    if dist == 2:
        mid = (ping_pos[0] + dx // 2, ping_pos[1] + dy // 2)
        if mid in occupied:
            return False

    return True


def get_ping_orthogonal_moves(ping_pos: Point, occupied: Set[Point]) -> List[Point]:
    destinations = []
    for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        step1 = (ping_pos[0] + dx, ping_pos[1] + dy)
        if in_bounds(step1) and step1 not in occupied:
            destinations.append(step1)
            step2 = (ping_pos[0] + 2 * dx, ping_pos[1] + 2 * dy)
            if in_bounds(step2) and step2 not in occupied:
                destinations.append(step2)
    return destinations


def is_cyan_safe(cyan_pos: Point, mag_k_pos: Optional[Point], mag_p_pos: Optional[Point]) -> bool:
    if mag_k_pos is None:
        return True

    # Threat 1: Enemy Kernel within 2 king-moves
    if max(abs(cyan_pos[0] - mag_k_pos[0]), abs(cyan_pos[1] - mag_k_pos[1])) <= 2:
        return False

    # Threat 2: Enemy Ping diagonal capture (immediate or after 1 orthogonal move)
    if mag_p_pos is not None:
        occupied = {mag_k_pos}
        if ping_has_clear_diagonal(mag_p_pos, cyan_pos, occupied):
            return False

        occupied_with_cyan = {mag_k_pos, cyan_pos}
        for nxt_p in get_ping_orthogonal_moves(mag_p_pos, occupied_with_cyan):
            if ping_has_clear_diagonal(nxt_p, cyan_pos, {mag_k_pos}):
                return False

    return True


def solve_position(
    cyan_str: str,
    mag_k_str: str,
    mag_p_str: Optional[str] = None
) -> List[Tuple[Point, Point]]:
    cyan_start = parse_coord(cyan_str)
    mag_k = parse_coord(mag_k_str) if mag_k_str else None
    mag_p = parse_coord(mag_p_str) if mag_p_str else None

    safe_paths: List[Tuple[Point, Point]] = []

    for s1 in get_kernel_steps(cyan_start):
        k1 = None if (mag_k and s1 == mag_k) else mag_k
        p1 = None if (mag_p and s1 == mag_p) else mag_p

        if k1 is None:
            safe_paths.append((s1, s1))
            continue

        for s2 in get_kernel_steps(s1):
            k2 = None if (k1 and s2 == k1) else k1
            p2 = None if (p1 and s2 == p1) else p1

            if k2 is None:
                safe_paths.append((s1, s2))
                continue

            if is_cyan_safe(s2, k2, p2):
                safe_paths.append((s1, s2))

    return safe_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate safe 2-step Cyan Kernel moves for Colony: One Level 1."
    )
    parser.add_argument("cyan_kernel", help="Cyan Kernel square (e.g. E4)")
    parser.add_argument("magenta_kernel", help="Magenta Kernel square (e.g. H4)")
    parser.add_argument(
        "magenta_ping",
        nargs="?",
        default=None,
        help="Optional Magenta Ping square (e.g. A4)"
    )
    args = parser.parse_args()

    paths = solve_position(args.cyan_kernel, args.magenta_kernel, args.magenta_ping)

    print(f"State: Cyan Kernel: {args.cyan_kernel.upper()} | "
          f"Magenta Kernel: {args.magenta_kernel.upper()} | "
          f"Magenta Ping: {args.magenta_ping.upper() if args.magenta_ping else 'None'}")

    if not paths:
        print("No safe move sequence found.")
        return

    print(f"Safe sequences ({len(paths)} found):")
    for s1, s2 in paths:
        path_str = f"{args.cyan_kernel.upper()} -> {format_coord(s1)} -> {format_coord(s2)}"
        print(f"  {path_str}")

    best = paths[0]
    print(f"\nRecommended move: {args.cyan_kernel.upper()} -> {format_coord(best[0])} -> {format_coord(best[1])}")


if __name__ == '__main__':
    main()
