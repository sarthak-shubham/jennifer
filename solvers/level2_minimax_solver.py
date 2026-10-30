"""
Colony: One - Level 2 Minimax Adversarial Solver
Computes forced-win sequences using game tree search with iterative deepening
and transposition memoization under 2-action turns and bodyguard rules.
"""

import argparse
from typing import Dict, List, Optional, Set, Tuple

Point = Tuple[int, int]

COL_NAMES: Dict[int, str] = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H'}
COL_INDEX: Dict[str, int] = {v: k for k, v in COL_NAMES.items()}


def parse_coord(s: str) -> Point:
    s = s.strip().upper()
    return (COL_INDEX[s[0]], int(s[1]))


def fmt(pos: Optional[Point]) -> str:
    if pos is None:
        return "None"
    return f"{COL_NAMES[pos[0]]}{pos[1]}"


def in_bounds(pos: Point) -> bool:
    return 1 <= pos[0] <= 8 and 1 <= pos[1] <= 8


def chebyshev(p1: Point, p2: Point) -> int:
    return max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))


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


def ping_can_capture_target(ping_pos: Point, target: Point, occupied: Set[Point]) -> bool:
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


def get_ping_moves(ping_pos: Point, occupied: Set[Point]) -> List[Point]:
    moves = []
    for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        s1 = (ping_pos[0] + dx, ping_pos[1] + dy)
        if in_bounds(s1) and s1 not in occupied:
            moves.append(s1)
            s2 = (ping_pos[0] + 2 * dx, ping_pos[1] + 2 * dy)
            if in_bounds(s2) and s2 not in occupied:
                moves.append(s2)
    return moves


def has_bodyguard(kernel_pos: Optional[Point], ping_pos: Optional[Point]) -> bool:
    if ping_pos is None or kernel_pos is None:
        return False
    return chebyshev(kernel_pos, ping_pos) == 1


def cyan_single_actions(
    ck: Point,
    cp: Optional[Point],
    mk: Optional[Point]
) -> List[Tuple[str, Point, Optional[Point], Optional[Point]]]:
    """Generates all legal single actions for Cyan and the resulting piece positions."""
    actions = []
    occupied = {pos for pos in (ck, cp, mk) if pos}

    if cp and mk:
        if ping_can_capture_target(cp, mk, {ck}):
            actions.append((f"Ping captures Kernel at {fmt(mk)}", ck, mk, None))
        for np in get_ping_moves(cp, occupied):
            actions.append((f"Ping moves to {fmt(np)}", ck, np, mk))

    for nk in get_kernel_steps(ck):
        if nk == cp:
            continue
        if nk == mk:
            actions.append((f"Kernel captures Kernel at {fmt(mk)}", nk, cp, None))
        else:
            actions.append((f"Kernel moves to {fmt(nk)}", nk, cp, mk))

    return actions


def cyan_turn_actions(
    ck: Point,
    cp: Optional[Point],
    mk: Point
) -> List[Tuple[List[str], Point, Optional[Point], Optional[Point]]]:
    """Generates all combinations of Cyan's 2 actions per turn."""
    turns = []
    for act1, k1, p1, m1 in cyan_single_actions(ck, cp, mk):
        if m1 is None:
            turns.append(([act1], k1, p1, None))
            continue
        for act2, k2, p2, m2 in cyan_single_actions(k1, p1, m1):
            turns.append(([act1, act2], k2, p2, m2))
    return turns


def magenta_turn_actions(
    mk: Point,
    ck: Point,
    cp: Optional[Point]
) -> List[Tuple[str, Optional[Point], Optional[Point]]]:
    """
    Generates all distinct 2-step endings for Magenta Kernel.
    Resolves the Bodyguard rule if Magenta attacks a guarded Kernel.
    """
    a1_states = []
    for m1 in get_kernel_steps(mk):
        if m1 == ck:
            if has_bodyguard(ck, cp):
                a1_states.append((m1, ck, None))
            else:
                return [("MAGENTA_WINS", ck, cp)]
        elif m1 == cp:
            a1_states.append((m1, ck, None))
        else:
            a1_states.append((m1, ck, cp))

    a2_states = set()
    for m1, cur_ck, cur_cp in a1_states:
        for m2 in get_kernel_steps(m1):
            if m2 == cur_ck:
                if has_bodyguard(cur_ck, cur_cp):
                    a2_states.add((m2, cur_ck, None))
                else:
                    return [("MAGENTA_WINS", cur_ck, cur_cp)]
            elif cur_cp and m2 == cur_cp:
                a2_states.add((m2, cur_ck, None))
            else:
                a2_states.add((m2, cur_ck, cur_cp))

    return list(a2_states)


# Transposition table for minimax search
memo: Dict[Tuple[Point, Optional[Point], Point, int], Tuple[int, Optional[List[str]]]] = {}


def get_min_turns_to_win(
    ck: Point,
    cp: Optional[Point],
    mk: Point,
    max_depth: int = 6
) -> Tuple[int, Optional[List[str]]]:
    """
    Minimax search for shortest forced win against optimal Magenta defense.
    Returns (turns_to_win, best_turn_actions).
    """
    state_key = (ck, cp, mk, max_depth)
    if state_key in memo:
        return memo[state_key]

    cyan_turns = cyan_turn_actions(ck, cp, mk)

    for acts, _, _, n_mk in cyan_turns:
        if n_mk is None:
            memo[state_key] = (0, acts)
            return (0, acts)

    if max_depth == 0:
        return (999, None)

    best_actions = None
    min_turns = 999

    for acts, n_ck, n_cp, n_mk in cyan_turns:
        if n_mk is None:
            continue

        m_responses = magenta_turn_actions(n_mk, n_ck, n_cp)
        if any(r[0] == "MAGENTA_WINS" for r in m_responses):
            continue

        worst_case_turns = 0
        all_branches_win = True

        for next_mk, next_ck, next_cp in m_responses:
            if next_mk is None or next_ck is None:
                continue
            turns_needed, _ = get_min_turns_to_win(next_ck, next_cp, next_mk, max_depth - 1)
            if turns_needed >= 999:
                all_branches_win = False
                break
            worst_case_turns = max(worst_case_turns, 1 + turns_needed)

        if all_branches_win and worst_case_turns < min_turns:
            min_turns = worst_case_turns
            best_actions = acts
            if min_turns == 1:
                break

    memo[state_key] = (min_turns, best_actions)
    return (min_turns, best_actions)


def solve_game(ck_str: str, cp_str: Optional[str], mk_str: str, max_search_depth: int = 7) -> None:
    memo.clear()
    ck = parse_coord(ck_str)
    cp = parse_coord(cp_str) if cp_str and cp_str.upper() != "NONE" else None
    mk = parse_coord(mk_str)

    print(f"State: Cyan Kernel: {fmt(ck)} | Cyan Ping: {fmt(cp)} | Magenta Kernel: {fmt(mk)}")

    for acts, _, _, n_mk in cyan_turn_actions(ck, cp, mk):
        if n_mk is None:
            print("Direct win in current turn:")
            for idx, a in enumerate(acts, 1):
                print(f"  {idx}. {a}")
            return

    for depth in range(1, max_search_depth + 1):
        turns, best_acts = get_min_turns_to_win(ck, cp, mk, max_depth=depth)
        if turns < 999 and best_acts:
            print(f"Forced win in {turns} opponent turn(s):")
            for idx, a in enumerate(best_acts, 1):
                print(f"  {idx}. {a}")
            return

    safe_moves = []
    for acts, n_ck, n_cp, n_mk in cyan_turn_actions(ck, cp, mk):
        if n_mk is None:
            continue
        m_responses = magenta_turn_actions(n_mk, n_ck, n_cp)
        if not any(r[0] == "MAGENTA_WINS" for r in m_responses):
            p_dist = chebyshev(n_cp, n_mk) if n_cp else 99
            safe_moves.append((acts, n_ck, n_cp, p_dist))

    safe_moves.sort(key=lambda x: x[3])
    if safe_moves:
        best = safe_moves[0]
        print("Best strategic move (no forced win within horizon):")
        for idx, a in enumerate(best[0], 1):
            print(f"  {idx}. {a}")
    else:
        print("No safe move sequence found.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Solve Colony: One Level 2 adversarial game states using minimax search."
    )
    parser.add_argument("cyan_kernel", help="Cyan Kernel square (e.g. C3)")
    parser.add_argument("cyan_ping", help="Cyan Ping square or 'None' (e.g. F5)")
    parser.add_argument("magenta_kernel", help="Magenta Kernel square (e.g. F8)")
    parser.add_argument(
        "--depth",
        type=int,
        default=7,
        help="Maximum search depth in turns (default: 7)"
    )
    args = parser.parse_args()

    solve_game(args.cyan_kernel, args.cyan_ping, args.magenta_kernel, max_search_depth=args.depth)


if __name__ == '__main__':
    main()
