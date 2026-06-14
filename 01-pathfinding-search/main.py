import heapq
import math
import os
import time



#  HEURISTICS


def heuristic_manhattan(a, b):
    """H1 – Manhattan distance (L1 norm). Admissible on 4-directional grids."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def heuristic_euclidean(a, b):
    """H2 – Euclidean distance (L2 norm). Never overestimates, but less
    informed than Manhattan for grid movement."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def heuristic_chebyshev(a, b):
    """H3 – Chebyshev distance (L∞ norm). Suitable when 8-directional moves
    are allowed; included here as an extra option to compare behaviour."""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


HEURISTICS = {
    "manhattan":  heuristic_manhattan,
    "euclidean":  heuristic_euclidean,
    "chebyshev":  heuristic_chebyshev,
}


#  GREEDY BEST-FIRST SEARCH


def greedy(maze, start, finish, heuristic_name="manhattan"):
    """
    Greedy best-first search.

    Parameters:
    - maze           : 2-D list; 0 = open cell, 1 = wall.
    - start          : (row, col) tuple of the starting position.
    - finish         : (row, col) tuple of the finishing position.
    - heuristic_name : one of "manhattan" | "euclidean" | "chebyshev".

    Returns:
    - num_steps : length of the path found (number of moves), or -1 if none.
    - viz       : dict with everything needed for step-by-step visualisation.
    """
    rows = len(maze)
    cols = len(maze[0]) if rows else 0

    h = HEURISTICS.get(heuristic_name, heuristic_manhattan)

    # guard: start or finish on a wall / out of bounds 
    def in_bounds(r, c):
        return 0 <= r < rows and 0 <= c < cols

    if not in_bounds(*start) or not in_bounds(*finish):
        return -1, _empty_viz(maze, start, finish, heuristic_name)
    if maze[start[0]][start[1]] == 1 or maze[finish[0]][finish[1]] == 1:
        return -1, _empty_viz(maze, start, finish, heuristic_name)

    # trivial case 
    if start == finish:
        viz = _empty_viz(maze, start, finish, heuristic_name)
        viz["visited_order"] = [start]
        viz["path"] = [start]
        return 0, viz

    # open list: (h_value, node)
    open_heap = []
    heapq.heappush(open_heap, (h(start, finish), start))

    came_from   = {start: None}   # node -> parent
    visited_set = set()           # nodes popped from open list
    visited_order = []            # order in which nodes were *expanded*
    open_set    = {start}         # nodes currently in the heap

    DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]   # 4-directional

    found = False
    while open_heap:
        _, current = heapq.heappop(open_heap)
        open_set.discard(current)

        if current in visited_set:
            continue                    # stale heap entry

        visited_set.add(current)
        visited_order.append(current)

        if current == finish:
            found = True
            break

        r, c = current                  #r,c = row, column 
        for dr, dc in DIRECTIONS:       #dr, dc = delta_row, delta_column
            nr, nc = r + dr, c + dc     #nr, nc = new_row, new_column 
            neighbour = (nr, nc)
            if (in_bounds(nr, nc)
                    and maze[nr][nc] == 0
                    and neighbour not in visited_set
                    and neighbour not in open_set):
                came_from[neighbour] = current
                open_set.add(neighbour)
                heapq.heappush(open_heap, (h(neighbour, finish), neighbour))

    # ── reconstruct path ──────────────────────────────────────────────────
    if not found:
        path = []
        num_steps = -1
    else:
        path = []
        node = finish
        while node is not None:
            path.append(node)
            node = came_from[node]
        path.reverse()
        num_steps = len(path) - 1   # moves = nodes - 1

    viz = {
        "maze":           maze,
        "start":          start,
        "finish":         finish,
        "visited_order":  visited_order,
        "path":           path,
        "heuristic_name": heuristic_name,
    }
    return num_steps, viz


def _empty_viz(maze, start, finish, heuristic_name):
    return {
        "maze":           maze,
        "start":          start,
        "finish":         finish,
        "visited_order":  [],
        "path":           [],
        "heuristic_name": heuristic_name,
    }



#  VISUALISATION


# Symbols

SYM_WALL    = "██"
SYM_OPEN    = "  "
SYM_START   = "S "
SYM_FINISH  = "F "
SYM_VISITED = "· "
SYM_PATH    = "▪ "
SYM_CURRENT = "@ "


def _render(maze, start, finish, visited_so_far, current, path_so_far):
    """Return a list of strings (one per row) for the current frame."""
    rows = len(maze)
    cols = len(maze[0])
    lines = []
    for r in range(rows):
        row_str = ""
        for c in range(cols):
            cell = (r, c)
            if maze[r][c] == 1:
                row_str += SYM_WALL
            elif cell == current:
                row_str += SYM_CURRENT
            elif cell in path_so_far:
                row_str += SYM_PATH
            elif cell == start:
                row_str += SYM_START
            elif cell == finish:
                row_str += SYM_FINISH
            elif cell in visited_so_far:
                row_str += SYM_VISITED
            else:
                row_str += SYM_OPEN
        lines.append(row_str)
    return lines


def vizualize(viz, delay=0.5, show_final_summary=True):
    """
    Visualisation function. Shows step-by-step the work of the search
    algorithm in the console.

    Parameters:
    - viz               : dict returned by greedy().
    - delay             : seconds between frames (set 0 for instant).
    - show_final_summary: print legend + result after animation.
    """
    maze          = viz["maze"]
    start         = viz["start"]
    finish        = viz["finish"]
    visited_order = viz["visited_order"]
    path          = viz["path"]
    hname         = viz["heuristic_name"]

    path_set     = set(path)
    visited_so_far = set()

    def _clear():
        os.system("cls" if os.name == "nt" else "clear")

    print(f"\n{'═'*40}")
    print(f"  Greedy Best-First Search  |  heuristic: {hname}")
    print(f"  Start: {start}   Finish: {finish}")
    print(f"{'═'*40}\n")
    time.sleep(1)

    # ── step through expansions ────────────────────────────────────────────
    for step_idx, current in enumerate(visited_order):
        visited_so_far.add(current)
        lines = _render(maze, start, finish, visited_so_far, current, set())
        _clear()
        print(f"  Heuristic: {hname}   Step: {step_idx + 1}/{len(visited_order)}")
        print(f"  Expanding: {current}")
        print()
        for ln in lines:
            print(" " + ln)
        print()
        print(f"  {SYM_CURRENT}= current   {SYM_VISITED}= visited   {SYM_WALL}= wall")
        time.sleep(delay)

    # ── show final path
    if path:
        lines = _render(maze, start, finish, visited_so_far, None, path_set)
        _clear()
        print(f"  Heuristic: {hname}   ✔ Path found!")
        print()
        for ln in lines:
            print(" " + ln)
        print()
        if show_final_summary:
            print(f"  {SYM_PATH}= path   {SYM_VISITED}= visited   {SYM_WALL}= wall")
            print(f"\n  Cells visited : {len(visited_order)}")
            print(f"  Path length   : {len(path) - 1} steps")
            print(f"  Path          : {' → '.join(str(p) for p in path)}")
    else:
        _clear()
        print(f"  Heuristic: {hname}   ✘ No path found.")
        print()
        lines = _render(maze, start, finish, visited_so_far, None, set())
        for ln in lines:
            print(" " + ln)
        print()
        print(f"  {SYM_VISITED}= visited   {SYM_WALL}= wall")
        print(f"\n  Cells visited : {len(visited_order)}")

    print(f"\n{'═'*40}\n")



#  STATIC SNAPSHOT PRINT 
# ─────────────────────────────────────────────

def print_result(label, maze, start, finish, heuristic_name):
    """Run greedy, print a static snapshot and return (num_steps, viz)."""
    num_steps, viz = greedy(maze, start, finish, heuristic_name)
    path     = viz["path"]
    visited  = viz["visited_order"]
    path_set = set(path)

    print(f"\n{'─'*50}")
    print(f"  {label}")
    print(f"  Heuristic : {heuristic_name}")
    print(f"  Start={start}  Finish={finish}")
    print(f"{'─'*50}")

    rows = len(maze)
    for r in range(rows):
        row_str = ""
        for c in range(maze[r].__len__()):
            cell = (r, c)
            if maze[r][c] == 1:
                row_str += SYM_WALL
            elif cell in path_set and cell not in (start, finish):
                row_str += SYM_PATH
            elif cell == start:
                row_str += SYM_START
            elif cell == finish:
                row_str += SYM_FINISH
            elif cell in set(visited):
                row_str += SYM_VISITED
            else:
                row_str += SYM_OPEN
        print(" " + row_str)

    if num_steps != -1:
        print(f"\n  ✔  Path length   : {num_steps} steps")
        print(f"     Cells visited : {len(visited)}")
    else:
        print(f"\n  ✘  No path found.")
        print(f"     Cells visited : {len(visited)}")
    return num_steps, viz



#  TEST CASES


def run_all_tests():
    results = []   # (label, heuristic, steps, cells_visited)

    # ── TC1: Provided sample maze ─
    maze1 = [
        [0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [1, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
    ]
    for h in HEURISTICS:
        n, viz = print_result("TC1 – Sample maze", maze1, (0,0), (4,4), h)
        results.append(("TC1", h, n, len(viz["visited_order"])))

    # ── TC2: Straight corridor
    maze2 = [
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0],
    ]
    for h in HEURISTICS:
        n, viz = print_result("TC2 – Straight corridor", maze2, (0,0), (4,4), h)
        results.append(("TC2", h, n, len(viz["visited_order"])))

    # ── TC3: Start == Finish
    maze3 = [[0, 0], [0, 0]]
    n, viz = print_result("TC3 – Start == Finish", maze3, (0,0), (0,0), "manhattan")
    results.append(("TC3", "manhattan", n, len(viz["visited_order"])))
    assert n == 0, f"TC3 FAILED: expected 0, got {n}"

    # ── TC4: Completely blocked (no path)
    maze4 = [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ]
    n, viz = print_result("TC4 – No path (fully blocked)", maze4, (0,0), (2,2), "manhattan")
    results.append(("TC4", "manhattan", n, len(viz["visited_order"])))
    assert n == -1, f"TC4 FAILED: expected -1, got {n}"

    # ── TC5: 1×1 maze
    maze5 = [[0]]
    n, viz = print_result("TC5 – 1×1 maze (trivial)", maze5, (0,0), (0,0), "manhattan")
    results.append(("TC5", "manhattan", n, len(viz["visited_order"])))
    assert n == 0, f"TC5 FAILED: expected 0, got {n}"

    # ── TC6: Single row ───────────────────────────────────────────────────
    maze6 = [[0, 0, 0, 0, 0]]
    n, viz = print_result("TC6 – Single row", maze6, (0,0), (0,4), "manhattan")
    results.append(("TC6", "manhattan", n, len(viz["visited_order"])))
    assert n == 4, f"TC6 FAILED: expected 4, got {n}"

    # ── TC7: Single column ─
    maze7 = [[0],[0],[0],[0],[0]]
    n, viz = print_result("TC7 – Single column", maze7, (0,0), (4,0), "manhattan")
    results.append(("TC7", "manhattan", n, len(viz["visited_order"])))
    assert n == 4, f"TC7 FAILED: expected 4, got {n}"

    # ── TC8: Start or finish on a wall
    maze8 = [[0, 1], [0, 0]]
    n8a, _ = greedy(maze8, (0,1), (1,1), "manhattan")   # start on wall
    n8b, _ = greedy(maze8, (0,0), (0,1), "manhattan")   # finish on wall
    print(f"\n{'─'*50}")
    print("  TC8 – Start/Finish on wall")
    print(f"  start-on-wall → {n8a}   finish-on-wall → {n8b}")
    assert n8a == -1, f"TC8a FAILED: expected -1, got {n8a}"
    assert n8b == -1, f"TC8b FAILED: expected -1, got {n8b}"
    results.append(("TC8a", "manhattan", n8a, 0))
    results.append(("TC8b", "manhattan", n8b, 0))

    # ── TC9: Large open maze (heuristic comparison focus) ───────
    maze9 = [[0]*10 for _ in range(10)]
    for h in HEURISTICS:
        n, viz = print_result("TC9 – 10×10 open grid", maze9, (0,0), (9,9), h)
        results.append(("TC9", h, n, len(viz["visited_order"])))

    # ── TC10: Maze where greedy takes a dead-end detour ───
    # Greedy is drawn toward the finish but hits a dead-end corridor.
    maze10 = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 1, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 0],
    ]
    for h in HEURISTICS:
        n, viz = print_result("TC10 – Dead-end detour", maze10, (0,0), (6,6), h)
        results.append(("TC10", h, n, len(viz["visited_order"])))

    # ── TC11: Narrow winding path ────────────────
    maze11 = [
        [0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ]
    for h in HEURISTICS:
        n, viz = print_result("TC11 – Narrow winding path", maze11, (0,0), (4,0), h)
        results.append(("TC11", h, n, len(viz["visited_order"])))

    # ── TC12: Out-of-bounds start ───────────────────
    maze12 = [[0, 0], [0, 0]]
    n12, _ = greedy(maze12, (-1, 0), (1, 1), "manhattan")
    print(f"\n{'─'*50}")
    print("  TC12 – Out-of-bounds start")
    print(f"  result → {n12}")
    assert n12 == -1, f"TC12 FAILED: expected -1, got {n12}"
    results.append(("TC12", "manhattan", n12, 0))

    # ── Summary table ─────
    print(f"\n\n{'═'*65}")
    print(f"  {'TEST':<8} {'HEURISTIC':<12} {'STEPS':>7} {'CELLS VISITED':>14}")
    print(f"{'═'*65}")
    for label, h, steps, cells in results:
        step_str = str(steps) if steps != -1 else "NO PATH"
        print(f"  {label:<8} {h:<12} {step_str:>7} {cells:>14}")
    print(f"{'═'*65}\n")
    print("  All assertions passed ✔\n")


#  MAIN  animation demo
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # ── 1. Run and print all test cases ────────────
    run_all_tests()
    time.sleep(5)

    # ── 2. Animated visualisation on the sample maze with each heuristic ──
    maze = [
        [0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [1, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
    ]
    start_position  = (0, 0)
    finish_position = (4, 4)

    for hname in HEURISTICS:
        num_steps, viz = greedy(maze, start_position, finish_position, hname)

        if num_steps != -1:
            print(f"Path from {start_position} to {finish_position} "
                  f"using greedy best-first search [{hname}] is {num_steps} steps.")
        else:
            print(f"No path from {start_position} to {finish_position} exists.")
        
        time.sleep(1)
        vizualize(viz, delay=0.5)
        time.sleep(1)
