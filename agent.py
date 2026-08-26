# agent.py

import random
import math
from collections import deque
import heapq


class GreedyGridAgent:
    """A simple agent that wanders randomly to clear the grid."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        return random.choice(self.actions_pool)


class SimpleReflexAgent:

    def sense_and_act(self, percept: dict) -> str:
        if percept['food_here']:
            return 'Up'
        if percept['wall_ahead']:
            return 'Left'
        return 'Up'


class ModelBasedAgent:
  
    DIRECTIONS = ['Up', 'Right', 'Down', 'Left']
    DIR_VECTORS = {'Up': (0, 1), 'Right': (1, 0), 'Down': (0, -1), 'Left': (-1, 0)}

    def __init__(self):
        self.est_pos = (0, 0)           # believed position, dead-reckoned
        self.visited_cells = {(0, 0)}   # cells the agent believes it has occupied
        self.last_action = None         # action chosen on the previous turn
        self.tried_while_blocked = []   # directions already tried since getting stuck here

    def _update_state(self, percept: dict):
    
        if self.last_action in self.DIR_VECTORS and not percept.get('bumped', False):
            dx, dy = self.DIR_VECTORS[self.last_action]
            self.est_pos = (self.est_pos[0] + dx, self.est_pos[1] + dy)
            self.visited_cells.add(self.est_pos)

    def _decide(self, percept: dict) -> str:
        if percept['wall_ahead']:
            # We know at least one direction is blocked: whichever way we
            # were last facing -- that's *why* wall_ahead is True now.
            if self.last_action in self.DIRECTIONS and self.last_action not in self.tried_while_blocked:
                self.tried_while_blocked.append(self.last_action)

            candidates = [d for d in self.DIRECTIONS if d not in self.tried_while_blocked]
            if not candidates:
                # Tried every direction from this spot -- boxed in, start over.
                self.tried_while_blocked = []
                candidates = list(self.DIRECTIONS)

            def leads_to_new_cell(d):
                dx, dy = self.DIR_VECTORS[d]
                return (self.est_pos[0] + dx, self.est_pos[1] + dy) not in self.visited_cells

            unexplored = [d for d in candidates if leads_to_new_cell(d)]
            action = (unexplored or candidates)[0]

            self.tried_while_blocked.append(action)
            return action

        # Either the way ahead is clear, or we're standing on food that's
        # already been auto-collected -- either way, keep heading the way
        # we were already going instead of resetting to some fixed
        # direction every turn (that fixed-default version is what got
        # ModelBasedAgent stuck oscillating in corners during testing --
        # it kept walking back the way it came instead of continuing on).
        self.tried_while_blocked = []
        return self.last_action if self.last_action in self.DIRECTIONS else 'Up'

    def sense_and_act(self, percept: dict) -> str:
        self._update_state(percept)
        action = self._decide(percept)
        self.last_action = action
        return action


class SearchAgent:
    """Goal-based agent: builds a full offline plan using BFS/DFS/UCS/A*
    over the exposed world model, then executes it action-by-action."""

    DIRECTIONS = ['Up', 'Down', 'Left', 'Right']
    DIR_VECTORS = {'Up': (0, 1), 'Down': (0, -1), 'Left': (-1, 0), 'Right': (1, 0)}

    def __init__(self):
        self.plan = []                 # Step 1.3: queued sequence of actions
        self.active_algo = 'BFS'       # 'BFS' | 'DFS' | 'UCS' | 'AStar' -- swap to compare

    # ---------- shared helpers ----------

    def _neighbors(self, pos, walls, grid_size):
        """Yield (action, next_pos) for every in-bounds, non-wall neighbour."""
        width, height = grid_size
        x, y = pos
        for action, (dx, dy) in self.DIR_VECTORS.items():
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in walls:
                yield action, (nx, ny)

    def _reconstruct_path(self, came_from, start_pos, goal_pos):
        """Walk the came_from chain backwards from goal to start,
        turning it into a forward list of actions."""
        if goal_pos != start_pos and goal_pos not in came_from:
            return None

        actions = []
        current = goal_pos
        while current != start_pos:
            prev_pos, action = came_from[current]
            actions.append(action)
            current = prev_pos
        actions.reverse()
        return actions

    # ---------- Step 1.1: heuristic functions ----------

    def manhattan_distance(self, pos, goal):
        """h(n) = |x1 - x2| + |y1 - y2| -- the cost of moving along grid
        axes only (no diagonals), which is exactly how this agent moves.
        Admissible and consistent on this 4-directional grid."""
        x1, y1 = pos
        x2, y2 = goal
        return abs(x1 - x2) + abs(y1 - y2)

    def euclidean_distance(self, pos, goal):
        """h(n) = sqrt((x1 - x2)^2 + (y1 - y2)^2) -- straight-line
        distance. Still admissible here (never overestimates), but looser
        than Manhattan since the agent can't actually move diagonally."""
        x1, y1 = pos
        x2, y2 = goal
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    # ---------- Step 1.2: the three uninformed search strategies ----------

    def bfs_search(self, start_pos, goal_pos, walls, grid_size):
        """Breadth-First Search -- FIFO frontier, explores shallowest nodes
        first. Optimal for uniform step-cost grids like this one."""
        start_pos, goal_pos, walls = tuple(start_pos), tuple(goal_pos), set(map(tuple, walls))

        if start_pos == goal_pos:
            return []

        frontier = deque([start_pos])
        reached = {start_pos}          # Graph Search: never revisit a state
        came_from = {}

        while frontier:
            current = frontier.popleft()       # FIFO
            if current == goal_pos:
                return self._reconstruct_path(came_from, start_pos, goal_pos)

            for action, nxt in self._neighbors(current, walls, grid_size):
                if nxt not in reached:
                    reached.add(nxt)
                    came_from[nxt] = (current, action)
                    frontier.append(nxt)

        return None  # goal unreachable

    def dfs_search(self, start_pos, goal_pos, walls, grid_size):
        """Depth-First Search -- LIFO frontier, dives down one branch as
        deep as possible before backtracking. Not optimal: expect long,
        winding paths compared to BFS/UCS."""
        start_pos, goal_pos, walls = tuple(start_pos), tuple(goal_pos), set(map(tuple, walls))

        if start_pos == goal_pos:
            return []

        frontier = [start_pos]
        reached = {start_pos}
        came_from = {}

        while frontier:
            current = frontier.pop()           # LIFO
            if current == goal_pos:
                return self._reconstruct_path(came_from, start_pos, goal_pos)

            for action, nxt in self._neighbors(current, walls, grid_size):
                if nxt not in reached:
                    reached.add(nxt)
                    came_from[nxt] = (current, action)
                    frontier.append(nxt)

        return None

    def ucs_search(self, start_pos, goal_pos, walls, grid_size):
        """Uniform-Cost Search -- priority queue ordered by path cost g(n).
        Every move here costs 1, so UCS degenerates to BFS's path length,
        but the frontier discipline (always expand cheapest g(n) first)
        is what generalises to weighted grids."""
        start_pos, goal_pos, walls = tuple(start_pos), tuple(goal_pos), set(map(tuple, walls))

        if start_pos == goal_pos:
            return []

        counter = 0  # tie-breaker so heapq never tries to compare tuples of positions
        frontier = [(0, counter, start_pos)]
        best_cost = {start_pos: 0}
        came_from = {}

        while frontier:
            cost, _, current = heapq.heappop(frontier)

            if current == goal_pos:
                return self._reconstruct_path(came_from, start_pos, goal_pos)

            if cost > best_cost.get(current, float('inf')):
                continue  # stale entry, a cheaper route to `current` was already popped

            for action, nxt in self._neighbors(current, walls, grid_size):
                new_cost = cost + 1  # uniform step cost
                if nxt not in best_cost or new_cost < best_cost[nxt]:
                    best_cost[nxt] = new_cost
                    came_from[nxt] = (current, action)
                    counter += 1
                    heapq.heappush(frontier, (new_cost, counter, nxt))

        return None

    def astar_search(self, start_pos, goal_pos, walls, grid_size, heuristic_type='manhattan'):
        """A* Search -- priority queue ordered by f(n) = g(n) + h(n), where
        g(n) is the exact cost-so-far and h(n) is a heuristic estimate of
        the remaining cost. Combines UCS's cost-awareness with heuristic
        guidance towards the goal, so it typically expands far fewer nodes
        than BFS/UCS on open maps while still finding the optimal path
        (since both heuristics above are admissible)."""
        start_pos, goal_pos = tuple(start_pos), tuple(goal_pos)
        walls = set(map(tuple, walls))

        if start_pos == goal_pos:
            return []

        heuristic_fn = self.manhattan_distance if heuristic_type == 'manhattan' else self.euclidean_distance

        # Priority queue entries: (f_cost, g_cost, current_pos, path_taken)
        g_start = 0
        h_start = heuristic_fn(start_pos, goal_pos)
        frontier = [(g_start + h_start, g_start, start_pos, [])]
        reached_states = set()

        while frontier:
            f_cost, g_cost, current_pos, path_taken = heapq.heappop(frontier)

            if current_pos == goal_pos:
                return path_taken

            if current_pos in reached_states:
                continue  # stale entry -- a cheaper route here was already expanded

            reached_states.add(current_pos)

            for action, nxt in self._neighbors(current_pos, walls, grid_size):
                if nxt not in reached_states:
                    g_new = g_cost + 1               # uniform step cost
                    h_new = heuristic_fn(nxt, goal_pos)
                    f_new = g_new + h_new
                    heapq.heappush(frontier, (f_new, g_new, nxt, path_taken + [action]))

        return None  # goal unreachable

    # ---------- Step 1.3: offline planning + execution ----------

    def _closest_food(self, agent_pos, all_food):
        if not all_food:
            return None
        return min(
            all_food,
            key=lambda f: abs(f[0] - agent_pos[0]) + abs(f[1] - agent_pos[1])  # Manhattan distance
        )

    def sense_and_act(self, percept: dict) -> str:
        if not self.plan:
            # Step 1.3: pull the global state the planner needs straight out
            # of the (fully-observable) percept dict.
            agent_pos = tuple(percept['agent_pos'])
            remaining_food = percept['remaining_food']   # count, just for bookkeeping/logging
            goal_pos = self._closest_food(agent_pos, percept['all_food'])

            if goal_pos is None or remaining_food == 0:
                return random.choice(self.DIRECTIONS)  # nothing left to hunt

            walls = percept['walls']
            grid_size = percept['grid_size']

            if self.active_algo == 'BFS':
                new_plan = self.bfs_search(agent_pos, goal_pos, walls, grid_size)
            elif self.active_algo == 'DFS':
                new_plan = self.dfs_search(agent_pos, goal_pos, walls, grid_size)
            elif self.active_algo == 'UCS':
                new_plan = self.ucs_search(agent_pos, goal_pos, walls, grid_size)
            elif self.active_algo == 'AStar':
                new_plan = self.astar_search(agent_pos, goal_pos, walls, grid_size)
            else:
                raise ValueError(f"Unknown active_algo: {self.active_algo!r}")

            # If somehow unreachable, fall back to a random nudge rather than crashing
            self.plan = new_plan if new_plan else [random.choice(self.DIRECTIONS)]

        return self.plan.pop(0)


if __name__ == "__main__":
    # ---- Step 1.1 Testing Checkpoint ----
    # Mock start (0, 0) and goal (3, 4):
    #   Manhattan should print 7
    #   Euclidean should print 5.0
    test_agent = SearchAgent()
    start, goal = (0, 0), (3, 4)

    print("Manhattan distance:", test_agent.manhattan_distance(start, goal))
    print("Euclidean distance:", test_agent.euclidean_distance(start, goal))