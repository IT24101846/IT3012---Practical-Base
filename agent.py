# agent.py

import random


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


    def bfs_search(self, start_pos, goal_pos, walls, grid_size):
        raise NotImplementedError("Practical 3: implement BFS pathfinding here.")
