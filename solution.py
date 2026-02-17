from collections import deque
from wumpus import Orientation, Actions, Percepts

class Agent:
    def __init__(self, size=(4, 4)):
        # World size (width x height)
        self.size = size
        self.width, self.height = size
        self.new_episode()

    def new_episode(self):
        # Agent position and orientation
        self.pos = (0, 0)
        self.orientation = Orientation.NORTH

        # World knowledge
        self.visited = set()        # Cells already visited
        self.safe = set()           # Cells known to be safe
        self.frontier = set()       # Safe but not yet visited cells
        self.possible_pits = {}     # pit suspicion count
        self.possible_wumpus = {}   # wumpus suspicion count

        # Internal state
        self.has_gold = False
        self.run_home = False
        self.plan = deque()         # Planned sequence of actions
        self.has_arrow = True
        self.wumpus_alive = True

        # The starting cell is always safe
        self.visited.add(self.pos)
        self.safe.add(self.pos)

    # Utility functions

    def _in_bounds(self, pos):
        # Check whether a position is inside the world.
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def _neighbors(self, pos):
        # all valid neighboring cells (4-connected grid).
        x, y = pos
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            new_pos = (x+dx, y+dy)
            if self._in_bounds(new_pos):
                yield new_pos

    def _forward_pos(self, pos=None, orientation=None):
        # Return the cell directly in front of the agent.
        if pos is None:
            pos = self.pos
        if orientation is None:
            orientation = self.orientation
        x, y = pos
        if orientation == Orientation.NORTH:
            return (x, y+1)
        if orientation == Orientation.SOUTH:
            return (x, y-1)
        if orientation == Orientation.EAST:
            return (x+1, y)
        if orientation == Orientation.WEST:
            return (x-1, y)
        return pos

    def _orient_left(self, o):
        return (o - 1) % 4

    def _orient_right(self, o):
        return (o + 1) % 4

    # Path planning
    def _shortest_path(self, start, goal):
        # Breadth-first search for the shortest path using only safe cells.
        if start == goal:
            return [start]
        queue = deque([start])
        parent = {start: None}
        while queue:
            cur = queue.popleft()
            if cur == goal:
                break
            for nb in self._neighbors(cur):
                if nb in self.safe and nb not in parent:
                    parent[nb] = cur
                    queue.append(nb)
        if goal not in parent:
            return None
        # Reconstruct path from goal to start
        path = []
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        return path

    # Convert a path (list of cells) into a sequence of LEFT and FORWARD actions.
    def _plan_move_along_path(self, path):
        actions = []
        orientation = self.orientation
        pos = self.pos
        for next_cell in path[1:]:
            cx, cy = pos
            nx, ny = next_cell
            # Determine desired orientation
            if nx > cx:
                target_o = Orientation.EAST
            elif nx < cx:
                target_o = Orientation.WEST
            elif ny > cy:
                target_o = Orientation.NORTH
            else:
                target_o = Orientation.SOUTH

            # Rotate until facing the correct direction
            while orientation != target_o:
                orientation = self._orient_left(orientation)
                actions.append(Actions.LEFT)

            # Move forward into the next cell
            actions.append(Actions.FORWARD)
            pos = next_cell
        return actions

    # Update internal knowledge based on the current percept.
    def _update_knowledge(self, percept):
        stench = bool(percept[Percepts.STENCH])
        breeze = bool(percept[Percepts.BREEZE])

        # Current cell is always safe
        self.safe.add(self.pos)
        self.visited.add(self.pos)

        # If no danger is perceived, all neighbors are safe
        if not breeze and not stench:
            for nb in self._neighbors(self.pos):
                self.safe.add(nb)
                if nb not in self.visited:
                    self.frontier.add(nb)
            return

        # If Breeze -> at least one neighboring pit
        if breeze:
            for nb in self._neighbors(self.pos):
                if nb not in self.safe:
                    self.possible_pits[nb] = self.possible_pits.get(nb, 0) + 1

        # If Stench -> nearby Wumpus
        if stench and self.wumpus_alive:
            for nb in self._neighbors(self.pos):
                if nb not in self.safe:
                    self.possible_wumpus[nb] = self.possible_wumpus.get(nb, 0) + 1

        # Update frontier with known safe but unvisited neighbors
        for nb in self._neighbors(self.pos):
            if nb in self.safe and nb not in self.visited:
                self.frontier.add(nb)

    # Target selection
    def _select_exploration_target(self):
        # Select the safest frontier cell based on minimal risk score.
        best_cell = None
        best_score = float("inf")
        for cell in list(self.frontier):
            if cell in self.visited:
                self.frontier.discard(cell)
                continue
            if cell not in self.safe:
                continue
            pit_risk = self.possible_pits.get(cell, 0)
            wump_risk = self.possible_wumpus.get(cell, 0)
            score = pit_risk * 10 + wump_risk * 5
            if score < best_score:
                best_score = score
                best_cell = cell
        return best_cell

    # Main agent logic

    def get_action(self, percept, reward):
        # Update knowledge based on current percept
        self._update_knowledge(percept)

        # Grab gold immediately if perceived
        if percept[Percepts.GLITTER] and not self.has_gold:
            self.has_gold = True
            self.plan.clear()
            return Actions.GRAB

        # After grabbing gold, plan a return path home
        if self.has_gold and not self.run_home:
            path_home = self._shortest_path(self.pos, (0, 0))
            if path_home is None:
                self.run_home = True
            else:
                self.plan = deque(self._plan_move_along_path(path_home))
                self.run_home = True

        # Exit the cave when back at the start with gold
        if self.has_gold and self.pos == (0, 0):
            return Actions.CLIMB

        # Execute planned actions if available
        if self.plan:
            action = self.plan.popleft()
        else:
            # Choose a new exploration target
            target = self._select_exploration_target()
            if target is not None and target != self.pos:
                path = self._shortest_path(self.pos, target)
                if path is not None:
                    self.plan = deque(self._plan_move_along_path(path))
                    action = self.plan.popleft()
                else:
                    self.frontier.discard(target)
                    action = Actions.LEFT
            else:
                # No safe exploration possible -> return home or exit
                if self.pos != (0, 0):
                    path_home = self._shortest_path(self.pos, (0, 0))
                    if path_home is not None:
                        self.plan = deque(self._plan_move_along_path(path_home))
                        action = self.plan.popleft()
                    else:
                        action = Actions.LEFT
                else:
                    return Actions.CLIMB

        # Update internal position and orientation
        if action == Actions.LEFT:
            self.orientation = self._orient_left(self.orientation)
        elif action == Actions.RIGHT:
            self.orientation = self._orient_right(self.orientation)
        elif action == Actions.FORWARD:
            new_pos = self._forward_pos()
            if self._in_bounds(new_pos):
                self.pos = new_pos

        return action
