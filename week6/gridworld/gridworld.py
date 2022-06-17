import numpy as np


ACTION_SPACE = ('U', 'D', 'L', 'R')


class GridBuilderHelper():

    action_mapping = {
        'U': (0, -1),
        'D': (0, 1),
        'L': (1, -1),
        'R': (1, 1)
    }

    def __init__(self, i, j):
        self.i = i
        self.j = j
        self.border = (i, j)

    def new_state_result(self, state, action):
        state = list(state)
        act_map = self.action_mapping[action]
        axis = act_map[0]
        move = act_map[1]
        state[axis] += move
        return tuple(state)

    def breaking_border(self, state, action):
        act_map = self.action_mapping[action]
        axis = act_map[0]
        move = act_map[1]
        new_state = state[axis] + move
        return (new_state < 0) or (new_state > self.border[axis])

    def build_state_actions(self, terminal):
        state_action = {}
        for x in range(self.i + 1):
            for y in range(self.j + 1):
                key = (x, y)
                if key in terminal:
                    continue
                state_action[key] = []
                for action in ACTION_SPACE:
                    if not self.breaking_border(key, action):
                        state_action[key].append(action)
        return state_action

    def build_action_probs(self, state_action):
        action_probs = {}
        for state, actions in state_action.items():
            for action in actions:
                key = (state, action)
                new_state = self.new_state_result(state, action)
                action_probs[key] = {new_state: 1.0}
        return action_probs


class WindyGrid:
    def __init__(self, rows, cols, start):
        self.rows = rows
        self.cols = cols
        self.start = start
        self.reset()

    def reset(self):
        self.i = self.start[0]
        self.j = self.start[1]

    def set(self, rewards, actions, probs):
        # rewards should be a dict of: (i, j): r (row, col): reward
        # actions should be a dict of: (i, j): A (row, col): list of possible actions
        self.rewards = rewards
        self.actions = actions
        self.probs = probs

    def set_state(self, s):
        self.i = s[0]
        self.j = s[1]

    def current_state(self):
        return (self.i, self.j)

    def move(self, action):
        s = (self.i, self.j)
        a = action

        next_state_probs = self.probs[(s, a)]
        next_states = list(next_state_probs.keys())
        next_probs = list(next_state_probs.values())
        next_state_idx = np.random.choice(len(next_states), p=next_probs)
        s2 = next_states[next_state_idx]

        # update the current state
        self.i, self.j = s2

        # return a reward (if any)
        return self.rewards.get(s2, 0)

    def game_over(self):
        # returns true if game is over, else false
        # true if we are in a state where no actions are possible
        return (self.i, self.j) not in self.actions

    def all_states(self):
        # possibly buggy but simple way to get all states
        # either a position that has possible next actions
        # or a position that yields a reward
        return set(self.actions.keys()) | set(self.rewards.keys())

    def get_state_actions(self):
        return self.actions

    def get_actions(self, state):
        return self.actions[state]


def windy_grid(x_max: int, y_max: int, start: tuple):
    helper = GridBuilderHelper(x_max, y_max)
    g = WindyGrid(x_max, y_max, start)
    rewards = {(2, 5): 1}
    terminal = rewards.keys()
    actions = helper.build_state_actions(terminal)
    probs = helper.build_action_probs(actions)

    ## here you can set a custom transition chance, e.g. move 2 fields up instead of one
    ## syntax: probs[(state, action)] = {s_prime: probability}
    # probs[((0, 0), 'R')] = {(2, 5): 1.0}
    g.set(rewards, actions, probs)
    return g
