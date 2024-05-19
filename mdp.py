from attrs import field, frozen
from math import isclose
from random import choice, choices


@frozen
class State:
    level: int | None = field(default=None)
    tired: bool | None = field(default=None)

    terminal: bool = field(default=False)
    title: str | None = field(default=None)

    def __str__(self):
        if self.terminal:
            return self.title

        return f"L{self.level} ({'T' if self.tired else 'R'})"


class MDP:
    def __init__(self, states, actions, transitions=None, rewards=None, gamma=0.9, eps=1e-6, T=100, costOfLiving=0, startingState = None):
        self.states = states
        self.actions = actions
        self.transitions = transitions
        self.rewards = rewards
        self.gamma = gamma
        self.eps = eps
        self.T = T
        self.costOfLiving = costOfLiving

        self.startingState = startingState if startingState else states[0]
        self.currentState = self.startingState
        self.stepsTaken = 0

        self.checkProbabilities()

    def checkProbabilities(self):
        for s, a in self.transitions.items():
            for action, outcomes in a.items():
                try:
                    assert isclose(sum(outcomes.values()), 1, abs_tol=1e-4)  # Making sure the sum of outcomes is effectively 1
                except AssertionError:
                    raise AssertionError(f"Probability not 1: {s} + {action} -> {outcomes.values()}")

                for o, chance in outcomes.items():
                    try:
                        assert 0 <= chance <= 1
                    except AssertionError:
                        raise AssertionError(f"Invalid probability 1: {s} + {action} -> {o} ({chance})")


    def reset(self):
        self.currentState = self.startingState
        self.stepsTaken = 0

        return self.currentState

    @property
    def stateSpace(self) -> int:
        return len(self.states)

    @property
    def allActions(self) -> set:
        return set([a for A in self.actions.values() for a in A])

    @property
    def actionSpace(self) -> int:
        return len(self.allActions)  # To set to remove duplicate actions

    def getActions(self) -> list:
        return self.actions[self.currentState]

    def step(self, action):
        if action not in self.actions[self.currentState]:
            print('\tInvalid action:', action)
            return (self.currentState,
                    0,
                    False,
                    False,
                    "Invalid")

        possibleOutcomes = self.transitions[self.currentState][action]  # {s: chance, s: chance, etc.}
        newState = choices(list(possibleOutcomes.keys()), possibleOutcomes.values())[0]
        reward = self.rewards.get(self.currentState, {}).get(action, {}).get(newState, 0)

        self.currentState = newState
        self.stepsTaken += 1

        return (newState,
                reward + self.costOfLiving,
                newState not in self.transitions,  # In our case we treat 'sinks' (states with no further transitions) as terminal.
                self.stepsTaken >= self.T,
                None)  # What should info be?



def Test(mdp: MDP, how: str, maxLength: int) -> None:
    state = mdp.reset()  # reset/re-initialize
    totalRewards = 0

    print(f"Testing MDP with action '{how}' for {maxLength} turns.")
    for _ in range(maxLength):
        action = choice(mdp.getActions()) if how == 'random' else how

        newState, reward, terminated, truncated, info = mdp.step(
            action)  # execute an action in the current state of MDP

        if info == 'Invalid':
            return

        totalRewards += reward
        print(f"\t{state} -> {newState} via {action} (reward of {reward:.2f}, total of {totalRewards:.2f})")

        if terminated:
            print('\tDone!')
            break

        state = newState


def ActionsFromTransitions(t: dict) -> dict:  # state: {action1: outcomes, action2: outcomes} -> state: [actions]
    return {s: list(A) for s, A in t.items()}


def StatesFromTransitions(t: dict) -> list:
    return list(t)
