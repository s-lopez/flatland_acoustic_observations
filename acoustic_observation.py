from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple, Union, Any
import networkx as nx
import numpy as np
import math
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.rail_env import RailEnv
from flatland.envs.agent_utils import EnvAgent
from copy import deepcopy
import random
import warnings

# Nomenclature and constants
NORTH, EAST, SOUTH, WEST = 0, 1, 2, 3
HEADINGS = (NORTH, EAST, SOUTH, WEST)
GRID_INCREMENTS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
LEFT, STRAIGHT, RIGHT, BACK = -1, 0, 1, 2
DIRECTIONS = (LEFT, STRAIGHT, RIGHT, BACK)
USABLE_SWITCH, UNUSABLE_SWITCH = 1, 2


def turn(direction: int, heading: int) -> int:
    """Return the new heading after turning from the original heading to direction"""
    return (direction + heading) % 4


def rel_direction(new_heading: int, old_heading: int) -> int:
    """Return the relative direction that an agents needs to turn to to face new_heading from old_heading"""
    direction = (new_heading - old_heading) % 4
    return direction if direction < 3 else -1


@dataclass
class TransitionMatrix:
    """transition_map is a uint16 coded as NN NE NS .... SE SS SW
    The matrix looks like:
               Exits (Axis=0)
     Headings   N E S W
     (Axis=1) N 0 1 0 0
              E 0 1 0 0
              S 0 0 0 0
              W 0 0 1 1
             """
    transition_uint16: int
    transitions: np.ndarray = field(init=False)
    exits: np.ndarray = field(init=False)
    exits_tuple: Tuple = field(init=False)

    def __post_init__(self):
        assert self.transition_uint16 > 0, "TransitionMatrix instantiated on an empty cell"
        self.transitions = np.array([int(x) for x in bin(self.transition_uint16)[2:].zfill(16)]).reshape(4, 4)
        self.exits = self.transitions.any(axis=0)
        self.exits_tuple = tuple(exit_ for exit_, exists in enumerate(self.exits) if exists)  # I love this

    def can_transition(self, old_heading: int, new_heading: int) -> int:
        return self.transitions[old_heading, new_heading]

    def exits_from(self, heading: int) -> np.ndarray:
        return self.transitions[heading, :]


@dataclass
class Sound:
    """An observable unit of information, emitted in a certain heading and able to propagate across the rail.

    Attributes:
        position (Tuple): The sound's position on the network.
        heading (int): The sound's heading.
        frequency (float): The sound's frequency.
        reach (int): The number of times this sound can turn before extinguishing.
        capacity (int): The size of the _turns and _distances deques.
        doppler (int): Indicates whether the sound is moving in the same or contrary direction as the emitter.
        malfunction (int): Information on the emitter's malfunction, if existing.
        origin (Tuple): The emitter's position.
        valid_journey (bool): True if the sound didn't turn illegally.
        total_distance_travelled (int): The number of nodes that this sound has visited.
    """
    position: Tuple
    heading: int
    frequency: float
    reach: int
    capacity: int
    doppler: int = 0
    malfunction: int = 0
    origin: Tuple[Tuple[int], int] = field(init=False, default=None)  # For debugging/traceability
    valid_journey: bool = field(init=False, default=True)
    total_distance_travelled: int = field(init=False, default=0)
    _visited_nodes: Set[Tuple[Tuple[int], int]] = field(init=False, default_factory=set)  # set of (position, heading)
    _turns: deque = field(init=False)
    _distances: deque = field(init=False)
    _distance_travelled: int = field(init=False, default=0)
    _n_turns: int = field(init=False, default=0)

    def __post_init__(self):
        self._visited_nodes.add((self.position, self.heading))
        self._turns = deque(maxlen=self.capacity)
        self._distances = deque(maxlen=self.capacity)
        self.origin = (self.position, self.heading) if self.origin is None else self.origin

    def split(self, turn_: int = STRAIGHT, is_legal=1) -> 'Sound':  # For compatibility with Python 3.6
        """Return a deep copy of the sound after turning to direction turn_ and saving the propagation history"""
        new_sound = deepcopy(self)
        new_sound.heading = turn(turn_, new_sound.heading)
        if is_legal:
            new_sound._turns.append(turn_)
        else:
            new_sound._turns.append(turn_ * UNUSABLE_SWITCH)
            new_sound.valid_journey = False
        new_sound._distances.append(new_sound._distance_travelled)
        new_sound._distance_travelled = 0
        new_sound._n_turns += 1
        return new_sound

    def copy(self):
        """Return a deep copy of the sound without further changes."""
        return deepcopy(self)

    def listen_from(self, heading: int, is_legal: bool):
        """Called when an ear listens to this sound. Appends the last value to _turns, which depends on the agent's
        and the sound's heading."""
        direction = rel_direction(new_heading=turn(BACK, self.heading), old_heading=heading)
        if is_legal:
            self._turns.append(direction)
        else:
            self._turns.append(direction * UNUSABLE_SWITCH)
            self.valid_journey = False
        self._distances.append(self._distance_travelled)
        return deepcopy(self)

    def step(self) -> bool:
        """Move one step along our heading. Return True if the node-heading pair has been visited already"""
        self.position = tuple(pos + inc for pos, inc in zip(self.position, GRID_INCREMENTS[self.heading]))
        # We don't want to walk a path again and we don't want to return where we came from, so:
        # Do not step if we have been in this position with same or opposite heading
        if ((self.position, self.heading) in self._visited_nodes) \
                or ((self.position, turn(BACK, self.heading)) in self._visited_nodes):
            return True
        else:
            self._visited_nodes.add((self.position, self.heading))
            self._distance_travelled += 1
            self.total_distance_travelled += 1
            return False

    def is_exhausted(self) -> bool:
        """True iff self._n_turns >= self.reach"""
        return self._n_turns >= self.reach

    def pop_all(self) -> Tuple:
        """Pop all elements from the sound's deques (Last In First Out!)"""
        assert len(self._turns) == len(self._distances), "Length mismatch while encoding a sound"
        L = len(self._turns)
        turns = [self._turns.pop() for _ in range(L)]
        distances = [self._distances.pop() for _ in range(L)]
        if L < self.capacity:
            turns.extend([0] * (self.capacity - L))
            distances.extend([0] * (self.capacity - L))
        return turns, distances


@dataclass
class Ear:
    """An interface for an agent's 'ear'."""
    agent: EnvAgent

    def listen(self, sounds: Union[Sound, List[Sound]]) -> None:
        """Called by AcousticObservation whenever one or more sounds reach this ear."""
        raise NotImplementedError

    def encode(self):
        """Returns an encoding of the sounds that this ear has been listening to."""
        raise NotImplementedError


@dataclass
class FixedLengthEar(Ear):
    """An Ear that has a fixed length/sound capacity and uses padding."""
    ear_capacity: int
    sound_capacity: int
    agent_frequency: float
    cell: TransitionMatrix = field(init=False, default=None)
    buffer: List[Sound] = field(init=False, default_factory=list)

    def listen(self, sounds: Union[Sound, List[Sound]]) -> None:
        """Add one or more sounds to the ear's buffer"""
        assert self.cell is not None, f"No cell has been defined for agent's {self.agent.handle} ear"
        if not isinstance(sounds, list):
            sounds = [sounds]
        for sound in sounds:
            if sound.frequency - self.agent_frequency < 1e-6:
                # Don't listen to yourself
                continue
            elif sound.heading == self.agent.direction:
                # Don't look back
                continue
            is_legal = self.cell.can_transition(new_heading=sound.heading, old_heading=self.agent.direction)
            sound.listen_from(self.agent.direction, is_legal)
            self.buffer.append(sound)

    def encode(self) -> np.ndarray:
        """Sample length elements at random from buffer."""
        # Sample the rest at random if needed
        if len(self.buffer) > self.ear_capacity:
            sounds = random.sample(self.buffer, self.ear_capacity)
            # Always keep the sound with the shortest valid distance to the target
            first = True
            for sound in self.buffer:
                if sound.valid_journey & (sound.frequency == -math.floor(self.agent_frequency)):
                    if first:
                        total_distance_travelled = sound.total_distance_travelled
                        shortest_valid_sound = sound
                        first = False
                        continue
                    if sound.total_distance_travelled < total_distance_travelled:
                        shortest_valid_sound = sound
            if first:
                warnings.warn(f"Agent {self.agent.handle} could not hear its target! Maybe the reach is too low?")
            elif shortest_valid_sound not in sounds:
                sounds[0] = shortest_valid_sound
        else:
            sounds = self.buffer

        # Extract deques (Last In First Out!!)
        encoding = np.zeros((self.ear_capacity, self.sound_capacity + 2))
        for sound, row in zip(sounds, encoding):
            turns, distances = sound.pop_all()
            malfunction = sound.malfunction
            doppler = sound.doppler
            row[:] = turns + distances + [malfunction, doppler]  # List concatenation!

        self.reset()
        return np.reshape(encoding, (1, self.ear_capacity * (self.sound_capacity + 2)))

    def reset(self) -> None:
        """Clear the ear's buffer"""
        self.buffer.clear()


class AcousticObservation(ObservationBuilder):
    def __init__(self,
                 ear_constructor: Any = FixedLengthEar,
                 ear_parameters: dict = None,
                 sound_parameters: dict = None,
                 max_propagation_steps: int = 500,
                 ):
        super().__init__()
        self.graph = nx.Graph()
        self.agent_freqs = {}
        self.agent_at_position = {}
        self.station_freqs = {}
        self.sounds = deque()
        self.sound_parameters = {"reach": 15, "capacity": 10} if sound_parameters is None else sound_parameters
        self.ear_parameters = {"ear_capacity": 20, "sound_capacity": 10} if ear_parameters is None else ear_parameters
        self.ear_constructor = ear_constructor
        self.ears = {}
        self.max_propagation_steps = max_propagation_steps

    def transitions_at(self, position: tuple) -> TransitionMatrix:
        """Syntactic sugar to return the transition matrix at position"""
        return self.graph.nodes[position]['transitions']

    def sounds_at(self, position: tuple) -> List[Sound]:
        """Syntactic sugar to return the pre-computed sounds at position"""
        return self.graph.nodes[position]['sounds']

    def append_sound(self, sound: Sound) -> None:
        """Syntactic sugar to append a sound to the associated node"""
        self.graph.nodes[sound.position]['sounds'].append(sound)

    def get(self, handle: int = 0):
        """For efficiency reasons, it's better to compute the observations for all agents at once."""
        raise NotImplementedError

    def get_many(self, handles: Optional[List[int]] = None):
        # Update the position of all agents present in the grid
        self.agent_at_position = {agent.position: agent.handle
                                  for agent in self.env.agents if agent.position is not None}

        # Update ear's transition matrix & listen to station sounds (pre-computed in self.reset())
        for position, handle in self.agent_at_position.items():
            self.ears[handle].cell = self.transitions_at(position)
            self.ears[handle].listen(self.sounds_at(position))

        # Generate sounds for all agents in the grid
        for position, handle in self.agent_at_position.items():
            if position is not None:
                frequency = self.agent_freqs[handle]
                malfunction = self.env.agents[handle].malfunction_data['malfunction']
                agent_heading = self.env.agents[handle].direction
                for heading in self.transitions_at(position).exits_tuple:
                    if agent_heading == heading:
                        doppler = 1
                    elif agent_heading == turn(BACK, heading):
                        doppler = -1
                    else:
                        doppler = 0
                    self.sounds.append(Sound(position=position,
                                             frequency=frequency,
                                             doppler=doppler,
                                             heading=heading,
                                             malfunction=malfunction,
                                             **self.sound_parameters))

        # Propagate the sounds in the graph
        while len(self.sounds) > 0:
            self.propagate_sound(self.sounds.popleft(), save_to_node=False)

        # Get the encodings
        return {handle: ear.encode() for handle, ear in self.ears.items()}

    def reset(self):
        # Reset internal values
        self.graph = nx.Graph()
        self.agent_freqs = {}
        self.agent_at_position = {}
        self.station_freqs = {}
        self.sounds = deque()
        self.ears = {}

        # Save the env's rail grid as a Networkx graph
        grid = self.env.rail.grid
        grid_shape = grid.shape
        for row in range(grid_shape[0]):
            for column in range(grid_shape[1]):
                current_cell = grid[row, column]  # is a uint16
                if current_cell:
                    transitions = TransitionMatrix(current_cell)  # is a 4x4 matrix
                    self.graph.add_node((row, column), transitions=transitions, sounds=[])  # Add or update
                    for connection, increment in zip(transitions.exits, GRID_INCREMENTS):
                        if connection:
                            self.graph.add_edge((row, column), (row + increment[0], column + increment[1]))

        # Update the positions & frequencies of stations
        targets = defaultdict(list)
        for agent in self.env.agents:
            targets[agent.target].append(agent.handle)
        self.station_freqs = {position: -frequency for frequency, position in enumerate(targets)}

        # Update the frequencies of agents
        for station, agent_handles in targets.items():
            n = len(agent_handles)
            for i, agent_handle in enumerate(agent_handles):
                self.agent_freqs[agent_handle] = -self.station_freqs[station] + (i+1)/(n+2)

        # Create ears
        for agent in self.env.agents:
            self.ears[agent.handle] = self.ear_constructor(agent,
                                                           agent_frequency=self.agent_freqs[agent.handle],
                                                           **self.ear_parameters)

        # Generate sounds for all possible station exits
        for position, frequency in self.station_freqs.items():
            for heading in self.transitions_at(position).exits_tuple:
                self.sounds.append(Sound(position=position,
                                         frequency=frequency,
                                         doppler=0,
                                         heading=heading,
                                         **self.sound_parameters))

        # Propagate and save the sounds in the graph
        while len(self.sounds) > 0:
            self.propagate_sound(self.sounds.popleft(), save_to_node=True)

    def propagate_sound(self, sound: Sound, save_to_node=False) -> None:
        for _ in range(self.max_propagation_steps):
            visited = sound.step()
            if visited:  # The new node-heading pair has already been visited - extinguish
                break
            if save_to_node:
                self.append_sound(sound.copy())
            if sound.position in self.agent_at_position:
                handle = self.agent_at_position[sound.position]
                self.ears[handle].listen(sound)
            cell = self.transitions_at(sound.position)
            exits = [exit_ for exit_ in cell.exits_tuple if exit_ != turn(BACK, sound.heading)]
            if len(exits) == 0:  # Dead end - extinguish
                break
            elif len(exits) == 1:  # Straight - Continue propagating in the same direction, not necessarily same heading
                sound.heading = exits[0]
            else:  # Crossing - Split and continue straight if possible
                if not sound.is_exhausted():
                    for exit_ in exits:
                        if exit_ != sound.heading:
                            is_legal = cell.can_transition(old_heading=sound.heading, new_heading=exit_)
                            direction = rel_direction(new_heading=exit_, old_heading=sound.heading)
                            new_sound = sound.split(turn_=direction, is_legal=is_legal)
                            self.sounds.append(new_sound)
                if sound.heading not in exits:  # Cannot continue straight - extinguish
                    break
        else:
            raise RuntimeError(f"Propagation loop didn't finish after {self.max_propagation_steps} steps.")
