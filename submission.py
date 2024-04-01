from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random

MAX_DIST = 10

def get_better_package(env: WarehouseEnv, robot_id: int):
    if not env or not (robot_id == 0 or robot_id == 1):
        return None
    r = env.get_robot(robot_id)
    r_other = env.get_robot(1 - robot_id)
    if r.package is not None:
        return r.package
    p0 = env.packages[0]
    p1 = env.packages[1]
    if not p0.on_board:
        return p1
    if not p1.on_board:
        return p0
    # both pacakges are on board
    # need the package thats closer to me than to the other r
    my_dist_0, my_dist_1 = manhattan_distance(r.position, p0.position), manhattan_distance(r.position, p1.position)
    other_dist_0, other_dist_1 = manhattan_distance(r_other.position, p0.position), manhattan_distance(r_other.position, p1.position)
    if my_dist_0 < other_dist_0 and my_dist_1 > other_dist_1:
        return p0
    elif my_dist_0 > other_dist_0 and my_dist_1 < other_dist_1:
        return p1
    else:
        return p0 if manhattan_distance(p0.position, p0.destination) - my_dist_0 > manhattan_distance(p1.position, p1.destination) - my_dist_1 else p1


# TODO: section a : 3
def smart_heuristic(env: WarehouseEnv, robot_id: int):
    if not env or not (robot_id == 0 or robot_id == 1):
        return None
    r = env.get_robot(robot_id)
    r_other = env.get_robot(1 - robot_id)
    # calc huristic
    # check if robot has package
    h = r.credit + (r.package is not None) * (MAX_DIST - manhattan_distance(r.position, r.package.destination))
    + (r.package is None) * (MAX_DIST - manhattan_distance(r.position, get_better_package(env, robot_id).position))
    # if robot is losing and doesn't have enough battery to reach package/dest, go to charging station
    if r.credit < r_other.credit and r.battery < manhattan_distance(r.position, get_better_package(env, robot_id).position) + manhattan_distance(get_better_package(env, robot_id).position, get_better_package(env, robot_id).destination):
        h = r.battery + MAX_DIST - min(manhattan_distance(r.position, env.charging_stations[0]), manhattan_distance(r.position, env.charging_stations[1]))
    return h


class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    # TODO: section b : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        


class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentExpectimax(Agent):
    # TODO: section d : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = ["move north", "move east", "move north", "move north", "pick_up", "move east", "move east",
                           "move south", "move south", "move south", "move south", "drop_off"]

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)