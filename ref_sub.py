from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random
import numpy as np
import time

MAX_DIST = 10


def get_charge_station_dist(env: WarehouseEnv, robot_id):
    agent = env.get_robot(robot_id)
    return min(manhattan_distance(agent.position, env.charge_stations[0].position),
               manhattan_distance(agent.position, env.charge_stations[1].position))


def get_best_package_position(env: WarehouseEnv, robot_id: int):
    agent = env.get_robot(robot_id)
    possible_package = []
    packages = [p for p in env.packages if p.on_board]
    for p in packages:
        dist = manhattan_distance(agent.position, p.position) + manhattan_distance(p.position, p.destination)
        if dist < agent.battery:
            possible_package.append(p)
    if len(possible_package) == 0:
        return None
    if len(possible_package) == 1:
        return possible_package[0].position
    p0 = possible_package[0]
    p0_value = 2 * manhattan_distance(p0.position, p0.destination) - manhattan_distance(p0.position, agent.position)
    p1 = possible_package[1]
    p1_value = 2 * manhattan_distance(p1.position, p1.destination) - manhattan_distance(p1.position, agent.position)
    if p0_value > p1_value:
        return p0.position
    return p1.position


def get_closest_package(env: WarehouseEnv, robot_id: int):
    agent = env.get_robot(robot_id)
    packages = [p for p in env.packages if p.on_board]
    min_dist = np.inf
    package_position = None
    for p in packages:
        if manhattan_distance(agent.position, p.position) < min_dist:
            package_position = p.position
    return package_position


def robot_loosing(env: WarehouseEnv, robot_id: int):
    agent = env.get_robot(robot_id)
    rival_robot = env.get_robot((robot_id + 1) % 2)
    return rival_robot.credit > agent.credit


def smart_heuristic_value(env: WarehouseEnv, robot_id: int):
    agent = env.get_robot(robot_id)
    agent_battery = agent.battery
    agent_credit = agent.credit
    desire_package = get_best_package_position(env, robot_id)
    if desire_package is None and robot_loosing(env, robot_id):
        return agent_battery * 1000 - (2 * get_charge_station_dist(env, robot_id))
    if agent.package is not None:
        return ((agent_credit * 1000) + 2 * manhattan_distance(agent.package.position, agent.package.destination)
                - manhattan_distance(agent.position, agent.package.destination))
    elif desire_package is None:
        desire_package = get_closest_package(env, robot_id)
    return agent_credit * 1000 - manhattan_distance(agent.position, desire_package)


def smart_heuristic(env: WarehouseEnv, robot_id: int):
    robot_h = smart_heuristic_value(env, robot_id)
    rival_h = smart_heuristic_value(env, (robot_id + 1) % 2)
    return robot_h - rival_h


def check_timeout(start_time, time_limit, depth):
    # if depth == 0:
    #     return True
    if time.time() - start_time >= time_limit - 0.05 or depth == 0:
        return True
    return False


class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    # TODO: section b : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        start_time = time.time()
        current_max = -np.inf
        action = None
        depth = 1
        operators = env.get_legal_operators(agent_id)
        while time.time() - start_time < time_limit - 0.05:
            children = [env.clone() for _ in operators]
            for child, op in zip(children, operators):
                child.apply_operator(agent_id, op)
                value = self.rb_minmax(child, agent_id, time_limit, start_time, (1 + agent_id) % 2, depth - 1)
                if value > current_max:
                    current_max = value
                    action = op
                if current_max == np.inf:
                    break
            depth += 1
        return action

    def rb_minmax(self, env: WarehouseEnv, agent_id, time_limit, start_time, turn, depth):
        if check_timeout(start_time, time_limit, depth):
            return smart_heuristic(env, agent_id)
        if env.done():
            if env.get_robot(agent_id).credit > env.get_robot((agent_id + 1) % 2).credit:
                return np.inf
            elif env.get_robot(agent_id).credit < env.get_robot((agent_id + 1) % 2).credit:
                return -np.inf
            else:
                return 0
        actions = env.get_legal_operators(turn)
        children = [env.clone() for _ in actions]
        if turn == agent_id:
            cur_max = -np.inf
            for child, op in zip(children, actions):
                child.apply_operator(turn, op)
                value = self.rb_minmax(child, agent_id, time_limit, start_time, (turn + 1) % 2, depth - 1)
                if value > cur_max:
                    cur_max = value
            return cur_max
        else:
            cur_min = np.inf
            for child, op in zip(children, actions):
                child.apply_operator(turn, op)
                value = self.rb_minmax(child, agent_id, time_limit, start_time, (turn + 1) % 2, depth - 1)
                if value < cur_min:
                    cur_min = value
            return cur_min


class AgentAlphaBeta(Agent):
    # TODO: section c : 1

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        start_time = time.time()
        current_max = -np.inf
        action = None
        depth = 1
        operators = env.get_legal_operators(agent_id)
        while time.time() - start_time < time_limit - 0.05:
            children = [env.clone() for _ in operators]
            for child, op in zip(children, operators):
                child.apply_operator(agent_id, op)
                value = self.ab_rb_minmax(child, agent_id, time_limit, start_time, (agent_id + 1) % 2, depth - 1,
                                          alpha=-np.inf, beta=np.inf)
                if value > current_max:
                    current_max = value
                    action = op
                if current_max == np.inf:
                    break
            depth += 1
        return action

    def ab_rb_minmax(self, env: WarehouseEnv, agent_id, time_limit, start_time, turn, depth, alpha, beta):
        if check_timeout(start_time, time_limit, depth):
            return smart_heuristic(env, agent_id)
        if env.done():
            if env.get_robot(agent_id).credit > env.get_robot((agent_id + 1) % 2).credit:
                return np.inf
            elif env.get_robot(agent_id).credit < env.get_robot((agent_id + 1) % 2).credit:
                return -np.inf
            else:
                return 0
        actions = env.get_legal_operators(turn)
        children = [env.clone() for _ in actions]
        if turn == agent_id:
            cur_max = -np.inf
            for child, op in zip(children, actions):
                child.apply_operator(turn, op)
                value = self.ab_rb_minmax(child, agent_id, time_limit, start_time, (turn + 1) % 2, depth - 1, alpha,
                                          beta)
                if value > cur_max:
                    cur_max = value
                alpha = max(cur_max, alpha)
                if cur_max >= beta:
                    return np.inf
            return cur_max
        else:
            cur_min = np.inf
            for child, op in zip(children, actions):
                child.apply_operator(turn, op)
                value = self.ab_rb_minmax(child, agent_id, time_limit, start_time, (turn + 1) % 2, depth - 1, alpha,
                                          beta)
                if value < cur_min:
                    cur_min = value
                beta = min(cur_min, beta)
                if cur_min <= alpha:
                    return -np.inf
            return cur_min


class AgentExpectimax(Agent):
    # TODO: section d : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        start_time = time.time()
        current_max = -np.inf
        action = None
        depth = 1
        operators = env.get_legal_operators(agent_id)
        while time.time() - start_time < time_limit - 0.05:
            children = [env.clone() for _ in operators]
            for child, op in zip(children, operators):
                child.apply_operator(agent_id, op)
                value = self.rb_expectimax(child, agent_id, time_limit, start_time, (agent_id + 1) % 2, depth - 1)
                if value > current_max:
                    current_max = value
                    action = op
                if current_max == np.inf:
                    break
            depth += 1
        return action

    def rb_expectimax(self, env: WarehouseEnv, agent_id, time_limit, start_time, turn, depth):
        if check_timeout(start_time, time_limit, depth):
            return smart_heuristic(env, agent_id)
        if env.done():
            if env.get_robot(agent_id).credit > env.get_robot((agent_id + 1) % 2).credit:
                return np.inf
            elif env.get_robot(agent_id).credit < env.get_robot((agent_id + 1) % 2).credit:
                return -np.inf
            else:
                return 0
        actions = env.get_legal_operators(turn)
        children = [env.clone() for _ in actions]
        if turn == agent_id:
            cur_max = -np.inf
            for child, op in zip(children, actions):
                child.apply_operator(turn, op)
                value = self.rb_expectimax(child, agent_id, time_limit, start_time, (turn + 1) % 2, depth - 1)
                if value > cur_max:
                    cur_max = value
            return cur_max
        else:
            value = 0
            counter = 0
            if 'move east' in actions:
                counter += 1
            if 'pick up' in actions:
                counter += 1
            for child, op in zip(children, actions):
                p = 1 / (len(actions) + counter)
                child.apply_operator(turn, op)
                if op == 'move east' or op == 'pick up':
                    p *= 2
                value += p * self.rb_expectimax(child, agent_id, time_limit, start_time, (turn + 1) % 2, depth - 1)
            return value


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
