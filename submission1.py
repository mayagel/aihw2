from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random
import time

MAX_DIST = 10

def get_better_package(env: WarehouseEnv, robot_id: int):
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

def better_package(env: WarehouseEnv, robot_id: int):
    r = env.get_robot(robot_id)`
    packages = [p for p in env.packages if p.on_board and manhattan_distance(r.position, p.position) +  manhattan_distance(p.position, p.destination)< r.battery]
    packages.sort(key=lambda p: manhattan_distance(p.position, p.destination) - manhattan_distance(r.position, p.position))
    return packages[-1] if packages else None

# TODO: section a : 3
def agent_val(env: WarehouseEnv, robot_id: int):
    if not env or not (robot_id == 0 or robot_id == 1):
        return None
    r = env.get_robot(robot_id)
    r_other = env.get_robot(1 - robot_id)
    # if robot is losing and doesn't have enough battery to reach package/dest, go to charging station
    if r.credit < r_other.credit and r.battery < manhattan_distance(r.position, better_package(env, robot_id).position) + manhattan_distance(get_better_package(env, robot_id).position, get_better_package(env, robot_id).destination):
        h = r.battery + MAX_DIST - min(manhattan_distance(r.position, env.charging_stations[0]), manhattan_distance(r.position, env.charging_stations[1]))
    # not losing or has enough battery, check if robot has a package
    elif r.package is not None:
        h = r.credit + manhattan_distance(r.package.position, r.package.destination)
        - manhattan_distance(r.position, r.package.position)
    else:
        h = r.credit - manhattan_distance(r.position, better_package(env, robot_id).position)
    return h

def smart_heuristic(env: WarehouseEnv, robot_id: int):
    return agent_val(env, robot_id) - agent_val(env, 1 - robot_id)

class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    # TODO: section b : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        start_t = time.time()
        max_val = float('-inf')
        step = None
        depth = 1
        while time.time() - start_t < time_limit - 0.05:
            print(f"depth: {depth}")
            children = [env.clone() for _ in env.get_legal_operators(agent_id)]
            for child, op in zip(children, env.get_legal_operators(agent_id)):
                child.apply_operator(agent_id, op)
                val = self.minimax(child, agent_id, 1 - agent_id, depth, start_t, time_limit)
                step = op if val > max_val else step
                max_val = max(val, max_val)
                if max_val == float('inf'):
                    break
            depth += 1
        return step
   
    def minimax(self, env: WarehouseEnv, agent_id, turn, depth, t_start, t_limit, alpha=float('-inf'), beta=float('inf')):
        if depth == 0 or time.time() - t_start >= t_limit - 0.05:
            return smart_heuristic(env, agent_id)
        r_credit = env.get_robot(agent_id).credit
        other_credit = env.get_robot(1 - agent_id).credit
        if env.done():
            if r_credit != other_credit:
                return float('inf') if r_credit > other_credit else float('-inf')
            return 0 
        children = [env.clone() for _ in env.get_legal_operators(agent_id)]
        curr_val = float('-inf') if turn == agent_id else float('inf')
        for child, op in zip(children, env.get_legal_operators(agent_id)):
            child.apply_operator(agent_id, op)
            val = self.minimax(child, agent_id, 1 - turn, depth - 1, t_start, t_limit)
            curr_val = max(val, curr_val) if turn == agent_id else min(val, curr_val)
            if type(self) == AgentAlphaBeta:
                if turn == agent_id:
                    alpha = max(alpha, curr_val)
                else:
                    beta = min(beta, curr_val)
                if beta <= alpha:
                    return float('inf') if turn == agent_id else float('-inf')
        return curr_val


class AgentAlphaBeta(AgentMinimax):
    # TODO: section c : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        return super().run_step(env, agent_id, time_limit)

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