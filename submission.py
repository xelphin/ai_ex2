from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random


# TODO: section a : 3

class Heuristic_Info():
    # Helper for heuristic, this way smart_heuristic() is more readable
    def __init__(self, env: WarehouseEnv, robot_id: int) -> None:
        self.env = env
        self.robot_id = robot_id
        self.my_robot = env.get_robot(robot_id)
        self.other_robot = env.get_robot(not robot_id)
        self.my_robot_package = self.my_robot.package
        self.not_my_robot_package = self.other_robot.package
        self.unpicked_packages = [package for package in env.packages if package.on_board==True]
        self.charge_stations = env.charge_stations
        self.num_steps_left = env.num_steps/2

    def robot_has_package(self):
        return (self.my_robot_package is not None)
        
    def other_robot_has_package(self):
        return (self.not_my_robot_package is not None)

    def dist_R_Pi(self, package_index):
        # Distance between my_robot and package_i
        if package_index >= len(self.unpicked_packages) or package_index < 0:
            return None
        return manhattan_distance(self.my_robot.position, self.unpicked_packages[package_index].position)
    
    def dist_ROther_Pi(self, package_index):
        # Distance between other_robot and package_i
        if package_index >= len(self.unpicked_packages) or package_index < 0:
            return None
        return manhattan_distance(self.other_robot.position, self.unpicked_packages[package_index].position)
    
    def dist_R_X(self):
        # Distance between my_robot and destination of package it picked up
        if self.my_robot_package is None:
            return None
        return manhattan_distance(self.my_robot.position, self.my_robot_package.destination)
        
    def dist_ROther_X(self):
        # Distance between my_robot and destination of package it picked up
        if self.not_my_robot_package is None:
            return None
        return manhattan_distance(self.other_robot.position, self.not_my_robot_package.destination) 
    
    def win_while_ahead(self):
        # If other robot is dead and we have more points than them, return true
        return self.my_robot.credit > self.other_robot.credit and self.other_robot.battery == 0
    
    def has_lost(self):
        return self.my_robot.credit < self.other_robot.credit and self.my_robot.battery == 0
    
    def closeness_to_package(self):
        if self.robot_has_package():
            return self.dist_R_X()
        else:
            min_p_dist = float('inf')
            for i in range(len(self.unpicked_packages)):
                min_p_dist = min(min_p_dist, self.dist_R_Pi(i))
            if min_p_dist == float('inf'):
                return float('-inf')
            return min_p_dist
        
    def other_closeness_to_package(self):
        if self.other_robot_has_package():
            return self.dist_ROther_X()
        else:
            min_p_dist = float('inf')
            for i in range(len(self.unpicked_packages)):
                min_p_dist = min(min_p_dist, self.dist_ROther_Pi(i))
            if min_p_dist == float('inf'):
                return float('-inf')
            return min_p_dist
 

def smart_heuristic(env: WarehouseEnv, robot_id: int):

    info = Heuristic_Info(env, robot_id)
    h_val = 0
    my_robot = env.get_robot(robot_id)
    other_robot = env.get_robot(not robot_id)

    # Other robot is dead and we have more points
    if info.win_while_ahead():
        return float('inf') # will pick North/South/East/West
    
    # We are dead and other robot has more points
    if info.has_lost():
        return float('-inf')
    
    # print(f"For {my_robot.position}:")
    
    h_val += 100*(my_robot.credit-other_robot.credit)
    # print(f"-- [credits +{100*(my_robot.credit-other_robot.credit)}]")
    h_val += 50*(my_robot.battery-other_robot.battery)
    # print(f"-- [battery +{50*(my_robot.battery-other_robot.battery)}]")
    h_val += 100*(int(info.robot_has_package()) - int(info.other_robot_has_package())) # -1 <= val <= 1
    # print(f"-- [has package +{100*(int(info.robot_has_package()) - int(info.other_robot_has_package()))}]")
    h_val += -info.closeness_to_package() + info.other_closeness_to_package()
    # print(f"-- [closeness us: -{info.closeness_to_package()} other: +{info.other_closeness_to_package()}")

    # print(f"For {my_robot.position} val {h_val}")

    return h_val


class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    # TODO: section b : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


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