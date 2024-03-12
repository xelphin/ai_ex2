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
        self.my_robot_has_package = self.my_robot.package
        self.unpicked_packages = [package for package in env.packages if package.on_board==True]
        self.charge_stations = env.charge_stations
        self.num_steps_left = env.num_steps/2

    def robot_has_package(self):
        return self.my_robot_has_package is not None
    
    def robot_position(self):
        return self.my_robot.position

    def dist_R_Pi(self, package_index):
        # Distance between my_robot and package_i
        if package_index >= len(self.unpicked_packages) or package_index < 0:
            return None
        return manhattan_distance(self.my_robot.position, self.unpicked_packages[package_index].position)
    
    def dist_Pi_Di(self, package_index):
        # Distance between package_i_position and package_i_destination
        if package_index >= len(self.unpicked_packages) or package_index < 0:
            return None
        return manhattan_distance(self.unpicked_packages[package_index].position, self.unpicked_packages[package_index].destination)
    
    def dist_ROther_Pi(self, package_index):
        # Distance between other_robot and package_i
        if package_index >= len(self.unpicked_packages) or package_index < 0:
            return None
        return manhattan_distance(self.other_robot.position, self.unpicked_packages[package_index].position)
    
    def dist_R_Ct(self, charging_station_index):
        # Distance between my_robot and charging_station_t
        if charging_station_index >= len(self.charge_stations) or charging_station_index < 0:
            return None
        return manhattan_distance(self.my_robot.position, self.charge_stations[charging_station_index].position)
    
    def dist_R_X(self):
        # Distance between my_robot and destination of package it picked up
        if self.my_robot_has_package is None:
            return None
        return manhattan_distance(self.my_robot.position, self.my_robot_has_package.destination)
    
    def is_worth_it_p(self, package_index):
        # True: Can get to package position and then to destination before battery dying
        if package_index >= len(self.unpicked_packages) or package_index < 0:
            return None
        return self.dist_R_Pi(package_index)+self.dist_Pi_Di(package_index) <= self.my_robot.battery
    
    def worth_it_packages(self):
        package_indices = []
        for i in range(len(self.unpicked_packages)):
            if self.is_worth_it_p(i) and self.other_robot.position != self.unpicked_packages[i].position:
                package_indices = package_indices + [i]
        return package_indices
    
    def is_worth_it_x(self):
        # True: Can get to package destination before battery dying
        if self.my_robot_has_package is None:
            return None
        return self.dist_R_X() <= self.my_robot.battery
        
    def is_on_charging_station(self):
        return self.my_robot.position == self.charge_stations[0].position or self.my_robot.position == self.charge_stations[1].position
    
    def should_charge(self):
        # TODO: Not sure how to calculate if it is worth charging robot
        # Need to make sure there's enough steps for me to accomplish what I want
        if self.my_robot.credit + self.my_robot.battery < self.num_steps_left/3:
            # Need to be able to get to charging station
            if self.dist_R_Ct(0) < self.my_robot.battery or self.dist_R_Ct(1) < self.my_robot.battery:
                if self.my_robot.credit > 9:
                    # Need to have enough credit to get p->d and then go back to charge
                    return 1
        return 0
    
    def win_while_ahead(self):
        # If other robot is dead and we have more points than them, return true
        return self.my_robot.credit > self.other_robot.credit and self.other_robot.battery == 0
        

def smart_heuristic(env: WarehouseEnv, robot_id: int):
    info = Heuristic_Info(env, robot_id)
    max_value = float('-inf')
    new_value = 0 # keep for prints

    # Other robot is dead and we have more points
    if info.win_while_ahead():
        # print("Win while ahead")
        return float('inf') # will pick North/South/East/West

    # Robot doesn't have a package
    if not info.robot_has_package():
        worth_it_packages = info.worth_it_packages()
        # If there is a worth it package (can get to it and destination on time) then go to it
        if len(worth_it_packages) != 0:
            for i in range(len(info.unpicked_packages)):
                new_value = -info.dist_R_Pi(i) + info.dist_ROther_Pi(i)*0.75 + (info.my_robot.credit+info.my_robot.battery)*100
                max_value = max(max_value, new_value)
            # print(f"(1) child {info.robot_position()} : {new_value} (to package) num steps left {env.num_steps}")        
        
        # Else if, it is worthwhile to lose all points in order to get battery do that
        elif info.should_charge():
            for t in range(len(env.charge_stations)):
                new_value = -info.dist_R_Ct(t) + (info.my_robot.credit+info.my_robot.battery)*100
                max_value = max(max_value, new_value)
                # print(f"(2) child {info.robot_position()} : {new_value} (to charging station) num steps left {env.num_steps}")

        # Else, steal package from other robot to kill time
        else:
            for i in range(len(info.unpicked_packages)):
                new_value = -info.dist_R_Pi(i) - info.dist_ROther_Pi(i)+ (info.my_robot.credit+info.my_robot.battery)*100
                max_value = max(max_value, new_value)
                # print(f"(3) child {info.robot_position()} : {new_value} (kill time -> steal package) num steps left {env.num_steps}")


    # Robot has a package
    else:
        # If can get to package destination on time, go to it
        if info.is_worth_it_x():
            new_value = 100 - info.dist_R_X() + (info.my_robot.credit+info.my_robot.battery)*100
            max_value = max(max_value, new_value)
            # print(f"(4) child {info.robot_position()} : {new_value} (to package destination) num steps left {env.num_steps}")
        # Else if, it is worth it to charge before going to destination, do that
        elif info.should_charge():
            for t in range(len(env.charge_stations)):
                new_value = 100 - info.dist_R_Ct(t) + (info.my_robot.credit+info.my_robot.battery)*100
                max_value = max(max_value, new_value)
            # print(f"(5) child {info.robot_position()} : {new_value} (to charging station) num steps left {env.num_steps}")
        # Else, steal package from other robot (block other robot path) to kill time
        else:
            for i in range(len(info.unpicked_packages)):
                new_value = -info.dist_R_Pi(i) - info.dist_ROther_Pi(i)+ (info.my_robot.credit+info.my_robot.battery)*100
                max_value = max(max_value, new_value)
            # print(f"(6) child {info.robot_position()} : {new_value} (kill time -> steal package) num steps left {env.num_steps}")

    return max_value

class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


def smart_heuristic_for_minmax(env: WarehouseEnv, robot_id: int):
    my_robot = env.get_robot(robot_id)
    other_robot = env.get_robot(not robot_id)

    # Case game end
    if my_robot.battery == 0 and other_robot.battery == 0:
        if my_robot.credit > other_robot.credit:
            return float('inf')
        elif my_robot.credit < other_robot.credit:
            return float('-inf')
        else:
            return 0
        
    # Case game continue
    # return smart_heuristic(env, robot_id)-(other_robot.battery + other_robot.credit)*100
    return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    # TODO: section b : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


def helper_d_str(depth):
    return "-"*depth

def helper_dfs_limit_pruned(env: WarehouseEnv, agent_id, depth_limit, heuristic, curr_depth, is_robot1 , alpha, beta):

    if depth_limit == curr_depth:
        return (heuristic(env, agent_id), None) # We want to do heurstic from pov of Robot0 (main_player)

    # Get child states
    operators = env.get_legal_operators(int(is_robot1))
    children = [env.clone() for _ in operators]
    for child, op in zip(children, operators):
        child.apply_operator(int(is_robot1), op)

    # # Optimal ordering 
    # sorted_children_op = sorted(children_op, key=lambda x: heuristic(x, agent_id)) 
    # if not is_main_player:
    #     sorted_children_op.reverse()

    # Apply DFS-L
    # TODO: Do i need new_alpha/new_beta or do i recursively use the same alpha/beta everywhere
    new_alpha = float('-inf')
    new_beta = float('inf')
    child_values = []

    for child, op in zip(children, operators):
        print(f"{helper_d_str(curr_depth)} Robot {int(is_robot1)}: {env.get_robot(int(is_robot1)).position} {op} -> {child.get_robot(int(is_robot1)).position}")
        (child_value, ret_op) = helper_dfs_limit_pruned(child, agent_id, depth_limit, heuristic, curr_depth+1, not is_robot1, new_alpha, new_beta)
        child_values += [(child_value, op)]
        print(f"{helper_d_str(curr_depth)} Robot {int(is_robot1)}: {env.get_robot(int(is_robot1)).position} {op} -> {child.get_robot(int(is_robot1)).position} RETURNED {child_value}")

        # Pruning
        if is_robot1:
            if child_value <= alpha:
                print(f"{helper_d_str(curr_depth)} Pruning it ({op}) alpha is {alpha}")
                return (child_value, op)
        else:
            if child_value >= beta:
                print(f"{helper_d_str(curr_depth)} Pruning it ({op}) beta is {beta}")
                return (child_value, op)
            
        # Update alpha, beta
        if child_value > new_alpha:
            new_alpha = child_value
        if child_value < new_beta:
            new_beta = child_value

    # Return best value for player (min/max)
    if is_robot1:
        return min(child_values, key=lambda x: x[0])
    else:
        return max(child_values, key=lambda x: x[0])


class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        alpha = float('-inf')
        beta = float('inf')
        (best_child_value, best_op) = helper_dfs_limit_pruned(env, agent_id, 8, smart_heuristic_for_minmax, 0, False, alpha, beta)
        return best_op


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