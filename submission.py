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
        self.num_steps_left = env.num_steps

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
    
    def there_is_a_worth_it_package(self):
        for i in range(len(self.unpicked_packages)):
            if self.is_worth_it_p(i):
                return True
        return False
    
    def is_worth_it_x(self):
        # True: Can get to package destination before battery dying
        if self.my_robot_has_package is None:
            return None
        return self.dist_R_X() <= self.my_robot.battery
        
    def is_on_charging_station(self):
        return self.my_robot.position == self.charge_stations[0].position or self.my_robot.position == self.charge_stations[1].position
    
    def should_charge(self):
        return self.my_robot.battery + self.my_robot.credit < self.num_steps_left/2
        

def smart_heuristic(env: WarehouseEnv, robot_id: int):
    info = Heuristic_Info(env, robot_id)
    max_value = float('-inf')

    # Robot doesn't have a package
    if not info.robot_has_package():
        # If there is a worth it package (can get to it and destination on time) then go to it
        if info.there_is_a_worth_it_package():
            for i in range(len(info.unpicked_packages)):
                    new_value = -info.dist_R_Pi(i) + info.dist_ROther_Pi(i)*0.75 + (info.my_robot.credit+info.my_robot.battery)*100
                    max_value = max(max_value, new_value)
                    # print(f"(1) child {info.robot_position()} : {new_value} (to package)")
        
        # Else if, it is worthwhile to lose all points in order to get battery do that
        elif info.should_charge():
            for t in range(len(env.charge_stations)):
                new_value = -info.dist_R_Ct(t) + (info.my_robot.credit+info.my_robot.battery)*100
                max_value = max(max_value, new_value)
                # print(f"(2) child {info.robot_position()} : {new_value} (to charging station)")

        # Else, steal package from other robot to kill time
        else:
            for i in range(len(info.unpicked_packages)):
                new_value = -info.dist_R_Pi(i) - info.dist_ROther_Pi(i) + (info.my_robot.credit+info.my_robot.battery)*100
                max_value = max(max_value, new_value)
                # print(f"(3) child {info.robot_position()} : {new_value} (kill time -> steal package)")


    # Robot has a package
    else:
        # If can get to package destination on time, go to it
        if info.is_worth_it_x():
            new_value = 100 - info.dist_R_X() + (info.my_robot.credit+info.my_robot.battery)*100 
            max_value = max(max_value, new_value)
            # print(f"(4) child {info.robot_position()} : {new_value} (to package destination)")
        # Else if, it is worth it to charge before going to destination, do that
        elif info.should_charge():
            for t in range(len(env.charge_stations)):
                new_value = 100 - info.dist_R_Ct(t) + (info.my_robot.credit+info.my_robot.battery)*100
                max_value = max(max_value, new_value)
                # print(f"(5) child {info.robot_position()} : {new_value} (to charging station)")
        # Else, steal package from other robot (block other robot path) to kill time
        else:
            for i in range(len(info.unpicked_packages)):
                    new_value = -info.dist_R_Pi(i) - info.dist_ROther_Pi(i) + (info.my_robot.credit+info.my_robot.battery)*100
                    max_value = max(max_value, new_value)
                    # print(f"(6) child {info.robot_position()} : {new_value} (kill time -> steal package)")


    # print(f"-> child {info.robot_position()} : {max_value}")
    return max_value

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