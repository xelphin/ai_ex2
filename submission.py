from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random
import time
import threading

#import signal
from multiprocessing import Process

# TODO: section a : 3


class TimeoutException(Exception):
    pass


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
        self.not_my_robot_package = self.other_robot.package

    def robot_has_package(self):
        return self.my_robot_has_package is not None
        
    def other_robot_has_package(self):
        return (self.not_my_robot_package is not None)

    
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
        if self.env.num_steps==0 and self.my_robot.credit > self.other_robot.credit:
            return True

        return self.my_robot.credit > self.other_robot.credit and self.other_robot.battery == 0
    
    def has_lost(self):
        if self.env.num_steps==0 and self.my_robot.credit < self.other_robot.credit:
            return True
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
    
    def dist_R_X(self):
        # Distance between my_robot and destination of package it picked up
        if self.my_robot_has_package is None:
            return None
        return manhattan_distance(self.my_robot.position, self.my_robot_has_package.destination)
        
    def dist_ROther_X(self):
        # Distance between my_robot and destination of package it picked up
        if self.not_my_robot_package is None:
            return None
        return manhattan_distance(self.other_robot.position, self.not_my_robot_package.destination) 
    
        

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
                new_value = -info.dist_R_Pi(i) + info.dist_ROther_Pi(i)*0.75 + (info.my_robot.credit+info.my_robot.battery*0.8)*1000
                max_value = max(max_value, new_value)
            # print(f"(1) child {info.robot_position()} : {new_value} (to package) num steps left {env.num_steps}")        
        
        # Else if, it is worthwhile to lose all points in order to get battery do that
        elif info.should_charge():
            for t in range(len(env.charge_stations)):
                new_value = -info.dist_R_Ct(t) + (info.my_robot.credit+info.my_robot.battery*0.8)*1000
                max_value = max(max_value, new_value)
                # print(f"(2) child {info.robot_position()} : {new_value} (to charging station) num steps left {env.num_steps}")

        # Else, steal package from other robot to kill time
        else:
            for i in range(len(info.unpicked_packages)):
                new_value = -info.dist_R_Pi(i) - info.dist_ROther_Pi(i)+ (info.my_robot.credit+info.my_robot.battery*0.8)*1000
                max_value = max(max_value, new_value)
                # print(f"(3) child {info.robot_position()} : {new_value} (kill time -> steal package) num steps left {env.num_steps}")


    # Robot has a package
    else:
        # If can get to package destination on time, go to it
        if info.is_worth_it_x():
            new_value = 100 - info.dist_R_X() + (info.my_robot.credit+info.my_robot.battery*0.8)*1000
            max_value = max(max_value, new_value)
            # print(f"(4) child {info.robot_position()} : {new_value} (to package destination) num steps left {env.num_steps}")
        # Else if, it is worth it to charge before going to destination, do that
        elif info.should_charge():
            for t in range(len(env.charge_stations)):
                new_value = 100 - info.dist_R_Ct(t) + (info.my_robot.credit+info.my_robot.battery*0.8)*1000
                max_value = max(max_value, new_value)
            # print(f"(5) child {info.robot_position()} : {new_value} (to charging station) num steps left {env.num_steps}")
        # Else, steal package from other robot (block other robot path) to kill time
        else:
            for i in range(len(info.unpicked_packages)):
                new_value = -info.dist_R_Pi(i) - info.dist_ROther_Pi(i)+ (info.my_robot.credit+info.my_robot.battery*0.8)*1000
                max_value = max(max_value, new_value)
            # print(f"(6) child {info.robot_position()} : {new_value} (kill time -> steal package) num steps left {env.num_steps}")

    return max_value


def smart_heuristic2(env: WarehouseEnv, robot_id: int):

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
    #print(f"-- [credits +{100*(my_robot.credit-other_robot.credit)}]")
    h_val += 50*(my_robot.battery-other_robot.battery)
    #print(f"-- [battery +{50*(my_robot.battery-other_robot.battery)}]")
    h_val += 25*(int(info.robot_has_package()) - int(info.other_robot_has_package())) # -1 <= val <= 1
    #print(f"-- [has package +{100*(int(info.robot_has_package()) - int(info.other_robot_has_package()))}]")
    h_val += -info.closeness_to_package() + info.other_closeness_to_package()
    #print(f"-- [closeness us: -{info.closeness_to_package()} other: +{info.other_closeness_to_package()}")

    # print(f"For {my_robot.position} val {h_val}")

    return h_val


class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic2(env, robot_id)


class AgentMinimax(Agent):
    # TODO: section b : 1


    def __init__(self):
        self.best_op = 'park'
        self.env = None


    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic2(env, robot_id)
    
    
    def helper(self, env: WarehouseEnv, agent_id, depth, current_id, max_depth,time_limit, time_started):

        if (time.time()-time_started>=time_limit):
            raise TimeoutException

        if (self.heuristic(env,agent_id) ==float('-inf') or self.heuristic(env,agent_id) ==float('inf')):
            return (self.heuristic(env,agent_id), None)
        if (depth==max_depth):
            return (self.heuristic(env,agent_id), None)
        operators = env.get_legal_operators(current_id)
        
        children = [env.clone() for _ in operators]
        options = []
        
        for child, op in zip(children, operators):
            child.apply_operator(current_id, op)
            (child_val, act) = self.helper(child, agent_id, depth+1, not current_id, max_depth,time_limit, time_started)
            options.append((child_val, op))
        


        if current_id==agent_id:
            return max(options, key=lambda x: x[0])
        else:
            return min(options, key=lambda x: x[0])
        

    def helper_aux(self, agent_id, max_depth,time_limit, time_started):
        (best_value, best_op22)=self.helper(self.env, agent_id, 0, agent_id, max_depth,time_limit, time_started)
        self.best_op= best_op22



    def run_step(self, env: WarehouseEnv, agent_id, time_limit):

        time_started = time.time()
        time_limit=0.9*time_limit
        self.env = env

        max_depth=2
        forced_parking = False

        (best_value, self.best_op) = (None, 'park')

        try:
            while True:
                if (env.num_steps < max_depth):
                    forced_parking = True
                    break
                
                #thread = Process(target=self.helper_aux, args=(agent_id, depth, agent_id, max_depth,))
                #thread.start()
                #thread.join(timeout=time_limit)
                helper_thread = threading.Thread(target=self.helper_aux, args=(agent_id, max_depth, time_limit, time_started))
                helper_thread.start()
                helper_thread.join(timeout=time_limit)
                max_depth+=2
                if time_limit<=time.time()-time_started:
                    break

        except TimeoutException as e:
            pass
        print(max_depth)        
        # not sure why is it None
        if forced_parking or self.best_op==None:
            operators = env.get_legal_operators(agent_id)
            if 'park' in operators:
                return 'park'
            else:
                for i in operators:
                    if i.startswith('move'):
                        return i

        
        return self.best_op


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