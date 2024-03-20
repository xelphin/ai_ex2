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
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    # TODO: section b : 1


    def __init__(self):
        self.best_op = 'park'
        self.env = None


    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)
    
    def greedy(self, agent_id):
        operators = self.env.get_legal_operators(agent_id)
        
        children = [self.env.clone() for _ in operators]
        options = []
        
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
            child_val = self.heuristic(self.env, agent_id)
            options.append((child_val, op))
        
        (best_child, best_op) = max(options, key=lambda x: x[0])
        return best_op

    
    def helper(self, env: WarehouseEnv, agent_id, depth, current_id, max_depth,time_limit, time_started):

        if (time.time()-time_started>=time_limit):
            raise TimeoutException
        
        operators = env.get_legal_operators(current_id)
        if (self.heuristic(env,agent_id) ==float('-inf') or self.heuristic(env,agent_id) ==float('inf')):
            return (self.heuristic(env,agent_id), operators[0])
        if (depth==max_depth):
            return (self.heuristic(env,agent_id), None)
        
        
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
        max_depth=1
        if env.num_steps< 2: # last move should be greedy
            return self.greedy(agent_id)

        operators = env.get_legal_operators(agent_id)
        (best_value, self.best_op) = (None, operators[0])

        try:
            while True:
                if (env.num_steps < max_depth):
                    break
                self.helper_aux(agent_id, max_depth, time_limit, time_started)
                max_depth+=1
                if time_limit<=time.time()-time_started:
                    break

        except TimeoutException as e:
            pass        
        # not sure why is it None
        if self.best_op==None: # sholdnt enter here
            return self.greedy(agent_id)

     
        return self.best_op


class AgentAlphaBeta(Agent):

    def __init__(self):
        self.best_op = 'park'
        self.env = None


    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)
    
    def greedy(self, agent_id):
        operators = self.env.get_legal_operators(agent_id)
        
        children = [self.env.clone() for _ in operators]
        options = []
        
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
            child_val = self.heuristic(self.env, agent_id)
            options.append((child_val, op))
        
        (best_child, best_op) = max(options, key=lambda x: x[0])
        return best_op
    
    def sort_by_shallow_heuristic(self, children, operators, current_id, agent_id):
        # Sort children by shallow heuristic
        child_op_heuristic = []
        for child, op in zip(children, operators):
            child.apply_operator(current_id, op)
            child_heuristic = self.heuristic(child, agent_id)
            child_op_heuristic += [(child, op, child_heuristic)]
        
        # sort smallest heursitic to highest
        sorted_child_op_heuristic = sorted(child_op_heuristic, key=lambda x: x[2])

        if (current_id is agent_id):
            sorted_child_op_heuristic.reverse()

        # Return only (child, op)
        return [(x[0], x[1]) for x in sorted_child_op_heuristic]
    
    def max_value(self, env: WarehouseEnv, agent_id, depth, current_id, max_depth,time_limit, time_started, alpha, beta):

        assert (current_id == agent_id) # TODO: Erase

        if (time.time()-time_started>=time_limit):
            raise TimeoutException
        
        operators = env.get_legal_operators(current_id)

        if (self.heuristic(env,agent_id) == float('-inf') or self.heuristic(env,agent_id) == float('inf')):
            return (self.heuristic(env,agent_id), operators[0])
        
        if (depth == max_depth):
            return (self.heuristic(env,agent_id), None)
        
        
        children = [env.clone() for _ in operators]
        sorted_child_op_zip = self.sort_by_shallow_heuristic(children, operators, current_id, agent_id)

        options = []
        for child, op in sorted_child_op_zip:
            (child_val, act) = self.min_value(child, agent_id, depth+1, not current_id, max_depth, time_limit, time_started, alpha, beta)
            options.append((child_val, op))

            # Pruning
            if child_val >= beta:
                return (child_val, op)
            alpha = max(alpha, child_val)

        return max(options, key=lambda x: x[0])


    def min_value(self, env: WarehouseEnv, agent_id, depth, current_id, max_depth,time_limit, time_started, alpha, beta):

        assert (current_id != agent_id) # TODO: Erase

        if (time.time()-time_started>=time_limit):
            raise TimeoutException
        
        operators = env.get_legal_operators(current_id)

        if (self.heuristic(env,agent_id) == float('-inf') or self.heuristic(env,agent_id) == float('inf')):
            return (self.heuristic(env,agent_id), operators[0])
        
        if (depth == max_depth):
            return (self.heuristic(env,agent_id), None)
        
        
        children = [env.clone() for _ in operators]
        sorted_child_op_zip = self.sort_by_shallow_heuristic(children, operators, current_id, agent_id)

        options = []
        for child, op in sorted_child_op_zip:
            (child_val, act) = self.max_value(child, agent_id, depth+1, not current_id, max_depth, time_limit, time_started, alpha, beta)
            options.append((child_val, op))

            # Pruning
            if child_val <= alpha:
                return (child_val, op)
            beta = min(beta, child_val)

        return min(options, key=lambda x: x[0])

        

    def helper_aux(self, agent_id, max_depth,time_limit, time_started):
        alpha = float('-inf')
        beta = float('inf')
        (best_value, best_op22)=self.max_value(self.env, agent_id, 0, agent_id, max_depth,time_limit, time_started, alpha, beta)
        self.best_op= best_op22


    def run_step(self, env: WarehouseEnv, agent_id, time_limit):

        time_started = time.time()
        time_limit=0.9*time_limit
        self.env = env
        max_depth=1
        if env.num_steps < 2: # last move should be greedy
            return self.greedy(agent_id)

        operators = env.get_legal_operators(agent_id)
        (best_value, self.best_op) = (None, operators[0])

        try:
            while True:
                if (env.num_steps < max_depth):
                    break
                self.helper_aux(agent_id, max_depth, time_limit, time_started)
                max_depth+=1
                if time_limit<=time.time()-time_started:
                    break

        except TimeoutException as e:
            pass        
        # not sure why is it None
        if self.best_op==None: # sholdnt enter here
            return self.greedy(agent_id)

        # print(f"Reached depth {max_depth}")
        return self.best_op


class AgentExpectimax(Agent):
    def __init__(self):
        self.best_op = 'park'
        self.env = None
        self.agent_id = -1

    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)
    
    def greedy(self):
        operators = self.env.get_legal_operators(self.agent_id)
        
        children = [self.env.clone() for _ in operators]
        options = []
        
        for child, op in zip(children, operators):
            child.apply_operator(self.agent_id, op)
            child_val = self.heuristic(self.env, self.agent_id)
            options.append((child_val, op))
        
        (best_child, best_op) = max(options, key=lambda x: x[0])
        return best_op
    
    def operator_probability(self, operators):

        operator_probability_arr = [] # [(operator, probability)]

        denom = len(operators) + (int('move east' in operators)) + (int('pick up' in operators))

        for op in operators:
            if op == 'move east' or op == 'pick up':
                operator_probability_arr += [(op, (2/denom))]
            else:
                operator_probability_arr += [(op, (1/denom))]

        return operator_probability_arr
    
    def max_value(self, env: WarehouseEnv, depth, current_id, max_depth,time_limit, time_started):

        if (time.time()-time_started>=time_limit):
            raise TimeoutException
        
        operators = env.get_legal_operators(current_id)

        if (self.heuristic(env,self.agent_id) == float('-inf') or self.heuristic(env,self.agent_id) == float('inf')):
            return (self.heuristic(env,self.agent_id), operators[0])
        
        if (depth == max_depth):
            return (self.heuristic(env,self.agent_id), None)
        
        
        children = [env.clone() for _ in operators]
        sorted_child_op_zip = zip(children, operators)

        options = []
        for child, op in sorted_child_op_zip:
            child.apply_operator(current_id, op)
            (child_val, act) = self.exp_value(child, depth+1, not current_id, max_depth, time_limit, time_started)
            options.append((child_val, op))

        return max(options, key=lambda x: x[0])
    

    def exp_value(self, env: WarehouseEnv, depth, current_id, max_depth,time_limit, time_started):

        if (time.time()-time_started>=time_limit):
            raise TimeoutException
        
        operators = env.get_legal_operators(current_id)
        operator_probability_arr = self.operator_probability(operators)

        if (self.heuristic(env,self.agent_id) == float('-inf') or self.heuristic(env,self.agent_id) == float('inf')):
            return (self.heuristic(env,self.agent_id), operators[0])
        
        if (depth == max_depth):
            return (self.heuristic(env,self.agent_id), None)
        
        children = [env.clone() for _ in operators]

        v = 0
        for child, (op, prob) in zip(children, operator_probability_arr):
            child.apply_operator(current_id, op)
            (child_val, act) = self.max_value(child, depth+1, not current_id, max_depth, time_limit, time_started)
            v += prob*child_val

        return (v, None)

        

    def helper_aux(self, max_depth,time_limit, time_started):
        (best_value, best_op22)=self.max_value(self.env, 0, self.agent_id, max_depth,time_limit, time_started)
        self.best_op= best_op22


    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        self.agent_id = agent_id
        time_started = time.time()
        time_limit=0.9*time_limit
        self.env = env
        max_depth=1
        if env.num_steps/2 < 2: # last move should be greedy
            return self.greedy()

        operators = env.get_legal_operators(self.agent_id)
        (best_value, self.best_op) = (None, operators[0])

        try:
            while True:
                if (env.num_steps/2 < max_depth):
                    break
                self.helper_aux(max_depth, time_limit, time_started)
                max_depth+=1
                if time_limit<=time.time()-time_started:
                    break

        except TimeoutException as e:
            pass        
        # not sure why is it None
        if self.best_op==None: # sholdnt enter here
            return self.greedy()

        # print(f"Reached depth {max_depth}")
        return self.best_op


# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = ["move east", "move east", "pick_up", "pick_up", "pick_up", "move east", "move east",
                           "move east", "move east", "move east", "pick_up", "move east",
                           "pick_up","move east", "pick_up", "pick_up", "pick_up", "pick_up" , "move east"]

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