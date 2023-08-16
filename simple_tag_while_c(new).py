
# v1.0.2

'''
simple tag -> tag while c

### Arguments

``` python
tag_while_c.env(max_cycles=25, envQuality='H')
```

`max_cycles`:  number of frames (a step for each agent) until game terminates

`envQuality`: Whether agent action spaces are discrete(default) or continuous

'''

'''
to be done
num_collection
绘制边界框

constraint:
1. 一个收集器只能有一个父控制器
2. 两个controller两层不同agent取平均是不可取的（？）

note:
执行顺序：observation -> reward(OK)
收集到资源点的reward: collector: 1, controller: 0.5

edit:
8.16.2023
1. 修复了collector的reward计算错误，更新num_collection的时机应该在collector计算reward之后


older version: v1.0.0 v1.0.1
older edition in older files
--- new environment version ---
old: simple_tag_while_c(v1.0.1) before 8.16 (未保存,可还原)
new: simple_tag_while_c(v1.0.2) after 8.16
--- new environment version ---
old: simple_tag_while_c(v1.0.2) before 8.16 9AM(UTC+06)
new: simple_tag_while_c(v1.0.3) after 8.16 9AM(UTC+06)
'''

import numpy as np
from gymnasium.utils import EzPickle
import random

from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn
# from my_envs.utils import PQ
from pettingzoo.mpe.simple_tag.myutils import PQ
from typing import Optional, Union, List, Any, Dict, Tuple

'''环境参量'''

COMMUNICATE_THRESH = 0.5 # 通讯半径
DETECT_THRESH = 0.5 # 探测半径
ACCELERATE_THRESH = 0.2 # 加速半径

SIZES = [0.075,0.05,0.02]
J_SIZE = 0.01
ACCEL = [3.0,4.0,5.0] 
''' before 8.15 0816PM [3.0,4.0,3.0]'''
# ACCEL = [0.1,0.1,0.1]
MAX_SPEED = [1.0,1.3,1.5]
''' before 8.15 0816PM [1.0,1.3,1.0]'''
# MAX_SPEED = [0.01,0.01,0.01]
BOUNDARY = [[np.array([-1,1]),np.array([1,1])],
            [np.array([1,1]),np.array([1,-1])],
            [np.array([1,-1]),np.array([-1,-1])],
            [np.array([-1,-1]),np.array([-1,1])]]
AGENT_COLORS = [
    np.array([0.85, 0.35, 0.35]), # red: adversary
    np.array([0.85, 0.85, 0.35]), # yellow: controller
    np.array([0.35, 0.85, 0.35]), # green: collector
]
J_COLOR = np.array([0.25, 0.25, 0.25]) # gray: resource point
LINE_COLOR = np.array([0.25, 0.25, 0.85]) # blue: line color
DISTANCE_SHAPING_ADV = False
DISTANCE_SHAPING_CTL = False
COMMUTE_SHAPING_CONTROLLER = True
COMMUTE_SHAPING_COLLECTOR = True

'''应该改为可选参数'''
N_DETECT = 3 # number of resources detected by collector
STRICT_MODE = False # 严格模式，丢失通信后，collector消失
# DRAW_DEAD_POINT = False # 绘制已经被收集的J，为红色
DRAW_DEAD_POINT = True

class raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        num_adversaries=1,
        num_controller=1,
        num_collector_each=2,
        num_J=10,
        max_cycles=25,
        continuous_actions=False,
        render_mode=None,
        updating_J=True
    ):
        # use EzPickle to easily support pickle/unpickle
        EzPickle.__init__(
            self,
            # num_good=num_good,
            num_adversaries=num_adversaries,
            num_controller=num_controller,
            num_collector_each=num_collector_each,
            num_J=num_J,
            # num_obstacles=num_obstacles,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
            updating_J=updating_J
        )
        scenario = Scenario_new()
        
        world = scenario.make_world(num_adversaries, num_controller, num_collector_each, num_J, updating_J)
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
        )
        self.metadata["name"] = "simple_tag_while_c"

env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario_new(BaseScenario):
    ''' update_J: 在collector附近随机生成 '''
    def make_world(self, num_adversaries=1, num_controller=1, num_collector_each=2, num_J=6, updating_J=True, envQuality='L'):
        world = World()
        
        # set any world properties first
        num_good_agents = num_controller
        num_adversaries = num_adversaries
        num_collectors = num_collector_each*num_controller
        num_agents = num_adversaries + num_good_agents + num_collectors
        num_landmarks = num_J
        world.dim_c = 2
        world.num_controller = num_controller
        world.num_adversaries = num_adversaries
        world.num_collector_each = num_collector_each
        world.num_collectors = num_collectors
        world.num_J = num_J  
        world.quality = envQuality 
        world.N_detect = N_DETECT # number of resources detected by collector default: 3
        world.strict_mode = STRICT_MODE # 严格模式，丢失通信后，collector消失
        world.draw_line = []
        world.line_color = LINE_COLOR # blue: line color
        world.updating_J = updating_J
        world.draw_dead_point = DRAW_DEAD_POINT
        if world.draw_dead_point:
            world.dead_point = []
        
        # add agents
        # [adversary_0, controller_0, collector_00, collector_01]
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.adversary = True if i < num_adversaries else False
            agent.controller = True if (i>=num_adversaries and i<num_adversaries+num_controller) else False
            if agent.adversary:
                base_name = 'adversary'
                base_index = i
                agent.kind = 0
                agent.accelerate_threshold = ACCELERATE_THRESH # 加速半径
                agent.base_index = base_index
            elif agent.controller:
                base_name = 'controller'
                base_index = i - num_adversaries
                agent.kind = 1
                agent.comm_threshold = COMMUNICATE_THRESH # 通信半径（范围）
                agent.base_index = base_index
                '''COMMUNICATE_THRESH 保存于每个controller的属性'''
            else:
                base_name = 'collector'
                father_index = (i - num_adversaries - num_controller)//num_collector_each
                son_index = (i - num_adversaries - num_controller)%num_collector_each
                base_index = '{}_{}'.format(father_index, son_index)
                agent.kind = 2
                agent.father_index = father_index
                agent.son_index = son_index
                agent.detect_threshold = DETECT_THRESH # 探测半径
                agent.lost = False
                agent.num_collection = 0
            agent.name = f"{base_name}_{base_index}"
            agent.collide = True
            agent.silent = True
            agent.size = SIZES[agent.kind] # 0.075 if agent.adversary else 0.05
            agent.accel = ACCEL[agent.kind] # 3.0 if agent.adversary else 4.0
            agent.max_speed = MAX_SPEED[agent.kind]# 1.0 if agent.adversary else 1.3
        # end for
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)] # J
        world.resoueces = world.landmarks # 不同的名字，同样的内容
        
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "J_%d" % i
            landmark.collide = False # 不应该设为可碰撞
            landmark.movable = False
            landmark.size = J_SIZE # 0.01
            landmark.boundary = False
            landmark.color = J_COLOR
            # landmark.state.p_pos = self.random_J_pos(i, world)
            # 不需要在此处...
        # make initial conditions
        world.total_J = num_landmarks

        # 增加边界
        world.boundaries = [Landmark() for i in range(4)]
        for i, landmark in enumerate(world.boundaries):
            landmark.name = "bound_%d" % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.01
            landmark.boundary = True
            landmark.color = J_COLOR # 与资源点相同
            # 起止位置 
            landmark.pos_start = BOUNDARY[i][0]
            landmark.pos_end = BOUNDARY[i][1]

        return world
        
    def random_J_pos(self, idx, world):
        # 每个collector的探测范围内至少有两个J
        if idx < world.num_collectors * 2:
            collector_id = idx % world.num_collectors
        else:
            collector_id = np.random.randint(0, world.num_collectors)
        collector = world.agents[world.num_adversaries + world.num_controller + collector_id]
        
        legal = False
        while not legal:
            J_pos = np.random.rand(2) * 2 - 1
            legal = True if np.sum(np.square(J_pos)) < 1 and in_bound(J_pos * collector.detect_threshold + collector.state.p_pos) else False 
        
        # 保证在探测范围内
        return J_pos * collector.detect_threshold + collector.state.p_pos
        
    def reset_world(self, world, np_random):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = AGENT_COLORS[agent.kind]
        # random properties for landmarks(resource points)
        for i, landmark in enumerate(world.landmarks):
            landmark.color = J_COLOR
            
        for agent in world.agents:
            '''追逐者和控制器随机初始化位置，收集器初始位置与父控制器相同'''
            if agent.kind in [0, 1]:
                agent.state.p_pos = np_random.uniform(-0.9, +0.9, world.dim_p)
            else:
                agent.state.p_pos = world.agents[world.num_adversaries + agent.father_index].state.p_pos + (np.random.rand(world.dim_p) - 0.5) * 0.1
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
           
        '''旧版的初始化J的方法''' 
        # for i, landmark in enumerate(world.landmarks):
        #     legal = False
        #     ''' 保证初始化的landmark在collector的探测范围之内'''
        #     while not legal:
        #         landmark.state.p_pos = np_random.uniform(-0.95, +0.95, world.dim_p)
        #         legal = True if in_detect(landmark, world.agents) else False
        #     # end while
        #     landmark.state.p_vel = np.zeros(world.dim_p)
        
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = self.random_J_pos(i, world)
            landmark.state.p_vel = np.zeros(world.dim_p)

        # 清空dead point
        world.dead_point = []
            
    '''to be done'''
    def benchmark_data(self, agent, world):
    # returns data for benchmarking purposes
        if agent.adversary:
            num_collision = 0
            for a in self.controllers(world):
                if self.is_collision(a, agent):
                    num_collision += 1
            return num_collision
        # 评判一个agent的好坏？在某一时刻的好坏？
        elif agent.kind == 2:
            return agent.num_collection
        else: # controller
            return 0 # to be done
        
    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False
    
    # return all agents that are not adversaries
    def controllers(self, world):
        return [agent for agent in world.agents if agent.kind == 1]
    
    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]
    
    def reward(self, agent, world):
        # print("reward called") # to debug
        # Agents are rewarded based on minimum agent distance to each landmark
        if agent.kind in [0,1]:
            main_reward = (
                self.adversary_reward(agent, world)
                if agent.adversary
                else self.controller_reward(agent, world)
            )
            return main_reward
        # Collectors are rewarded ...
        else:
            return self.collector_reward(agent, world)
        
    def controller_reward(self, agent, world):
        '''collision reward, bound reward, communication reward, [distance reward], [commute_D reward]'''
    # Agents are negatively rewarded if caught by adversaries
        rew = 0
        shape = DISTANCE_SHAPING_CTL # default: shape = False
        adversaries = self.adversaries(world)
        collectors = self.son_collectors(agent, world)
        if (shape):  
        # reward can optionally be shaped (increased reward for increased distance from adversary)
            for adv in adversaries:
                rew += 0.1 * np.sqrt(
                    np.sum(np.square(agent.state.p_pos - adv.state.p_pos))
                )
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= 10
                    
        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)
        
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)
            
        commute_shape = COMMUTE_SHAPING_CONTROLLER # defult: False
        if commute_shape:
        # reward can optionally be shaped (decreased reward for increased out_of_range_distance from collectors)
            '''for adv in collectors'''
            for clt in collectors: # adv stands for each collector here
                dist = float(np.sqrt(np.sum(np.square(agent.state.p_pos - clt.state.p_pos))))
                rew -= 0.1 * max(0, dist - agent.comm_threshold)
            
        # agents are penalized 丢失子收集器的通信
        ''' 非strict_mode: controller的发送半径为inf,
            strict_mode: 丢失通信后，collector消失
        
        '''
        if agent.num_lost > 0:
            if world.strict_mode:
                rew -= 5*agent.num_lost
            else:
                rew -= 1*agent.num_lost

        # 子收集器收集到资源点的reward
        for clt in collectors:
            rew += 5 * clt.num_collection # 应该在更新controller之前先更新collector
            # OK
            if clt.num_collection>0: print('collection reward to father controller: ', 5*clt.num_collection) # to inform or debug
            # 更新collector的reward之后再重置
            # clt.num_collection = 0
        # end for

        return rew

    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        shape = DISTANCE_SHAPING_ADV # defualt:False
        agents = self.controllers(world)
        adversaries = self.adversaries(world)
        if (shape):  
            # reward can optionally be shaped (decreased reward for increased distance from agents)
            for adv in adversaries:
                rew -= 0.1 * min(
                    np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos)))
                    for a in agents
                )
        if agent.collide:
            for ag in agents:
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        rew += 10
        return rew
        
    '''
    收集者的reward
    agent.lost, agent.num_collection
    即判断丢失通信，判断收集到资源点都位于step函数中
    
    '''
    def collector_reward(self, agent, world):
        rew = 0
        shape = COMMUTE_SHAPING_COLLECTOR # defult:True
        '''communication reward最近的父中继器'''
        if (shape):
            # reward can optionally be shaped (decreased reward for increased distance from its father controller)
            # rew -= 0.1*np.min([np.sqrt(np.sum(np.square(agent.state.p_pos - father.state.p_pos)))
                                # for father in self.father_controller(agent, world)])
            father = self.father_controller(agent, world)[0]
            rew -= 0.1*np.sqrt(np.sum(np.square(agent.state.p_pos - father.state.p_pos)))
            
        if agent.lost:
            if world.strict_mode:
                rew -= 10
            else:
                rew -= 1

        '''collecton reward'''
        rew += 5 * agent.num_collection # num_collection is 0 or 1
        if agent.num_collection > 0: # to debug
            print('collector reward for collection:', 5*agent.num_collection) # OK
            # 更新完reward之后再重置num_collection
            agent.num_collection = 0

        return rew
        
    def generate_landmark(self, collector, world)->Landmark: # generate J
        assert world.updating_J == True, 'Code Criterion Programming Error'
        legal = False
        while not legal:
            # J_pos = random.randrange(-1,1,world.dim_p)
            J_pos = np.random.rand(2) * 2 - 1
            legal = True if np.sum(np.square(J_pos)) < 1 and in_bound(J_pos*collector.detect_threshold + collector.state.p_pos) else False
        resource_point = Landmark()
        resource_point.state.p_pos = J_pos*collector.detect_threshold + collector.state.p_pos # 保证在探测范围内
        resource_point.color = J_COLOR
        resource_point.movable = False
        resource_point.collide = False
        resource_point.size = 0.01
        return resource_point
        
    '''主函数:
    
    
        目前只能对应于一个收集器只有一个父控制器的情况
    '''
    def observation(self, agent, world):
        # print("obervation called") # to debug

        if agent.kind == 0:
            return self.adversary_observation(agent, world)
        elif agent.kind == 1:
            # 重置collector和controller的位置        
            world.draw_line = []
            agent.num_lost = 0
            for son in self.son_collectors(agent, world):
                if not in_commute(son, agent):
                    son.lost = True
                    agent.num_lost += 1
                    # print('Error: son collector lost') # to inform
                else:
                    son.lost = False
                    world.draw_line.append([son.state.p_pos, agent.state.p_pos])
            return self.controller_observation(agent, world)
        else:
            return self.collector_observation(agent, world)
        
    def adversary_observation(self, agent, world):
        # ctl_pos = []
        # for ctl in self.controllers(world):
            # ctl_pos.append(ctl.state.p_pos - agent.state.p_pos)
        other_pos = []
        other_vel = []
        
        for other in world.agents:
            if other is agent:
                continue
            if other.kind == 2: # collector
                continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            if not other.adversary:
                other_vel.append(other.state.p_vel)
                
        return np.concatenate(
            [agent.state.p_vel]
            + [agent.state.p_pos]
            + other_pos
            + other_vel
        )      
    
    def controller_observation(self, agent, world):
        other_pos = []
        other_vel = []
        
        for other in world.agents:
            if other is agent:
                continue
            elif other.kind == 0:
                other_pos.append(other.state.p_pos - agent.state.p_pos)
                
            elif other.kind == 1: # controller
                other_pos.append(other.state.p_pos - agent.state.p_pos)
                other_vel.append(other.state.p_vel)
                
            else: # collector
                if world.quality in ['L','H','VH']:
                    other_pos.append(other.state.p_pos - agent.state.p_pos)
                if world.quality in ['VH']:
                    other_vel.append(other.state.p_vel)
        # end for
        
        num_sons = len(self.son_collectors(agent, world))
        if num_sons < world.num_collector_each: # 子收集器已经丢失，保证每个controller的观测值维度相同
            for i in range(world.num_collector_each - num_sons):
                other_pos.append(np.zeros(world.dim_p))
                other_vel.append(np.zeros(world.dim_p))
                
        return np.concatenate(
            [agent.state.p_vel]
            + [agent.state.p_pos]
            + other_pos
            + other_vel
        )
        
    def collector_observation(self, agent, world):
        other_pos = []
        other_vel = []
        # get positions of all entities in this agent's reference frame
        
        # to be done
        if agent.lost and world.strict_mode:
            # del agent 
            # 找到对应的原agent，删除
            idx = world.agents.index(agent)
            del world.agents[idx]
            return 0
        
        '''无论是否lost，free mode下都可以观测到father controller'''
        for father in self.father_controller(agent, world): # should be only one father
            other_pos.append(father.state.p_pos - agent.state.p_pos) # 观测要传输相对位置
            if world.quality in ['H','VH']:
                other_vel.append(father.state.p_vel)
            for sibling in self.son_collectors(father, world):
                if sibling is agent:
                    continue
                if world.quality in ['L','H','VH']:
                    other_pos.append(sibling.state.p_pos - agent.state.p_pos)
                if world.quality in ['VH']:
                    other_vel.append(sibling.state.p_vel)
        
        # get positions of {N_detect} resources in this agent's detect range
        priority_queue = PQ(world.N_detect, True) # reverse=True, 按照dist由小到大排列
        
        '''重要: 判断收集资源点'''
        collected = []
        # agent.num_collection = 0 # reward更新时再重置
        '''edited'''
        for i, resource in enumerate(world.landmarks):
            if in_detect(resource, agent):
                '''基本上不会一次收集两个资源点'''
                '''两个collector不能同时收集一个资源点，判断有先后'''
                if dist(resource, agent) < agent.size:
                    agent.num_collection = agent.num_collection + 1
                    collected.append(resource)
                    print('collection success:')

                else:
                    priority_queue.push(resource.state.p_pos - agent.state.p_pos, dist(resource, agent))
        if agent.num_collection>0: print('num_collection:', agent.num_collection)
        # end for

        '''OK'''
        if len(collected) > 0: # 处理收集到的资源点，并生成新的资源点
            if len(collected) > 1:
                # raise ValueError('一次收集了多于一个资源点') # to be done
                print('一次收集了多于一个资源点') # done
            for J in collected:
                if world.draw_dead_point:
                    world.dead_point.append(J.state.p_pos) # 目前只保存了位置
                # end if
                idx = world.landmarks.index(J)
                del world.landmarks[idx] # 删去该资源点
            # print(len(world.landmarks))
            # 生成新的资源点
            if world.updating_J:
                for i in range(len(collected)):
                    new_landmark = self.generate_landmark(agent, world)
                    world.landmarks.append(new_landmark)
                    priority_queue.push(new_landmark.state.p_pos - agent.state.p_pos, dist(new_landmark, agent))
                # end for
                # print(len(world.landmarks))
            
        return np.concatenate(
            [agent.state.p_vel]
            + [agent.state.p_pos]
            + other_pos
            + other_vel
            + priority_queue.out()
        )       
        
    """self defined"""
    
    def son_collectors(self, agent, world):
        return [a for a in world.agents if a.kind == 2 and a.father_index == agent.base_index]
    
    def father_controller(self, agent, world):
        return [a for a in world.agents if a.kind == 1 and a.base_index == agent.father_index]
        
def dist(A: Union[Agent, Landmark], B: Union[Agent, Landmark]):
    return np.sqrt(np.sum(np.square(A.state.p_pos - B.state.p_pos)))

def in_detect(J, collector):
    return True if dist(J, collector) < collector.detect_threshold else False

def in_commute(collector, father):
    return True if dist(collector, father) < father.comm_threshold else False # comm_threshold = 0.5

def in_bound(entity: Union[Agent, Landmark, np.ndarray]):
    if hasattr(entity, 'state'): # Entity(Agent or Landmark)
        return True if np.all(np.abs(entity.state.p_pos) <= 1) else False
    else: # np.adarray
        return True if np.all(np.abs(entity) <= 1) else False
        
def in_range(entity: Union[Agent, Landmark, np.ndarray], threshold: float):
    pass

def in_communication(entity: Union[Agent, Landmark, np.ndarray], threshold: float):
    pass


if __name__ == '__main__':
    print('env init')
    print('done')
    
                    
        
            
            
            
            
            
            
                
                    
                
            
        
                
            
            
        
        
            
        
        
        
