# using MADDPG to solve a novel multilayer MPE task



### 1 introduction to the environment

The new environment named "simple tag while collecting" is an MPE environment based on the simple tag environment. The difference is that the prey, which in this new environment is called *controller* controls two particles called *collector* to collect food. 

So simply there are four agents in this environment. One *controller*, one *adversary* and two *collector*. The *controller* escapes from the *adversary* while communicating with the *collectors* to collect food. The *adversary* tries to catch the *controller*. The *collector* try to collect food and keep the distance with the *controller* in a certain communication range.

The environment has a multi-layer design. The *controller* and the *adversary* are in the first layer(called *tag* layer). The *collectors* are in the second layer(called *collection* layer). The *collectors* in collection layer can communicate with the *controller* in tag layer, to know the position and velocity of the *controller*. But they cannot communicate with the *adversary*. Which means, the *controller* knows all other three objects, the *adversary* only knows the *controller*, and the *collectors* only know their corresponding father *controller*.

This environment can be applied to a physical scenario where some intelligent agents (corresponding to *collector*) have to communicate through an intermediate controller (*controller*) while collecting resources (food) , within certain range limitation for the communication. The intermediate controller is under a chasing attack of enemy (*adversary*) , so it has to lead the agents together together to avoid the Adversary's attacks. The repeater needs to weigh the two actions of collecting and escaping, and the collector also needs to balance the movement towards the resource and following the intermediate controller. This is an application of this two-layer multi-agent environment.

the multilayer environment design:

![multilayer structure](multilayer structure.png)

### 2 details of the environment

| Import | from petting.mpe.simple_tag import simple_tag_while_c # our environment |
| --- | --- |
| Actions | Discrete |
| Parallel API | Yes |
| Agents | ``agents= [adversary_0, controller_0, collector_00, collector_01]`` |
| Agents_num | 4 |
| State Shape | (56,) |
| State Values | (-inf,inf) |

|                    | adversary   | controller            | collector             |
| ------------------ | ----------- | --------------------- | --------------------- |
| Action Shape       | (5,)        | (5,)                  | (5,)                  |
| Action Values      | Discrete(5) | Discrete(5)           | Discrete(5)           |
| Observation Shape  | (8,)        | (10,)                 | (20,)                 |
| Observation Values | (-inf,inf)  | (-inf,inf) and (-1,1) | (-inf,inf) and (-1,1) |

*note: observation of food is in (-1, 1), when other observations is in (-inf, inf)*



1. **Agents**: There are three kind of agents in this environment. These are the adversary, the controller, and the collector.
   - Aadversary: located in the first layer of the structure, chasing the controller
   
   - Controller: located in the first layer of the network structure, to avoid the adversary, and communicate with the collector
   
   - Collector: located in the second layer of the network structure, collects resources and ensures that it is within a certain distance from its corresponding controller.
   
2. **Observation space**:
   - Observation space of the adversary: its own and the controller's position, speed, each for a two-dimensional vector, taking the value of (-inf,inf)
   - Observation space of the controller: position and velocity of itself; position of the controller; positions of the `N_clt = 2` corresponding collectors
   - Observation space of the collector: own position, velocity; position of the nearest `N_detect = 3` food around it; position of the corresponding controllers

1. **Action space**:

   - All three kinds of agents use a discrete action space, and there are only five actions in the discrete action space, which are ``[no_action, move_left, move_right, move_down, move_up]``. The action is an integer taking the values ``[0, 1, 2, 3, 4]``.

2. **Reward**:
   - collision-reward: reward 20p for adversary and -20p for controller
   - communication-reward: punish both the collector -1p and controller -1p if the collector is out of the range of controller
   - collect-reward: reward 5p to collector and 2p to controller for getting each food point ``J``.
   - bound-reward: punish the controller if it is too far from the centre point(same as simple_tag)

### 3 introduction to the algorithm

In this environment, we use *MADDPG* algorithm to solve this multiagent problem. MADDPG agorithm proposes the paradigm of centralised training with decentralized execution (CTDE). Each agent has an independent Actor network, but shares a centralised Critic network. During training, the Critic network provides guidance for the action strategy of each agent, but during execution, each agent still makes decisions independently according to its own strategy, achieving decentralised execution. CTDE speed up the action taking process while effectively using the global information during training to achieve better and more stable training results. This paradigm fits the multilayer structure because agents have to take action based on their own observation in evaluating, but need to know global characteristics in training.

*MADDPG* algorithm is based on Actor-Critic model. There are `N = 4` Actors and one Critic. Both Actor and Critic is a multilayer perceptron(MLP) network. A 4 layers MLP has suitable complexity for Actor and Critic to learn about the features of the environment. Because the input of the network is location and velocity, not an image. 

*MADDPG* algorithm also use a storage buffer to storage experiences. It delays the update of the target Actor network and target Critic network to improve the stability of training.

### 4 code modifications

Our work is based on [pettingzoo environment](https://pettingzoo.farama.org/environments/mpe/simple_tag/) . The *MADDPG* algorithm implementation refers [maddpg-pettingzoo-pytorch](https://github.com/Git-123-Hub/maddpg-pettingzoo-pytorch).

**in simple_tag_while_c.py**

```python
# imports
from pettingzoo.mpe.simple_tag.myutils import PQ
from typing import Optional, Union, List, Any, Dict, Tuple

```

**add global environment arguments:**

```python
'''global arguments'''
COMMUNICATE_THRESH = 0.5 # communication radius
DETECT_THRESH = 0.8 # detection radius defualt: 0.5
ACCELERATE_THRESH = 0.2 # acceleration radius(not in use)

SIZES = [0.075,0.05,0.03]
J_SIZE = 0.02
ACCEL = [3.0,4.0,10.0]
'''[3.0,4.0,3.0]'''
MAX_SPEED = [1.0,1.3,2.0]
'''[1.0,1.3,1.0]'''
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
COMMUTE_SHAPING_CONTROLLER = False
COMMUTE_SHAPING_COLLECTOR = False

N_DETECT = 6 # number of resources detected by collector default: 3
STRICT_MODE = False # in strict mode, a collector is down if it is out of communication
# DRAW_DEAD_POINT = False # draw the collected J, in red.
DRAW_DEAD_POINT = True
```

**add 3 kinds of agent and their correspondence:**

```python
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

    world.draw_line = []
    world.line_color = LINE_COLOR # blue, to draw the communication line
    world.updating_J = updating_J
    world.draw_dead_point = DRAW_DEAD_POINT
    if world.draw_dead_point:
        world.dead_point = []

    # add agents
    # [adversary_0, controller_0, collector_0_0, collector_0_1]
    world.agents = [Agent() for i in range(num_agents)]
    for i, agent in enumerate(world.agents):
        agent.adversary = True if i < num_adversaries else False
        agent.controller = True if (i>=num_adversaries and i<num_adversaries+num_controller) else False
        if agent.adversary:
            base_name = 'adversary'
            base_index = i
            agent.kind = 0
            agent.accelerate_threshold = ACCELERATE_THRESH # not implemented
            agent.base_index = base_index
        elif agent.controller:
            base_name = 'controller'
            base_index = i - num_adversaries
            agent.kind = 1
            agent.comm_threshold = COMMUNICATE_THRESH # communication radius
            agent.base_index = base_index
        else:
            base_name = 'collector'
            father_index = (i - num_adversaries - num_controller)//num_collector_each
            son_index = (i - num_adversaries - num_controller)%num_collector_each
            base_index = '{}_{}'.format(father_index, son_index)
            agent.kind = 2
            agent.father_index = father_index
            agent.son_index = son_index
            agent.detect_threshold = DETECT_THRESH # detection redius
            agent.lost = False
            agent.num_collection = 0
        agent.name = f"{base_name}_{base_index}"
        agent.collide = True
        agent.silent = True
        agent.size = SIZES[agent.kind] # 0.075 if agent.adversary else 0.05
        agent.accel = ACCEL[agent.kind] # 3.0 if agent.adversary else 4.0
        agent.max_speed = MAX_SPEED[agent.kind]# 1.0 if agent.adversary else 1.3
```

**add food(J):**

```python
	world.landmarks = [Landmark() for i in range(num_landmarks)] # J
    world.resources = world.landmarks # they are the same

    for i, landmark in enumerate(world.landmarks):
        landmark.name = "J_%d" % i
        landmark.collide = False
        landmark.movable = False
        landmark.size = J_SIZE # 0.01
        landmark.boundary = False
        landmark.color = J_COLOR
        # landmark.state.p_pos = self.random_J_pos(i, world) # in reset function

    world.total_J = num_landmarks

'''the position of J is randomly selected in the detection range of collectors'''
def random_J_pos(self, idx, world):
    # randomly select in detection of which collector, at least 2 food in each collector's detection
    if idx < world.num_collectors * 2:
        collector_id = idx % world.num_collectors
    else:
        collector_id = np.random.randint(0, world.num_collectors)
    collector = world.agents[world.num_adversaries + world.num_controller + collector_id]

    # to make sure the food is init in detection range
    legal = False
    while not legal:
        J_pos = np.random.rand(2) * 2 - 1
        legal = True if np.sum(np.square(J_pos)) < 1 and in_bound(J_pos * collector.detect_threshold + collector.state.p_pos) else False 

    return J_pos * collector.detect_threshold + collector.state.p_pos

def reset_world(self, world, np_random):
    for i, landmark in enumerate(world.landmarks):
        landmark.color = J_COLOR
    ...
    
    for i, landmark in enumerate(world.landmarks):
        landmark.state.p_pos = self.random_J_pos(i, world)
        landmark.state.p_vel = np.zeros(world.dim_p)
    ...
    
'''when a food is collected, generate a new one in detection of that collector'''
def generate_landmark(self, collector, world)->Landmark: # generate J
    # randomly select position
    legal = False
    while not legal:
        J_pos = np.random.rand(2) * 2 - 1
        legal = True if np.sum(np.square(J_pos)) < 1 and in_bound(J_pos*collector.detect_threshold + collector.state.p_pos) else False
    resource_point = Landmark()
    resource_point.state.p_pos = J_pos*collector.detect_threshold + collector.state.p_pos
    resource_point.color = J_COLOR
    resource_point.movable = False
    resource_point.collide = False
    resource_point.size = J_SIZE
    return resource_point
```

**change reward function:**

```python
def reward(self, agent, world):
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

# CTL for controller
def controller_reward(self, agent, world):
    '''collision reward, bound reward, communication reward, [distance reward], [commute_distance reward]'''
# Agents are negatively rewarded if caught by adversaries
    rew = 0
    shape = DISTANCE_SHAPING_CTL # default: shape = False
    adversaries = self.adversaries(world)
    collectors = self.son_collectors(agent, world)
    
    collision reward and distance reward[optional]
    if (shape):  
    	...
    if agent.collide:
        ...
    
    # agents are penalized for exiting the screen, so that they can be caught by the adversaries
    def bound(x):
        ...

    for p in range(world.dim_p):
        x = abs(agent.state.p_pos[p])
        rew -= bound(x)

    commute_shape = COMMUTE_SHAPING_CONTROLLER # defult: False
    if commute_shape:
    # reward can optionally be shaped (decreased reward for increased out_of_range_distance from collectors)
        '''for adv in collectors'''
        for clt in collectors: # adv stands for each collector here
            dist = float(np.sqrt(np.sum(np.square(agent.state.p_pos - clt.state.p_pos))))
            rew -= 1 * max(0, dist - agent.comm_threshold)

    # agents are penalized when losting the communication
    if agent.num_lost > 0:
        if world.strict_mode:
            rew -= 10*agent.num_lost
        else:
            rew -= 1*agent.num_lost

    # collection reward
    for clt in collectors:
        rew += 2 * clt.num_collection
        # if clt.num_collection>0: print('collection reward to father controller: ', 1*clt.num_collection) # to inform or debug
        # reset 0 after updating the collector rewards
        # clt.num_collection = 0
    # end for

    return rew

def adversary_reward(self, agent, world):
    ...
    (no changes)

'''
collector's reward:
communication reward, collection reward, [commute_distance reward]
'''
def collector_reward(self, agent, world):
    rew = 0
    shape = COMMUTE_SHAPING_COLLECTOR # defult:False
    '''communication reward w.r.t. its father controller'''
    if (shape):
        # reward can optionally be shaped (decreased reward for increased distance(larger than the communication radius) from its father controller)
        
        father = self.father_controller(agent, world)[0]
        rew -= 1 * max(np.sqrt(np.sum(np.square(agent.state.p_pos - father.state.p_pos))) - father.comm_threshold, 0)

    if agent.lost:
        if world.strict_mode:
            rew -= 5
        else:
            rew -= 1

    '''collecton reward'''
    rew += 5 * agent.num_collection # num_collection is 0 or 1
    if agent.num_collection > 0: # to debug
        # print('collector reward for collection:', 2*agent.num_collection) # OK
        # reset num_collection after updating rewards
        agent.num_collection = 0

    return rew

```

**change observation function:**

```python
def observation(self, agent, world):
    # print("obervation called") # to debug

    if agent.kind == 0:
        return self.adversary_observation(agent, world)
    elif agent.kind == 1:
        # reset the position        
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
        ...
        (no changes)
    
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
        if num_sons < world.num_collector_each: # son collectors lost，make sure controller has the same shape of observation each time.
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
            idx = world.agents.index(agent)
            del world.agents[idx]
            return 0
        
        '''set to be able to observe father controller even if out of communication range, for better training. Add negative reward instead'''
        for father in self.father_controller(agent, world): # should be only one father
            other_pos.append(father.state.p_pos - agent.state.p_pos)
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
        priority_queue = PQ(world.N_detect, True) # reverse=True, in dist order from smallest to largest
        
        collected = []
        already_collect = False
        for i, resource in enumerate(world.landmarks):
            if in_detect(resource, agent):
                '''a collector can only collect one food at a time'''
                if dist(resource, agent) < agent.size + resource.size and not already_collect:
                    agent.num_collection = agent.num_collection + 1 # 0 or 1
                    collected.append(resource)
                    already_collect = True
                    # print('collection success') # to inform

                else:
                    priority_queue.push(resource.state.p_pos - agent.state.p_pos, dist(resource, agent))
        # if agent.num_collection>0: print('num_collection:', agent.num_collection) # to debug
        # end for

        '''OK'''
        if len(collected) > 0: # deal with collected food, and generate new one.
            for J in collected:
                if world.draw_dead_point:
                    world.dead_point.append(J.state.p_pos) # to render collected food points
                # end if
                idx = world.landmarks.index(J)
                del world.landmarks[idx]
            # generate new food 
            if world.updating_J:
                for i in range(len(collected)):
                    new_landmark = self.generate_landmark(agent, world)
                    world.landmarks.append(new_landmark)
                    priority_queue.push(new_landmark.state.p_pos - agent.state.p_pos, dist(new_landmark, agent))
            # end for
            
        return np.concatenate(
            [agent.state.p_vel]
            + [agent.state.p_pos]
            + other_pos
            + other_vel
            + priority_queue.out()
        )       
```

**utility functions:**

in simple_tag_while_c.py and myutils.py

```python
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

# priority queue
class PQ:
    def __init__(self, max_size, reverse=False):
        self.queue = []
        self.size = 0
        self.max_size = max_size
        self.reverse = reverse # in decent order

    # item: np.ndarray, shape=(2,)
    def push(self, item: np.ndarray, priority):
        if self.reverse:
            priority = -priority
        
        insert_index = self.size
        for i in range(self.size):
            if priority > self.queue[i][1]:
                insert_index = i
                break
        self.queue.insert(insert_index, (item, priority))
        self.size += 1
        if self.size > self.max_size:
            self.pop()
        
    def pop(self):
        self.size -= 1
        temp = self.queue[-1][0]
        del self.queue[-1]
        return temp # return value，without priority
    
    def top(self):
        return self.queue[0][0]
    
    # fill with 0 if not enough
    def out(self):
        return  [x[0] for x in self.queue] + [np.zeros(2)] * (self.max_size - self.size)
```

**change render objects:**

in simple_env.py

```python
"""Backward compatible with simple_tag"""
def draw(self):
    ...
    for e, entity in enumerate(self.world.entities):
        ...

        '''added by Xiao Xunman'''
        if hasattr(entity, "accelerate_threshold"):
            pygame.draw.circle(
                self.screen, entity.color * 200, (x, y), entity.accelerate_threshold * 350, 1
            )  # acc enable range
        if hasattr(entity, "comm_threshold"):
            pygame.draw.circle(
                self.screen, entity.color * 200, (x, y), entity.comm_threshold * 350, 1
            )  # communication range
        if hasattr(entity, "detect_threshold"):
            pygame.draw.circle(
                self.screen, entity.color * 200, (x, y), entity.detect_threshold * 350, 1
            )  # detection range

        ...
    # end for
    
    '''draw communication line'''
    if hasattr(self.world, "draw_line") and len(self.world.draw_line) > 0:
        for i in range(len(self.world.draw_line)):
            
            start_pos = self.draw_transform(*self.world.draw_line[i][0], cam_range)
            end_pos = self.draw_transform(*self.world.draw_line[i][1], cam_range)
            
            rtn = pygame.draw.line(self.screen, self.world.line_color * 200, start_pos, end_pos, 3)
    
"""convieniently transform coords"""
def draw_transform(self, x, y, cam_range):
    y *= (-1)  # this makes the display mimic the old pyglet setup (ie. flips image)
    x = (x / cam_range) * self.width // 2 * 0.9  # the .9 is just to keep entities from appearing "too" out-of-bounds
    y = (y / cam_range) * self.height // 2 * 0.9
    x += self.width // 2
    y += self.height // 2
    return x, y

```

the above code is also appended as follows:

### 4 code explanation

| file                                              | explanation                               |
| ------------------------------------------------- | ----------------------------------------- |
| ./Agent.py                                        | all the agents                            |
| ./Buffers.py                                      | implement the replay buffer               |
| ./MADDPG.py                                       | implement *MADDPG* algorithm              |
| ./main.py                                         | train a model                             |
| ./evaluate.py                                     | evaluate a trained model and generate gif |
| ./pettingzoo.mpe/_mpe_utils/simple_env.py         | change render code                        |
| ./pettingzoo.mpe/simple_tag/simple_tag_while_c.py | new environment                           |
| ./pettingzoo.mpe/simple_tag/myutils.py            | utility class and function                |
| ./pettingzoo.mpe/simple_tag/test_env.py           | to test the new environment if it works   |

### results

We train the network several time, in different ways. We also adjust the environment global args during different training progress. We train the network successfully and it converges at about 4 



A better training is to have ralitively high rewards at convergence and converge faster and more stable. Accoding to this, We find two main affective factors: reward shaping and learning rate decay.

| training setting | without reward shaping       | with reward shaping            |
| ---------------- | ---------------------------- | ------------------------------ |
| without lr_decay | converge, not very stable[1] | faster converge, not stable[3] |
| with lr_decay    | converge, stable[2]          | faster converge, stable[4]     |

the training reward curve:













