# test_env.py

from pettingzoo.mpe.simple_tag import simple_tag, simple_tag_while_c
import numpy as np
import time

# env_name = 'simple_tag'
env_name = 'simple_tag_while_c'

# render = False
render = True
episode_len = 100

if env_name == 'simple_tag':
    env = simple_tag.env()
    if render:
        env = simple_tag.env(max_cycles=25, render_mode='human')
    '''为什么obstacles会动？因为边界框在动，图像展示尺寸不固定'''
else:
    env = simple_tag_while_c.env()
    if render:
        env = simple_tag_while_c.env(max_cycles=100, render_mode='human')

# add some properties to env
# env.n = 4
env.n = 4

rtn = env.reset()
# print(rtn) # None
done = False
# exit(0)

# for aid, agent in enumerate(env.agent_iter()):     
#     print(aid, agent)
#     env.step()

# 与环境交互进行训练
num_steps = 0
cumulated_rewards = np.zeros(env.n)
# while not done:
for aid, agent in enumerate(env.agent_iter()):
    # print(num_steps, aid, agent)
    num_steps += 1
    
    if render: env.render()
    time.sleep(0.05)
    # time.sleep(1)
    
    new_observation, rewards, terminated, truncated, info = env.last()
    # if num_steps > 8:
    #     exit(0)
    
    done = terminated or truncated
    if done:
        break
    
    action = env.action_space(agent).sample()
    if not render: print(num_steps, agent, action, rewards)
    
    env.step(action)
    
    cumulated_rewards[aid % 4] += rewards
    
    # observations = new_observation
    
# end while
    
# 打印回报
print('return:', cumulated_rewards)
'''为什么return都一样？ return没有都一样'''

print('num_steps:', num_steps)
    
# 关闭环境    
env.close()