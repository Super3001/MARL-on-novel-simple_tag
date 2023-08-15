
'''# communication of all other agents
comm = []
other_pos = []
other_vel = []
for other in world.agents:
    if other is agent:
        continue
    comm.append(other.state.c)
    other_pos.append(other.state.p_pos - agent.state.p_pos)
    if not other.adversary:
        other_vel.append(other.state.p_vel)'''