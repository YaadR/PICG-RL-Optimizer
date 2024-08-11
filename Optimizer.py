#%%
from Agent import Agent,ReplayBuffer,deque
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import entropy
from utils import create_video_from_frames,custom_argmax
from runx import Environment,tqdm
import warnings

# Ignore all warnings
# warnings.filterwarnings("ignore")

#%% Reward calculations
def consequtive_unique(vec):
    count=1
    vec = [round(element, 3) for element in vec]
    for i in range(1, len(vec)):
        if vec[i] == vec[i - 1]:
            pass
        else:
            count+= 1
    return count
        
def reward_calc_o(objective_prev,objective):
    if objective_prev == 0:
        return 0

    diff = objective_prev-objective
    diff = diff/(diff**2+1e-03)

    if objective < 3:
        return -1

    if abs(objective-6)<2:
        return objective
    
    return float(diff) if diff > 0 else float(diff*2)

def reward_calc_t(vec,vec_prev,target=15):
    prev_dist = abs(target - consequtive_unique(vec_prev))
    dist = abs(target - consequtive_unique(vec))
    direction = (prev_dist - dist)
    if dist<5 :
        return 5    
    if direction >= 0:
        return 1   #*np.float32(((target*2)/(abs(dist)+1))) 
    elif direction < 0:
        return -1  #*np.float32(((target*2)/(abs(dist)+1))) 

def reward_calc(s1,s2,o1,o2):
    r1 = reward_calc_o(o1,o2)
    r2 = reward_calc_t(s1,s2)
    optional_reward.append(r2)
    # return round((r1+r2)/2)
    return r1
optional_reward = [0]
#%% Initializations:

# Initialize Environment:
env = Environment()
mp,xp,f,m,x0 = env.create_data()

# Initialize RL agent:
action_size = 6
state_size = len(env.state)
agent = Agent(state_size=state_size+2,action_size=action_size,sigma=0.2)

# Initialize the replay buffer
replay_buffer = ReplayBuffer(capacity=1000)
replay_init = 100
replay_scene_iteration = replay_init//5
while replay_buffer.__len__()<replay_init:
    iter=0
    env.reset()
    state = env.state
    state = np.concatenate((state, np.array([env.w,env.delta])))
    while (replay_buffer.__len__()<replay_init) and iter<50:
        print(f"Replay Buffer Initialization {round(100*(replay_buffer.__len__()/replay_init))}%", end="\r") 
        iter+=1
        
        state = np.array(state)
        action,policy = agent.get_action(state)
        next_state = env.step(action)
        done = 1 if iter == 50 else 0
        # reward = reward_calc(next_state,state)
        # reward = reward_calc(env.objective_prev,env.O(env.state,env.zero_state,env.delta,env.w))
        reward = reward_calc(next_state,state,env.objective_prev,env.O(env.state,env.zero_state,env.delta,env.w))
        
        next_state = np.concatenate((next_state, np.array([env.w,env.delta])))
        replay_buffer.push(state, policy, reward, next_state, done)
        state = next_state
        if done or (None in state):
            break

num_rounds = 300
num_episodes = 15


# Setting plots
fig, ((ax1,ax3,ax5),(ax2,ax4,ax6)) = plt.subplots(2, 3, figsize=(12, 6))
MAKE_VIDEO = True
index = 0

def update_plot():
    line3_2.set_data(m, env.state)  
    ax3.relim()  # Update the limits of the axes
    ax3.autoscale_view()  # Autoscale the axes

    line1_1.set_data(np.arange(len(env.delta_array)),env.delta_array)
    ax1.relim()
    ax1.autoscale_view() 

    line2_1.set_data( np.arange(len(env.w_array)), env.w_array)
    ax2.relim()
    ax2.autoscale_view() 

    # line4_1.set_data( np.arange(len(agent.entropy_array)), agent.entropy_array)
    line4_1.set_data( np.arange(len(optional_reward)), optional_reward) 
    ax4.relim()
    ax4.autoscale_view() 

    line5_1.set_data( np.arange(len(cumulative_reward_array)),cumulative_reward_array)
    ax5.relim()
    ax5.autoscale_view() 

    # line6_1.set_data( np.arange(len(reward_array)), reward_array)
    line6_1.set_data( np.arange(len(objective_array)), objective_array)
    ax6.relim()
    ax6.autoscale_view() 

    plt.pause(0.01)  # Pause to allow the plot to update

def initiate_plot():
    plt.cla()
    ax1.cla()
    ax2.cla()
    ax3.cla()
    ax4.cla()
    ax5.cla()
    ax6.cla()   
    
    ax1.set_title('delta', ha='left', va='top', fontsize=18, color='purple')
    ax2.set_title( 'w', ha='left', va='top', fontsize=18, color='blue')
    ax3.set_title(f"Generation: {i+1}")
    # ax4.set_title("Policy Entropy")
    ax4.set_title("Optional Reward")
    ax5.set_title("Cumulative Reward")
    # ax6.set_title("Reward")
    ax6.set_title("Objective Function")

    ax1.set_yscale('log')
    ax2.set_yscale('log')

    dots, = ax3.plot(mp, xp, 'o',color='purple') 
    line1_1, = ax1.plot(np.arange(len(env.delta_array)),env.delta_array , '-',color='purple')  
    line2_1, = ax2.plot( np.arange(len(env.w_array)), env.w_array, '-',color='blue')
    line3_2, = ax3.plot(m, x0, '-',color='orange')  
    line3_2, = ax3.plot( m, env.state, '-',color='green')
    # line4_1, = ax4.plot( np.arange(len(agent.entropy_array)), agent.entropy_array, '-',color='green') 
    line4_1, = ax4.plot( np.arange(len(optional_reward)), optional_reward, '-',color='green')
    line5_1, = ax5.plot( np.arange(len(cumulative_reward_array)), cumulative_reward_array, '-',color='orange')
    # line6_1, = ax6.plot( np.arange(len(reward_array)), reward_array, '-',color='pink')
    line6_1, = ax6.plot( np.arange(len(objective_array)), objective_array, '-',color='pink')

    return dots,line1_1,line2_1,line3_2,line4_1,line5_1,line6_1

#%% Training loop
for i in range(num_episodes):
    env.reset()
    state = env.state
    agent.entropy_array = [entropy([1/(action_size//2) for _ in range((action_size//2)) ])]
    reward_array = deque(maxlen=100)
    cumulative_reward_array = [0]
    objective_array = []
    optional_reward = deque(maxlen=80)
    dots,line1_1,line2_1,line3_2,line4_1,line5_1,line6_1 = initiate_plot()

    state = np.concatenate((state, np.array([env.w,env.delta])))
    for j in tqdm(range(num_rounds), desc="Optimization:"):
    # for j in range(num_rounds):
        
        
        state = np.array(state)

        action,policy = agent.get_action(state)
        next_state = env.step(action)

        objective_array.append(env.objective_prev)
        # reward = reward_calc(next_state,state)
        # reward = reward_calc(env.objective_prev,env.O(env.state,env.zero_state,env.delta,env.w))
        reward = reward_calc(next_state,state,env.objective_prev,env.O(env.state,env.zero_state,env.delta,env.w))
        done = 1 if i == num_rounds-1 else 0

        next_state = np.concatenate((next_state, np.array([env.w,env.delta])))

        # Train Online
        # agent.train(state, policy, reward, next_state, done)
        
        # Store experience in replay buffer
        replay_buffer.push(state, policy, reward, next_state, done)

        state  = next_state

        # Reward Arrays
        reward_array.append(reward)
        cumulative_reward_array.append(sum(reward_array))
        
        # Create a video frame
        if MAKE_VIDEO:
            plt.savefig(f'frames/frame_{index}.png')  # Save each frame as an image
            index+=1
        update_plot()

        # Replay Batch Train
        if done or (j%25==0):
            # Sample from replay buffer and train the agent
            batch = replay_buffer.sample(batch_size=64)
            agent.train(*batch)
            if done or (max(state)-min(state)>2.5):
                break

    # plt.plot(mp, xp, 'o', m, x0, '-', m, xi, '-')

#%% Results

if MAKE_VIDEO:
    frames_directory = 'frames'
    output_filename = 'output\output_video.mp4' 
    create_video_from_frames(frames_directory, output_filename)
# %%

    