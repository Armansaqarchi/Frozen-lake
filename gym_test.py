
#%%
import copy

import numpy as np
import gymnasium as gym
from IPython.display import HTML
from base64 import b64encode
import imageio
#%% md
## Utils
#%%
def record_video(env, policy, out_directory, fps=1, random_action=False, max_steps=100):
    images = []
    done = False
    truncated = False
    state, info = env.reset()
    img = env.render()
    images.append(img)
    total_reward = 0
    i = 0
    while not done and not truncated:
        i += 1
        if i > max_steps:
            break
        action = np.random.randint(4) if random_action else policy[state]
        state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        img = env.render()
        images.append(img)
        if not random_action:
            print(f"action: {action}, state: {state}, reward: {reward}, done: {done}, truncated: {truncated}, info: {info}")
    imageio.mimsave(out_directory, [np.array(img) for i, img in enumerate(images)], fps=fps)
    return total_reward
#%%
def show_video(video_path, video_width=500):
    video_file = open(video_path, "r+b").read()
    video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
    return HTML(f"""<video width={video_width} controls><source src="{video_url}"></video>""")
#%% md
## Random Walk
#%%
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode='rgb_array')
total_reward = record_video(env, None, 'frozenlake_random.mp4', fps=3, random_action=True)
print(f"total reward: {total_reward}")
show_video('frozenlake_random.mp4', video_width=500)
