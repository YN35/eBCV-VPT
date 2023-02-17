import gym
import pyglet

from pyvirtualdisplay import Display

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# PygletのOpenGLコンテキストを作成する
config = pyglet.gl.Config(double_buffer=True)
context = config.create_context()

# Parallel environments
env = make_vec_env("CartPole-v1", n_envs=4)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25000)
model.save("ppo_cartpole")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_cartpole")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    img = env.render(mode='rgb_array')
    
env.close()
window.close()