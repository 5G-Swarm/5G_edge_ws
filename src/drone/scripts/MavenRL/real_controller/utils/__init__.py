from gym.envs.registration import register

register(
    id='Real_env-v0',
    entry_point='real_env:Real_env',
    max_episode_steps=200,
)