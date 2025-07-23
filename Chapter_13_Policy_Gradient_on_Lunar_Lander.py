import fire
import gymnasium as gym
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import math
import optax
import random

def mlp_init_params(prng_key, num_features, hidden_layer_sizes, num_classes):
  weights = []
  biases = []
  for num_prev, num_next in zip(
      [num_features] + hidden_layer_sizes, hidden_layer_sizes + [num_classes]):
    prng_key, sub_key = jax.random.split(prng_key)
    weights.append(jax.random.normal(
      sub_key, shape=[num_prev, num_next]) * math.sqrt(2 / num_prev))
    biases.append(jnp.zeros(num_next, dtype=float))
  return (weights, biases)


@jax.jit
def mlp_forward_pass(params, features):
  weights, biases = params
  output = features
  n_layer = 0
  for w, b in zip(weights, biases):
    if n_layer:
      output = jax.nn.relu(output)
    n_layer += 1
    output = jnp.dot(output, w) + b
  return output


@jax.jit
def policy_sample(prng_key, params, state_batch):
  logits = mlp_forward_pass(params, state_batch)
  return jax.random.categorical(prng_key, logits)


@jax.jit
def policy_loss_fn(params, state_batch, action_batch, reward_batch, num_episodes):
  logits = mlp_forward_pass(params, state_batch)
  indices = jnp.arange(state_batch.shape[0]), action_batch
  log_probs = logits[indices] - logsumexp(logits, axis=1)
  return -jnp.sum(log_probs * reward_batch) / num_episodes


class Agent:
  def __init__(self,
               gym_env_seed, jax_prng_seed, num_features, hidden_layer_sizes,
               num_actions, learning_rate):
    self.gym_env_seed = gym_env_seed
    self.prng_key = jax.random.key(jax_prng_seed)
    self.prng_key, sub_key = jax.random.split(self.prng_key)
    self.params = mlp_init_params(sub_key, num_features, hidden_layer_sizes, num_actions)
    self.optimizer = optax.adam(learning_rate)
    self.optimizer_state = self.optimizer.init(self.params)

  def act(self, observations):
    self.prng_key, sub_key = jax.random.split(self.prng_key)
    return policy_sample(sub_key, self.params, observations)

  def sample_episode(self, envs, batch_size):
    state, _ = envs.reset(seed=self.gym_env_seed)
    self.gym_env_seed += batch_size
    active = jnp.full([batch_size], True)
    state_list = []   # shape: [eps_len, batch_size, obs_size]
    action_list = []   # shape: [eps_len, batch_size]
    prv_reward_list = []   # shape: [eps_len, batch_size]
    active_list = []  # shape: [eps_len, batch_size]
    acc_rewards = jnp.zeros([batch_size], dtype=float)  # shape: [batch_size]
    num_success = 0
    while jnp.any(active):
      active_list.append(active)
      prv_reward_list.append(acc_rewards.copy())
      state_list.append(state)
      actions = self.act(state)
      action_list.append(actions)
      state, rewards, terminated, truncated, infos = envs.step(actions.tolist())
      num_success += jnp.sum(rewards[active & (truncated | terminated)] == 100.0)
      acc_rewards += jnp.where(active, rewards, jnp.zeros_like(rewards))
      active &= ~(terminated | truncated)
    episode_len = len(action_list)
    print(f"Longest episode: {episode_len}, successful landings: {num_success}")
    avg_reward = jnp.mean(acc_rewards)
    all_state = jnp.array(state_list).reshape([episode_len * batch_size, -1])
    all_actions = jnp.array(action_list).reshape([-1])
    # Ignore rewards before the state is encountered.
    all_future_reward = (acc_rewards[jnp.newaxis, :] - jnp.array(prv_reward_list)).reshape(-1)
    all_mask = jnp.array(active_list).reshape([-1])
    print('Masked percentage: ', jnp.sum(all_mask == False) / all_mask.size)
    return all_state, all_actions, all_future_reward, all_mask, avg_reward

  def train(self, envs):
    batch_size = len(envs.envs)
    all_state, all_actions, all_future_reward, all_mask, avg_reward = self.sample_episode(envs, batch_size)
    grads = jax.grad(policy_loss_fn)(self.params,
                                     all_state[all_mask],
                                     all_actions[all_mask],
                                     all_future_reward[all_mask],
                                     batch_size)
    updates, self.optimizer_state = self.optimizer.update(
      grads, self.optimizer_state, self.params)
    self.params = optax.apply_updates(self.params, updates)
    return avg_reward


def main(gym_env_seed: int | None = None,
         jax_prng_seed: int=0,
         episodes_per_batch: int=128,
         num_train_steps: int=200,
         policy_network_hidden_layers: list[int]=[8],
         policy_network_learning_rate: float=0.05):
  if not jax_prng_seed:
    jax_prng_seed = random.randint(1, 2 ** 32)
    print("RNG Seed for JAX: ", jax_prng_seed)
  if not gym_env_seed:
    gym_env_seed = random.randint(1, 2 ** 32)
    print("Gymnasium env seed: ", gym_env_seed)
  dummy_env = gym.make("LunarLander-v3")
  num_features = dummy_env.observation_space.shape[0]
  num_actions = dummy_env.action_space.n
  dummy_env.close()
  envs = gym.make_vec("LunarLander-v3", num_envs=episodes_per_batch, vectorization_mode="sync")
  agent = Agent(gym_env_seed,
                jax_prng_seed,
                num_features,
                policy_network_hidden_layers,
                num_actions,
                policy_network_learning_rate)
  for n_step in range(num_train_steps):
    avg_reward = agent.train(envs)
    print("Step %3d: avg_reward [%8.3f]" % (n_step, avg_reward))
  envs.close()

  env = gym.make("LunarLander-v3", render_mode="human")
  observation, _ = env.reset()
  acc_reward = 0.0
  for _ in range(2000):
    action = agent.act(jnp.array([observation])).tolist()[0]
    observation, reward, terminated, truncated, _ = env.step(action)
    acc_reward += reward
    if terminated or truncated:
      observation, _ = env.reset()
      print("Accumulated reward: ", acc_reward)
      acc_reward = 0
  env.close()

if __name__ == "__main__":
  fire.Fire(main)
