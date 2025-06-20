import flag
import gymnasium as gym
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import math
import random


flag_prng_seed = flag.int("prng_seed", 0, "Seed for JAX's pseudo random number generator.")

flag_batch_size = flag.int("batch_size", 64, "Number of episodes to simulate simultaneously for one gradient update.")

flag_learning_rate = flag.int("learning_rate", 0.00001, "Learning rate.")

flag_train_steps = flag.int("train_steps", 10, "Number of training steps.")


def probabilistic_selection(prng_key, probabilities):
  cumulative_probs = jnp.cumsum(probabilities, axis=-1)
  random_numbers = jax.random.uniform(prng_key, (*probabilities.shape[:-1], 1))
  selected_indices = jnp.argmax(cumulative_probs > random_numbers, axis=-1)
  return selected_indices


def mlp_logits_fn(weights, biases, observations):
  output = observations
  n_layer = 0
  for w, b in zip(weights, biases):
    if n_layer:
      output = jax.nn.relu(output)
    n_layer += 1
    output = jnp.dot(output, w) + b
  return output


class Agent:
  def __init__(self, prng_seed, num_features, hidden_layer_sizes, num_classes):
    self.weights = []
    self.biases = []
    self.prng_key = jax.random.key(prng_seed)
    num_previous = num_features
    for layer_size in hidden_layer_sizes + [num_classes]:
      self.prng_key, sub_key = jax.random.split(self.prng_key)
      self.weights.append(jax.random.normal(
        sub_key, shape=[num_previous, layer_size]) * math.sqrt(2 / num_previous))
      self.biases.append(jnp.zeros(layer_size, dtype=float))
      num_previous = layer_size


  def act(self, observations):
    logits = mlp_logits_fn(self.weights, self.biases, observations)
    probs = jnp.exp(logits - logsumexp(logits, axis=1)[:, jnp.newaxis])
    self.prng_key, sub_key = jax.random.split(self.prng_key)
    return probabilistic_selection(sub_key, probs)


  def train(self, envs, batch_size, delta):
    obs, infos = envs.reset(seed=42)
    any_active = True
    active = jnp.full([batch_size], True)
    obs_list = []   # shape: [eps_len, batch_size, obs_size]
    act_list = []   # shape: [eps_len, batch_size]
    active_list = []  # shape: [eps_len, batch_size]
    acc_rewards = jnp.zeros([batch_size], dtype=float)  # shape: [batch_size]
    num_success = 0
    while jnp.any(active):
      active_list.append(active)
      obs_list.append(obs)
      actions = self.act(obs)
      act_list.append(actions)
      obs, rewards, terminated, truncated, infos = envs.step(actions.tolist())
      num_success += jnp.sum(rewards[truncated | terminated] == 100.0)
      acc_rewards += jnp.where(active, rewards, jnp.zeros_like(rewards))
      active &= ~(terminated | truncated)
    print("Longest episode length: ", len(act_list))
    print("Number of successful landings: ", num_success)
    avg_reward = jnp.mean(acc_rewards)
    all_rewards = (jnp.zeros((len(act_list), batch_size)) + acc_rewards[jnp.newaxis, :]).reshape([-1])
    all_obs = jnp.array(obs_list).reshape([len(obs_list) * batch_size, -1])
    all_acts = jnp.array(act_list).reshape([-1])
    all_masks = jnp.array(active_list).reshape([-1])
    print('masked percentage: ', jnp.sum(all_masks == False) / all_masks.size)
    uniq_actions, frequency_count = jnp.unique(all_acts[all_masks], return_counts=True)
    print("actions frequency: ", dict(zip(uniq_actions.tolist(), frequency_count.tolist())))

    def get_obj_fn(all_obs, all_acts, all_rewards):
      def obj_fn(weights, biases):
        logits = mlp_logits_fn(weights, biases, all_obs)
        indices = (jnp.arange(all_acts.shape[0]), all_acts)
        log_probs = logits[indices] - logsumexp(logits, axis=1)
        return jnp.sum(log_probs * all_rewards)
      return obj_fn

    grad_weights, grad_biases = jax.grad(
      get_obj_fn(all_obs[all_masks], all_acts[all_masks], all_rewards[all_masks]), argnums=(0, 1))(self.weights, self.biases)
    for i in range(len(self.weights)):
      self.weights[i] += delta * grad_weights[i]
    for i in range(len(self.biases)):
      self.biases[i] += delta * grad_biases[i]
    return avg_reward


if __name__ == "__main__":
  flag.parse()
  prng_seed = random.randint(1, 2 ** 32) if flag_prng_seed.value == 0 else flag_prng_seed.value
  envs = gym.make_vec("LunarLander-v3", num_envs=flag_batch_size.value, vectorization_mode="sync")
  agent = Agent(prng_seed, 8, [8], 4)
  for n_step in range(flag_train_steps.value):
    avg_reward = agent.train(envs, flag_batch_size.value, flag_learning_rate.value)
    print("Step %3d: avg_reward [%8.3f]" % (n_step, avg_reward))
    print()
  envs.close()
