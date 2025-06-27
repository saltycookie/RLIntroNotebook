from absl import app
from absl import flags
import gymnasium as gym
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import math
import optax
import random

FLAGS = flags.FLAGS

flags.DEFINE_integer("prng_seed", 0, "Seed for JAX's pseudo random number generator.")

flags.DEFINE_integer("batch_size", 128,
                     "Number of episodes to simulate simultaneously for one gradient update.")

flags.DEFINE_float("learning_rate", 0.05, "Learning rate.")

flags.DEFINE_integer("train_steps", 200, "Number of training steps.")


def mlp_init_params(prng_key, num_features, hidden_layer_sizes, num_classes):
  weights = []
  biases = []
  num_previous = num_features
  for layer_size in hidden_layer_sizes + [num_classes]:
    prng_key, sub_key = jax.random.split(prng_key)
    weights.append(jax.random.normal(
      sub_key, shape=[num_previous, layer_size]) * math.sqrt(2 / num_previous))
    biases.append(jnp.zeros(layer_size, dtype=float))
    num_previous = layer_size
  return (weights, biases)


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


def policy_sample(prng_key, params, obs_batch):
  logits = mlp_forward_pass(params, obs_batch)
  return jax.random.categorical(prng_key, logits)  


def policy_loss_fn(params, obs_batch, action_batch, reward_batch, num_episodes):
  logits = mlp_forward_pass(params, obs_batch)
  indices = jnp.arange(obs_batch.shape[0]), action_batch
  log_probs = logits[indices] - logsumexp(logits, axis=1)
  return -jnp.sum(log_probs * reward_batch) / num_episodes


class Agent:
  def __init__(self, prng_seed, num_features, hidden_layer_sizes, num_actions, learning_rate):
    self.prng_key = jax.random.key(prng_seed)
    self.prng_key, sub_key = jax.random.split(self.prng_key)
    self.params = mlp_init_params(sub_key, num_features, hidden_layer_sizes, num_actions)
    self.optimizer = optax.adam(learning_rate)
    self.optimizer_state = self.optimizer.init(self.params)

  def act(self, observations):
    self.prng_key, sub_key = jax.random.split(self.prng_key)
    return policy_sample(sub_key, self.params, observations)

  def train(self, envs, batch_size):
    obs, _ = envs.reset(seed=42)
    any_active = True
    active = jnp.full([batch_size], True)
    obs_list = []   # shape: [eps_len, batch_size, obs_size]
    act_list = []   # shape: [eps_len, batch_size]
    prv_reward_list = []   # shape: [eps_len, batch_size]
    active_list = []  # shape: [eps_len, batch_size]
    acc_rewards = jnp.zeros([batch_size], dtype=float)  # shape: [batch_size]
    num_success = 0
    while jnp.any(active):
      active_list.append(active)
      prv_reward_list.append(acc_rewards.copy())
      obs_list.append(obs)
      actions = self.act(obs)
      act_list.append(actions)
      obs, rewards, terminated, truncated, infos = envs.step(actions.tolist())
      num_success += jnp.sum(rewards[active & (truncated | terminated)] == 100.0)
      acc_rewards += jnp.where(active, rewards, jnp.zeros_like(rewards))
      active &= ~(terminated | truncated)
    episode_len = len(act_list)
    print("Longest episode length: ", episode_len)
    print("Number of successful landings: ", num_success)
    avg_reward = jnp.mean(acc_rewards)
    all_obs = jnp.array(obs_list).reshape([episode_len * batch_size, -1])
    all_act = jnp.array(act_list).reshape([-1])
    # Ignore rewards before the state is encountered.
    all_future_reward = (acc_rewards[jnp.newaxis, :] - jnp.array(prv_reward_list)).reshape(-1)
    all_mask = jnp.array(active_list).reshape([-1])
    print('Masked percentage: ', jnp.sum(all_mask == False) / all_mask.size)
    uniq_actions, frequency_count = jnp.unique(all_act[all_mask], return_counts=True)
    print("Actions frequency: ", dict(zip(uniq_actions.tolist(), frequency_count.tolist())))
    grads = jax.grad(policy_loss_fn)(self.params,
                                     all_obs[all_mask],
                                     all_act[all_mask],
                                     all_future_reward[all_mask],
                                     batch_size)
    updates, self.optimizer_state = self.optimizer.update(
      grads, self.optimizer_state, self.params)
    self.params = optax.apply_updates(self.params, updates)
    return avg_reward


def main(argv):
  prng_seed = random.randint(1, 2 ** 32) if FLAGS.prng_seed == 0 else FLAGS.prng_seed
  print("Seed for generating Pseudo Random Number: ", prng_seed)
  dummy_env = gym.make("LunarLander-v3")
  num_features = dummy_env.observation_space.shape[0]
  num_actions = dummy_env.action_space.n
  dummy_env.close()
  envs = gym.make_vec("LunarLander-v3", num_envs=FLAGS.batch_size, vectorization_mode="sync")
  agent = Agent(prng_seed, num_features, [8], num_actions, FLAGS.learning_rate)
  for n_step in range(FLAGS.train_steps):
    avg_reward = agent.train(envs, FLAGS.batch_size)
    print("Step %3d: avg_reward [%8.3f]" % (n_step, avg_reward))
    print()
  envs.close()


if __name__ == "__main__":
  app.run(main)
