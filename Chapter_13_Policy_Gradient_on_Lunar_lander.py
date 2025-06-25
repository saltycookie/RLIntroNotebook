from absl import app
from absl import flags
import gymnasium as gym
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import math
import random

FLAGS = flags.FLAGS

flags.DEFINE_integer("prng_seed", 0, "Seed for JAX's pseudo random number generator.")

flags.DEFINE_integer("batch_size", 128,
                     "Number of episodes to simulate simultaneously for one gradient update.")

flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")

flags.DEFINE_integer("train_steps", 10, "Number of training steps.")


@jax.jit
def probabilistic_selection(prng_key, probabilities):
  cumulative_probs = jnp.cumsum(probabilities, axis=-1)
  random_numbers = jax.random.uniform(prng_key, (*probabilities.shape[:-1], 1))
  selected_indices = jnp.argmax(cumulative_probs > random_numbers, axis=-1)
  return selected_indices


class MLPNetwork:
  def __init__(self, prng_key, num_features, hidden_layer_sizes, num_classes):
    self.weights = []
    self.biases = []
    num_previous = num_features
    for layer_size in hidden_layer_sizes + [num_classes]:
      prng_key, sub_key = jax.random.split(prng_key)
      self.weights.append(jax.random.normal(
        sub_key, shape=[num_previous, layer_size]) * math.sqrt(2 / num_previous))
      self.biases.append(jnp.zeros(layer_size, dtype=float))
      num_previous = layer_size

  def logits(self, features):
      output = features
      n_layer = 0
      for w, b in zip(self.weights, self.biases):
        if n_layer:
          output = jax.nn.relu(output)
        n_layer += 1
        output = jnp.dot(output, w) + b
      return output

  def backprop(self, features, loss_fn, learning_rate):
    @jax.jit
    def train_fn(weights: list[jax.Array], biases: list[jax.Array]):
      output = features
      n_layer = 0
      for w, b in zip(weights, biases):
        if n_layer:
          output = jax.nn.relu(output)
        n_layer += 1
        output = jnp.dot(output, w) + b
      return loss_fn(output)

    grad_weights, grad_biases = jax.grad(train_fn, argnums=(0, 1))(self.weights, self.biases)
    new_weights = []
    for grad_w, w in zip(grad_weights, self.weights):
      new_weights.append(w - learning_rate * grad_w)
    self.weights = new_weights
    new_biases = []
    for grad_b, b in zip(grad_biases, self.biases):
      new_biases.append(b - learning_rate * grad_b)
    self.biases = new_biases


class Agent:
  def __init__(self, prng_seed, num_features, hidden_layer_sizes, num_classes):
    self.weights = []
    self.biases = []
    self.prng_key = jax.random.key(prng_seed)
    self.prng_key, sub_key = jax.random.split(self.prng_key)
    self.policy_network = MLPNetwork(sub_key, num_features, hidden_layer_sizes, num_classes)
    self.prng_key, sub_key = jax.random.split(self.prng_key)
    self.value_network = MLPNetwork(sub_key, num_features, hidden_layer_sizes, 1)

  def act(self, observations):
    logits = self.policy_network.logits(observations)
    probs = jnp.exp(logits - logsumexp(logits, axis=1)[:, jnp.newaxis])
    self.prng_key, sub_key = jax.random.split(self.prng_key)
    return probabilistic_selection(sub_key, probs)


  def train(self, envs, batch_size, learning_rate):
    obs, infos = envs.reset(seed=42)
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
    print("Longest episode length: ", len(act_list))
    print("Number of successful landings: ", num_success)
    avg_reward = jnp.mean(acc_rewards)
    all_obs = jnp.array(obs_list).reshape([len(obs_list) * batch_size, -1])
    # all_predicted_rewards = jnp.squeeze(self.value_network.logits(all_obs), axis=1) * 100
    all_acts = jnp.array(act_list).reshape([-1])
    # Ignore rewards before the state is encountered.
    all_future_rewards = (acc_rewards[jnp.newaxis, :] - jnp.array(prv_reward_list)).reshape(-1)
    all_target_rewards = all_future_rewards
    # all_target_rewards = all_future_rewards - all_predicted_rewards
    all_masks = jnp.array(active_list).reshape([-1])
    print('Masked percentage: ', jnp.sum(all_masks == False) / all_masks.size)
    uniq_actions, frequency_count = jnp.unique(all_acts[all_masks], return_counts=True)
    print("Actions frequency: ", dict(zip(uniq_actions.tolist(), frequency_count.tolist())))

    def get_policy_loss_fn(all_acts, all_rewards):
      def loss_fn(logits):
        indices = (jnp.arange(all_acts.shape[0]), all_acts)
        log_probs = logits[indices] - logsumexp(logits, axis=1)
        return -jnp.sum(log_probs * all_rewards) / batch_size
      return loss_fn

    self.policy_network.backprop(all_obs[all_masks],
                                 get_policy_loss_fn(all_acts[all_masks],
                                                    all_target_rewards[all_masks]),
                                 learning_rate)

    
    # def get_value_loss_fn(all_rewards):
    #   def loss_fn(logits):
    #     return jnp.mean((all_rewards - logits) ** 2)
    #   return loss_fn

    # self.value_network.backprop(all_obs[all_masks],
    #                             get_value_loss_fn(all_future_rewards[all_masks] / 100.0),
    #                             0.01)

    return avg_reward


def main(argv):
  prng_seed = random.randint(1, 2 ** 32) if FLAGS.prng_seed == 0 else FLAGS.prng_seed
  print("Seed for generating Pseudo Random Number: ", prng_seed)
  dummy_env = gym.make("LunarLander-v3")
  num_features = dummy_env.observation_space.shape[0]
  num_actions = dummy_env.action_space.n
  dummy_env.close()
  envs = gym.make_vec("LunarLander-v3", num_envs=FLAGS.batch_size, vectorization_mode="sync")
  agent = Agent(prng_seed, num_features, [8], num_actions)
  for n_step in range(FLAGS.train_steps):
    avg_reward = agent.train(envs, FLAGS.batch_size, FLAGS.learning_rate)
    print("Step %3d: avg_reward [%8.3f]" % (n_step, avg_reward))
    print()
  envs.close()


if __name__ == "__main__":
  app.run(main)
