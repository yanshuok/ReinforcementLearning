import tensorflow as tf
import gym, os
from scipy.signal import lfilter
from tensorflow.python.keras import models, losses, layers
import numpy as np
#os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
tf.enable_eager_execution()

np.random.seed(0)
tf.set_random_seed(0)

steps_per_epoch = 1000


# 计算折扣回报
def discount_cumsum(x, discount):
        return lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


# Actor
class Actor(tf.keras.Model):
    def __init__(self, action_dim):
        super(Actor, self).__init__()
        self.dense1 = layers.Dense(units=32, activation=tf.nn.tanh)
        self.dense2 = layers.Dense(units=32, activation=tf.nn.tanh)
        self.out = layers.Dense(units=action_dim, activation=None)

    def call(self, state):
        result = self.dense1(state)
        result = self.dense2(result)
        return self.out(result)


#Critic
class Critic(tf.keras.Model):
    def __init__(self):
        super(Critic, self).__init__()
        self.dense1 = layers.Dense(units=32, activation=tf.nn.tanh)
        self.dense2 = layers.Dense(units=32, activation=tf.nn.tanh)
        self.out = layers.Dense(units=1)

    def call(self, state):
        result = self.dense1(state)
        result = self.dense2(result)
        return self.out(result)


# 存储episode
class ReplayBuffer():

    # 最长序列5000
    def __init__(self):
        self.states = np.zeros([5000, 4])
        self.actions = np.zeros([5000, 2])
        self.rewards = np.zeros([5000])
        self.values = np.zeros([5000])
        self.adv_buffer = np.zeros([5000])
        self.ret_buffer = np.zeros([5000])
        self.ptr_start, self.ptr_end = 0, 0

    # 存储一个transition
    def save(self, s, a, r, v):
        self.states[self.ptr_end] = s
        ohe_a = np.zeros([2])
        ohe_a[a] = 1
        self.actions[self.ptr_end] = ohe_a
        self.rewards[self.ptr_end] = r
        self.values[self.ptr_end] = v
        self.ptr_end += 1

    # 完成寄一个episode之后计算advantage,return
    def final_path(self, last_value, gamma):
        vals = np.append(self.values[self.ptr_start: self.ptr_end], last_value)
        rews = np.append(self.rewards[self.ptr_start: self.ptr_end], last_value)
        deltas = rews[:-1] + gamma * vals[1:] - vals[:-1] #计算Advantage
        self.adv_buffer[self.ptr_start: self.ptr_end] = discount_cumsum(deltas, gamma)
        self.ret_buffer[self.ptr_start: self.ptr_end] = discount_cumsum(rews, gamma)[:-1]
        self.ptr_start = self.ptr_end

    # 获取所有的transition
    def get(self):
        # 标准化
        mean = np.mean(self.adv_buffer)
        std = np.std(self.adv_buffer)
        self.adv_buffer -= mean
        self.adv_buffer /= std
        states, actions, adv, ret = self.states[:self.ptr_end], \
                                    self.actions[:self.ptr_end], \
                                    self.adv_buffer[:self.ptr_end], \
                                    self.ret_buffer[:self.ptr_end]
        self.ptr_start, self.ptr_end = 0, 0
        return states, actions, adv, ret

# 直接梯度下降
def train(actor, critic, buffer, actor_optimizer, critic_optimizer):
    states, actions, adv_buffer, ret_buffer = buffer.get()
    with tf.GradientTape(persistent=True) as tape:
        logp = tf.nn.log_softmax(actor(states))
        critic_loss = tf.reduce_mean((critic(states) - ret_buffer)**2)
        actor_loss = -tf.reduce_mean(tf.reduce_sum(logp * actions, 1) * adv_buffer)

    grads = tape.gradient(actor_loss, actor.variables)
    actor_optimizer.apply_gradients(zip(grads, actor.variables))
    # 价值函数可以多来几次
    for _ in range(80):
        grads = tape.gradient(critic_loss, critic.variables)
        critic_optimizer.apply_gradients(zip(grads, critic.variables))



if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    actor = Actor(action_dim=2)
    critic = Critic()
    gamma = 0.99
    actor_optimizer = tf.train.AdamOptimizer(0.003)
    critic_optimizer = tf.train.AdamOptimizer(0.01)
    replay_buffer = ReplayBuffer()
    render = False
    for i in range(10000):
        state, reward, done, eps_reward = np.reshape(env.reset(), [-1, 4]), 0, 0, 0
        eps_rewards = []
        for t in range(steps_per_epoch):
            if render:
                env.render()
            action = int(tf.random.categorical(actor(state), 1).numpy()[0])
            value = critic(state).numpy()[0]
            replay_buffer.save(state, action, reward, value)
            observation, reward, done, _ = env.step(action)
            eps_reward += reward
            if done or t == steps_per_epoch - 1:
                if done:
                    eps_rewards.append(eps_reward)
                last_val = reward if done else critic(np.reshape(observation, [-1, 4])).numpy()[0]
                replay_buffer.final_path(last_val, 0.99)
                state, reward, done, eps_reward = np.reshape(env.reset(), [-1, 4]), 0, 0, 0
            state = np.reshape(observation, [-1, 4])
        print(i, max(eps_rewards), min(eps_rewards), np.mean(eps_rewards))
        if np.mean(eps_rewards) == 200:
            render = True
        train(actor, critic, replay_buffer, actor_optimizer, critic_optimizer)
