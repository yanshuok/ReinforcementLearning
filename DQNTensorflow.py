import gym
from collections import deque
import numpy as np
import tensorflow as tf
import random

np.random.seed(0)
tf.set_random_seed(0)
class DQN(object):

    def __init__(self, batch_size, gamma, action_dim, state_dim, epsilon):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.input_state = tf.placeholder(shape=[None, state_dim], dtype=tf.float32)
        self.input_y = tf.placeholder(shape=[None, action_dim], dtype=tf.float32)
        self.dqn = self.build_net()
        loss = tf.losses.mean_squared_error(self.input_y, self.dqn)
        self.train_op = tf.train.AdamOptimizer(0.01).minimize(loss)
        self.epsilon = epsilon
        self.max_replays = 5000
        self.replays = deque(maxlen=self.max_replays)
        self.batch_size = batch_size
        self.gamma = gamma
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    # 建立网络
    def build_net(self):
        h1 = tf.layers.dense(self.input_state, units=24, activation=tf.nn.tanh)
        h2 = tf.layers.dense(h1, units=48, activation=tf.nn.tanh)
        h3 = tf.layers.dense(h2, units=self.action_dim, activation=None)
        return h3

    # 获得Q值
    def predict(self, state):
        return self.sess.run(self.dqn, {self.input_state: np.reshape(state, [-1, self.state_dim])})[0]

    # 根据状态获取动作 采用epsilon-greedy
    def get_action(self, state):
        if np.random.uniform(0, 1) > self.epsilon:
            return np.argmax(self.predict(state))
        else:
            return np.random.randint(0, self.action_dim)

    def get_action_value(self, state):
        q = self.predict(state)
        if np.random.uniform(0, 1) > self.epsilon:
            return np.max(q)
        else:
            return q[np.random.randint(0, self.action_dim)]

    def save_experience(self, s, a, r, d, s_):
        self.replays.append([s, a, r, d, s_])

    def update_epislon(self, decay, min):
        if self.epsilon > min:
            self.epsilon *= decay

    def train(self):
        # 从replay Buffer采样
        batch = random.sample(self.replays, self.batch_size)
        batch_x, batch_y = [], []
        for x in batch:
            # 状态
            batch_x.append(x[0])
            action = x[1]
            reward = x[2]
            done = x[3]
            next_state = np.reshape(x[4], [-1, self.state_dim])
            y_target = self.predict(x[0])
            # 如果是下一个状态是最终状态Q就是reward 否则y=r+gamma*max(a)Q(s')
            y_target[action] = reward if done else reward + self.gamma * (np.max(self.predict(next_state)))
            batch_y.append(y_target)
        self.sess.run(self.train_op, {self.input_state: batch_x, self.input_y: batch_y})

if __name__ == '__main__':
    # replay buffer 有64个样本的时候开始训练
    warmup = 64
    # CartPole动作维度为2，状态维度为4
    env = gym.make('CartPole-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    score = deque(maxlen=100)
    agent = DQN(batch_size=32, gamma=0.995, action_dim=action_dim, state_dim=state_dim, epsilon=1)
    eps = 0
    render = False
    # 全部episode数量
    for i in range(3000):
        # 每次循环开始时重置环境
        state = env.reset()
        eps_reward = 0
        while True:
            if render:
                env.render()
            # 获得一个动作
            action = agent.get_action(state)
            # 执行一个动作
            observation, reward, done, _ = env.step(action)
            # 保存experience
            agent.save_experience(state, action, reward, done, observation)
            # 记录 reward
            eps_reward += reward
            if done:
                score.append(eps_reward)
                agent.update_epislon(0.999, 0.1)
                break
            state = observation
            eps+=1
        print(i,eps_reward)
        # 官方设定 连续100次reward的平均分大于195就解决了这个问题
        if np.mean(score)>195:
            print('CartPole Solved at',i)
            render = True
        if len(agent.replays) >= 32:
            agent.train()
