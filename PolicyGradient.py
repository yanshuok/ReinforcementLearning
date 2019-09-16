import tensorflow as tf
import numpy as np
import gym, os
from collections import deque

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

np.random.seed(0)
tf.set_random_seed(0)


class PolicyGradient(object):

    def __init__(self, state_dim, action_dim, gamma):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.input_state = tf.placeholder(dtype=tf.float32, shape=[None, state_dim])
        self.input_action = tf.placeholder(dtype=tf.int32, shape=[None, ])
        self.input_adv = tf.placeholder(dtype=tf.float32, shape=[None, ])
        self.states, self.actions, self.rewards = [], [], []
        self.action = self.build_network()
        self.probabilities = tf.nn.softmax(self.action)
        # 构建损失函数
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.action, labels=tf.one_hot(self.input_action, depth=action_dim))*self.input_adv)
        self.train_op = tf.train.AdamOptimizer(0.01).minimize(loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def build_network(self):
        h1 = tf.layers.dense(self.input_state, units=64, activation=tf.nn.tanh)
        h2 = tf.layers.dense(h1, units=64, activation=tf.nn.tanh)
        h3 = tf.layers.dense(h2, units=self.action_dim, activation=None)
        return h3

    # 计算折扣回报
    def compute_advantages(self, reword_exp):
        advantages = np.zeros_like(reword_exp)
        temp = 0
        for i in reversed(range(len(reword_exp))):
            temp = self.gamma*temp+reword_exp[i]
            advantages[i] = temp
        mean = np.mean(advantages)
        std = np.std(advantages)
        advantages -= mean
        advantages /= std
        return advantages

    # 选择一个动作
    def get_action(self, state):
        # 得到的是动作的概率分布，从中采样一个动作
        p = self.sess.run(self.probabilities, {self.input_state: np.reshape(state, [-1, 4])})[0]
        return np.random.choice(range(self.action_dim), p=p)

    def save_experience(self, s, a, r):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)

    def train(self):
        states = np.vstack(self.states)
        advantages = self.compute_advantages(np.vstack(self.rewards))
        self.sess.run(self.train_op, {self.input_state: states,
                                      self.input_action: self.actions,
                                      self.input_adv: advantages.flatten()})
        self.states, self.actions, self.rewards = [], [], []

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    batch_size = 32
    render = False
    scores = deque(maxlen=100)
    agent = PolicyGradient(state_dim=4, action_dim=2, gamma=0.95)
    for i in range(10000):
        state = env.reset()
        eps_reward = 0
        while True:
            if render:
                env.render()
            action = agent.get_action(state)
            observation, reward, done, _ = env.step(action)
            agent.save_experience(state, action, reward)
            eps_reward += reward

            if done:
                agent.train()
                scores.append(eps_reward)
                if np.mean(scores)>=195:
                    print('solved')
                    #render=True
                break
            state = observation
        print(i,'reward:',eps_reward)