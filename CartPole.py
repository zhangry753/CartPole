import numpy as np
import tensorflow as tf
import  tflearn

import gym

env = gym.make("CartPole-v0")

# 尝试随机行动来测试环境
for _ in range(3):
    env.reset()
    reward_sum = 0
    done = False
    while not done:
        env.render()
        observation, reward, done, _ = env.step(env.action_space.sample())
        reward_sum += reward
    print("本episode的收益：{}".format(reward_sum))


# 神经网络超参数及训练参数
hidden_layer_neurons = 13
gamma = .99
obs_dim = len(env.reset())
act_dim = env.action_space.n
learning_rate = 5e-3
num_episodes = 1000
batch_size = 10
display_interval = 10

# 初始化agent的神经网络
class Model:
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, obs_dim], name="input_x")
        layer1 = tflearn.fully_connected(self.x, hidden_layer_neurons, activation=tf.nn.relu)
        layer2 = tflearn.fully_connected(layer1, act_dim, activation='linear')
        self.pred = tf.nn.softmax(layer2)
        self.trainable_vars = tf.trainable_variables()
        # loss
        self.actrual_act = tf.placeholder(tf.float32, [None, act_dim], name="input_y")
        self.advantages = tf.placeholder(tf.float32, [None], name="reward_signal")
        log_lik = tf.nn.softmax_cross_entropy_with_logits(labels=self.actrual_act, logits=layer2)
        loss = tf.reduce_mean(log_lik * self.advantages)
        # gradients, 手动传入梯度进行训练
        self.new_grads = tf.gradients(loss, self.trainable_vars)
        self.batch_grads = [
            tf.placeholder(tf.float32, name="batch_W1_grad"),
            tf.placeholder(tf.float32, name="batch_b1_grad"),
            tf.placeholder(tf.float32, name="batch_W2_grad"),
            tf.placeholder(tf.float32, name="batch_b2_grad")
        ]
        adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.update_grads = adam.apply_gradients(zip(self.batch_grads, self.trainable_vars))



# 初始化环境
rendering = False
agent = Model()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(num_episodes):
    obs_all = []
    act_all = []
    reward_all = []
    for _ in range(batch_size):
        obs_iter = []
        act_iter = []
        observation = env.reset()
        reward_iter = 0
        done = False
        while not done:
            obs_iter.append(observation)
            act_prob = sess.run(agent.pred, feed_dict={agent.x: [observation]})
            action = [np.random.choice(act_dim, 1, p=each) for each in act_prob]
            action_onehot = np.where(action==np.arange(act_dim), 1., 0.)
            act_iter.append(action_onehot[0])
            observation, reward, done, _ = env.step(action[0][0])
            reward_iter += reward
        obs_all.append(obs_iter)
        act_all.append(act_iter)
        reward_all.append(reward_iter)
    # 采样结束
    reward_avg = np.mean(reward_all)
    reward_delta = (reward_all - reward_avg)/np.std(reward_all)
    gradients = np.zeros_like(agent.trainable_vars)
    for i in range(batch_size):
        gradients += sess.run(agent.new_grads, feed_dict={
            agent.x:obs_all[i],
            agent.actrual_act:act_all[i],
            agent.advantages:reward_delta[i:i+1],
        })
    sess.run(agent.update_grads, feed_dict={
        agent.batch_grads[0]:gradients[0],
        agent.batch_grads[1]:gradients[1],
        agent.batch_grads[2]:gradients[2],
        agent.batch_grads[3]:gradients[3],
    })
    if step % 10 ==0:
        print("episode = {} 时的平均收益：{}".format(step, reward_avg))
    if reward_avg >= 195:
        print("问题在episode = {} 时解决！".format(step))
        break

# 去除随机决策，测试agent的性能
while True:
    observation = env.reset()
    reward_sum = 0
    done = False
    while not done:
        env.render()
        x = np.reshape(observation, [1, obs_dim])
        y = sess.run(agent.pred, feed_dict={agent.x: x})
        y = 0 if y[0][0] > 0.5 else 1
        observation, reward, done, _ = env.step(y)
        reward_sum += reward
    print("最终分数: {}".format(reward_sum))