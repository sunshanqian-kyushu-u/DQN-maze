import pygame as pg
import numpy as np
import random
from torch import nn
import torch
import torch.nn.functional as F
from torch import optim

###
# agent
###
class Agent:
    def __init__(self):
        self.GAMMA = 0.99

        self.memory = ReplayMemory()

        self.online_net = DQN()
        self.target_net = DQN()

        self.target_net.load_state_dict(self.online_net.state_dict())  # 同步参数

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=1e-3)

###
# DQN
###
class DQN(nn.Module):
    def __init__(self):  # 状态是二维的 y, x，行为有上下左右
        super(DQN, self).__init__()
        self.n_states = 2
        self.n_actions = 4
        self.fc1 = nn.Linear(self.n_states, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.n_actions)

    def forward(self, x):
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))

    def act(self, state):
        state_tensor = torch.as_tensor(state, dtype=torch.float)
        q_value = self(state_tensor.unsqueeze(0))
        max_q_index = torch.argmax(input=q_value)
        action = max_q_index.detach().item()

        return action

    def suggest(self, state):
        state_tensor = torch.as_tensor(state, dtype=torch.float)
        q_value = self(state_tensor.unsqueeze(0))
        max_q_index = torch.argmax(input=q_value)
        action = max_q_index.detach().item()

        if action == 0:
            print("Suggestion: UP")
        elif action == 1:
            print("Suggestion: DOWN")
        elif action == 2:
            print("Suggestion: LEFT")
        elif action == 3:
            print("Suggestion: RIGHT")

###
# experience replay
###
class ReplayMemory:
    def __init__(self):
        self.CAPACITY = 1000
        self.BATCH_SIZE = 64

        self.n_s = 2
        self.n_a = 4

        self.all_s = np.empty(shape=(self.CAPACITY, self.n_s), dtype=int)
        self.all_a = np.random.randint(low=0, high=self.n_a, size=self.CAPACITY, dtype=int)
        self.all_r = np.empty(shape=self.CAPACITY, dtype=float)
        self.all_done = np.random.randint(low=0, high=2, size=self.CAPACITY, dtype=int)
        self.all_s_ = np.empty(shape=(self.CAPACITY, self.n_s), dtype=int)

        self.t_memo = 0
        self.t_max = 0

    def push(self, s, a, r, done, s_):
        self.all_s[self.t_memo] = s
        self.all_a[self.t_memo] = a
        self.all_r[self.t_memo] = r
        self.all_done[self.t_memo] = done
        self.all_s_[self.t_memo] = s_

        self.t_max = max(self.t_max, self.t_memo + 1)
        self.t_memo = (self.t_memo + 1) % self.CAPACITY

    def sample(self):
        if self.t_max >= self.BATCH_SIZE:
            indexes = random.sample(range(0, self.t_max), self.BATCH_SIZE)
        else:
            indexes = range(0, self.t_max)

        batch_s = []
        batch_a = []
        batch_r = []
        batch_done = []
        batch_s_ = []

        for index in indexes:
            batch_s.append(self.all_s[index])
            batch_a.append(self.all_a[index])
            batch_r.append(self.all_r[index])
            batch_done.append(self.all_done[index])
            batch_s_.append(self.all_s_[index])

        batch_s_tensor = torch.as_tensor(np.asarray(batch_s), dtype=torch.float)
        batch_a_tensor = torch.as_tensor(np.asarray(batch_a), dtype=torch.int64).unsqueeze(-1)
        batch_r_tensor = torch.as_tensor(np.asarray(batch_r), dtype=torch.float).unsqueeze(-1)
        batch_done_tensor = torch.as_tensor(np.asarray(batch_done), dtype=torch.int).unsqueeze(-1)
        batch_s__tensor = torch.as_tensor(np.asarray(batch_s_), dtype=torch.float)

        return batch_s_tensor, batch_a_tensor, batch_r_tensor, batch_done_tensor, batch_s__tensor

# 栈的实现
class Stack:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.items:
            return None
        else:
            return self.items.pop()


class Maze:
    def __init__(self):
        ###
        # 先把地图做出来，然后绘图
        ###
        self.x = 7  # x,y 需要是奇数
        self.y = 7
        self.map = np.zeros((self.y, self.x), dtype=int)  # 实际的地图
        self.map[1::2, 1::2] = 1  # 从第 1 行（列）开始，隔一行（列）取，即下面的“格子”
        # print(self.map)
        self.target = (random.randrange(1, self.y, 2), random.randrange(1, self.x, 2))  # 目标，第二个参数是开集
        # print("目标点：" + str(self.target))
        # return

        ###
        # 创建迷宫算法
        # 从 target 开始，观察四个方向是否有未被访问的格子
        # 若有，则将 target 入栈（先进后出），然后随机选择一个四周的格子并打破“墙”，将选择个格子作为新的 target
        # 如果没有空的格子，则进行出栈，将出栈的格子作为 target
        # 比如一个 3*3 的迷宫如下所示：
        # a b c
        # d e f
        # g h i
        # 假设 target 选择 b，则四周有未被访问的格子 a，c，e，那么把 b 入栈
        # 我随机选择了 e 点，同理周围有未被访问的格子 d，f 和 h（b 已经被访问过了），e 入栈
        # 我又随机选择了 d 点，未被访问的格子是 a 和 g，d 入栈
        # 选择 a，发现四周没有未被访问的格子，则把 d 出栈重新作为 target，发现周围还有个未被访问的 g，d 入栈
        # 如上循环
        ###
        self.stack = Stack()  # 自定义的栈
        self.visited = []  # 访问过的格子
        self.visited.append(self.target)  # 将目标放入“访问过”的列表中
        self.stack.push(self.target)  # 第一个格子周围必然有空，所以直接入栈
        self.t_target = self.target  # 当前被选择的格子
        # print(self.stack.items)
        # return

        while not self.stack.isEmpty():
            # 寻找四周未被访问的格子
            self.around_list = []
            if self.t_target[0] - 2 >= 1 and (self.t_target[0] - 2, self.t_target[1]) not in self.visited:
                self.around_list.append((self.t_target[0] - 2, self.t_target[1]))
            if self.t_target[1] - 2 >= 1 and (self.t_target[0], self.t_target[1] - 2) not in self.visited:
                self.around_list.append((self.t_target[0], self.t_target[1] - 2))
            if self.t_target[0] + 2 <= self.y - 2 and (self.t_target[0] + 2, self.t_target[1]) not in self.visited:
                self.around_list.append((self.t_target[0] + 2, self.t_target[1]))
            if self.t_target[1] + 2 <= self.x - 2 and (self.t_target[0], self.t_target[1] + 2) not in self.visited:
                self.around_list.append((self.t_target[0], self.t_target[1] + 2))

            if self.around_list:
                # 非空的话随机选取一个作为 s_target
                self.s_target = random.choice(self.around_list)
                # print("选择了：" + str(self.s_target))
                # 入栈
                self.stack.push(self.s_target)
                # 放入“访问过”列表
                self.visited.append(self.s_target)
                # 将之间的格子设为 1
                self.map[int((self.s_target[0] + self.t_target[0]) / 2),
                         int((self.s_target[1] + self.t_target[1]) / 2)] = 1
                # 将 s_target 赋值给 t_target
                self.t_target = self.s_target
            else:
                # 空的话出栈
                self.t_target = self.stack.pop()

        # 顺便在这把终点的值改为 2
        self.map[self.target[0], self.target[1]] = 2
        # print(self.map)

        # 绘图
        # 初始化
        pg.init()
        # 创建一个窗口对象
        self.BLOCK_PIXEL = 15  # 每个格子占的像素数
        self.PLAYER_BIAS = 1
        self.LEADER_BIAS = 3
        self.screen = pg.display.set_mode((self.x * self.BLOCK_PIXEL, self.y * self.BLOCK_PIXEL))
        # 更改标题
        pg.display.set_caption("DQNMaze")
        # 初始化背景为绿
        self.screen.fill("green")

        # 上色
        for y in range(self.y):
            for x in range(self.x):
                if self.map[y, x] == 0:                     # 墙涂灰
                    pg.draw.rect(self.screen, "gray",
                                 (x * self.BLOCK_PIXEL, y * self.BLOCK_PIXEL, self.BLOCK_PIXEL, self.BLOCK_PIXEL))
                else:                                       # 地面白
                    pg.draw.rect(self.screen, "white",
                                 (x * self.BLOCK_PIXEL, y * self.BLOCK_PIXEL, self.BLOCK_PIXEL, self.BLOCK_PIXEL))
        # 终点红
        pg.draw.rect(self.screen, "red",
                     (self.target[1] * self.BLOCK_PIXEL, self.target[0] * self.BLOCK_PIXEL,
                      self.BLOCK_PIXEL, self.BLOCK_PIXEL))

        # 随机选择一个非终点的出生点
        # print(np.nonzero(self.map))
        # print(np.nonzero(self.map)[0])
        # print(np.nonzero(self.map)[0].shape)
        while True:
            self.seed = random.randint(0, np.nonzero(self.map)[0].shape[0] - 1)  # 非 0 的即地面
            self.player = np.array([np.nonzero(self.map)[0][self.seed], np.nonzero(self.map)[1][self.seed]])
            if self.player[0] == self.target[0] and self.player[1] == self.target[1]:
                pass
            else:
                break
        # 起点黑
        pg.draw.rect(self.screen, "black",
                     (self.player[1] * self.BLOCK_PIXEL + self.PLAYER_BIAS,
                      self.player[0] * self.BLOCK_PIXEL + self.PLAYER_BIAS,
                      self.BLOCK_PIXEL - (2 * self.PLAYER_BIAS), self.BLOCK_PIXEL - (2 * self.PLAYER_BIAS)))

        # 刷新
        pg.display.flip()  # 这一步不能忘

    ###
    # 定义与 agent 交互的接口
    ###

    # 随机产生起始点
    def reset(self):
        while True:
            seed = random.randint(0, np.nonzero(self.map)[0].shape[0] - 1)  # 非 0 的即地面
            player = np.array([np.nonzero(self.map)[0][seed], np.nonzero(self.map)[1][seed]])
            if player[0] == self.target[0] and player[1] == self.target[1]:
                pass
            else:
                break
        return player

    # def reset(self):
    #     return self.player

    # 对于 state 和 action 给反应
    def step(self, state, action):
        if state[0] == self.target[0] and state[1] == self.target[1]:  # 说明是终点
            return np.array((-1, -1), dtype=int), -1, 1  # s_, r, done
        else:  # 不是终点
            if action == 0:  # 上
                if self.map[state[0] - 1, state[1]] == 0:  # 说明是墙
                    return np.array((-1, -1), dtype=int), -1, 1
                elif self.map[state[0] - 1, state[1]] == 1:  # 说明是地面
                    return np.array((state[0] - 1, state[1]), dtype=int), 0, 0
                elif self.map[state[0] - 1, state[1]] == 2:  # 说明是终点
                    return np.array((-1, -1), dtype=int), 100, 1
            elif action == 1:  # 下
                if self.map[state[0] + 1, state[1]] == 0:
                    return np.array((-1, -1), dtype=int), -1, 1
                elif self.map[state[0] + 1, state[1]] == 1:
                    return np.array((state[0] + 1, state[1]), dtype=int), 0, 0
                elif self.map[state[0] + 1, state[1]] == 2:
                    return np.array((-1, -1), dtype=int), 100, 1
            elif action == 2:  # 左
                if self.map[state[0], state[1] - 1] == 0:
                    return np.array((-1, -1), dtype=int), -1, 1
                elif self.map[state[0], state[1] - 1] == 1:
                    return np.array((state[0], state[1] - 1), dtype=int), 0, 0
                elif self.map[state[0], state[1] - 1] == 2:
                    return np.array((-1, -1), dtype=int), 100, 1
            elif action == 3:  # 右
                if self.map[state[0], state[1] + 1] == 0:
                    return np.array((-1, -1), dtype=int), -1, 1
                elif self.map[state[0], state[1] + 1] == 1:
                    return np.array((state[0], state[1] + 1), dtype=int), 0, 0
                elif self.map[state[0], state[1] + 1] == 2:
                    return np.array((-1, -1), dtype=int), 100, 1


# DQN 测试
# test_model = DQN()
# test_list = [[5, 5], [3, 3], [7, 8]]
# print(test_model.forward(np.array([5, 5], dtype=int)))  # 这需要是个 tensor 错误，无法运行
# print(test_model.act(np.array([5, 5], dtype=int)))
# print(test_model(torch.as_tensor(np.asarray(test_list), dtype=torch.float)))
# exit()

EPSILON_DECAY = 100000
EPSILON_START = 1.0
EPSILON_END = 0.02
TARGET_UPDATE_FREQUENCY = 10

n_episode = 5000
n_step = 1000

maze = Maze()
agent = Agent()

s = maze.reset()  # 初始化起始点
# print(type(s))
# print(s.shape)
# print(agent.online_net.act(s))

REWARD_BUFFER = np.zeros(shape=n_episode)  # 用于存放奖励

###
# DQN 算法部分
###
for episode_i in range(n_episode):
    episode_reward = 0
    for step_i in range(n_step):
        # 首先要计算 epsilon 看是否要随机选择一个 action 还是选择 network 输出的最大值对应的 action
        # np.interp() 是线性插值，三个参数分别对应 x, 横轴范围, 纵轴范围
        epsilon = np.interp(episode_i * n_step + step_i, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])  # 越来越小
        random_sample = random.random()
        if random_sample <= epsilon:  # 越来越不会随机
            a = random.randint(0, 3)  # 0, 1, 2, 3 = 上, 下, 左, 右
        else:
            a = agent.online_net.act(s)

        # 计算 s 状态下采取 a 行动后的下一个状态，奖励，是否结束
        s_, r, done = maze.step(s, a)

        # 存到经验池中
        agent.memory.push(s, a, r, done, s_)
        # if done == 0:
        #     agent.memory.push(s, a, r, done, s_)

        # 状态转移
        s = s_
        episode_reward += r

        if done == 1:  # 说明已经撞墙或者走到终点了
            s = maze.reset()  # 重新选择一个起始点
            REWARD_BUFFER[episode_i] = episode_reward
            break

        ###
        # 重点！这里要从经验池获取数据然后训练网络了
        ###
        # 这里取的时候是数量小于 BATCH_SIZE 时全取了，大于时取 BATCH_SIZE 个
        batch_s, batch_a, batch_r, batch_done, batch_s_ = agent.memory.sample()

        # 计算 target
        target_q_values = agent.target_net(batch_s_)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
        targets = batch_r + agent.GAMMA * (1 - batch_done) * max_target_q_values

        # 计算 q_values
        q_values = agent.online_net(batch_s)
        a_q_value = torch.gather(input=q_values, dim=1, index=batch_a)

        # 计算 loss
        loss = F.smooth_l1_loss(targets, a_q_value)

        # 梯度下降
        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()

        if step_i % TARGET_UPDATE_FREQUENCY == 0:  # 每 10 步更新一次
            agent.target_net.load_state_dict(agent.online_net.state_dict())

    if episode_i % TARGET_UPDATE_FREQUENCY == 0:  # 每 10 局看看效果
        print("Episode: {}".format(episode_i))
        # print("Reward: {}".format(REWARD_BUFFER[episode_i]))
        print("Avg Reward: {}".format(np.mean(REWARD_BUFFER[:episode_i])))

# 计算完网络后打印建议方向
agent.online_net.suggest(maze.player)

while True:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
            exit()
        elif event.type == pg.KEYDOWN:
            if event.key == pg.K_UP:
                # print("Press UP")
                if maze.player[0] - 1 == maze.target[0] and maze.player[1] == maze.target[1]:  # 到了终点退出
                    pg.quit()
                    exit()
                if maze.map[maze.player[0] - 1, maze.player[1]] == 1:  # 不是墙
                    pg.draw.rect(maze.screen, "white",
                                 (maze.player[1] * maze.BLOCK_PIXEL, maze.player[0] * maze.BLOCK_PIXEL,
                                  maze.BLOCK_PIXEL, maze.BLOCK_PIXEL))
                    maze.player[0] -= 1
                    pg.draw.rect(maze.screen, "black",
                                 (maze.player[1] * maze.BLOCK_PIXEL, maze.player[0] * maze.BLOCK_PIXEL,
                                  maze.BLOCK_PIXEL, maze.BLOCK_PIXEL))
                    pg.display.flip()  # 刷新
            if event.key == pg.K_DOWN:
                # print("Press DOWN")
                if maze.player[0] + 1 == maze.target[0] and maze.player[1] == maze.target[1]:  # 到了终点退出
                    pg.quit()
                    exit()
                if maze.map[maze.player[0] + 1, maze.player[1]] == 1:  # 不是墙
                    pg.draw.rect(maze.screen, "white",
                                 (maze.player[1] * maze.BLOCK_PIXEL, maze.player[0] * maze.BLOCK_PIXEL,
                                  maze.BLOCK_PIXEL, maze.BLOCK_PIXEL))
                    maze.player[0] += 1
                    pg.draw.rect(maze.screen, "black",
                                 (maze.player[1] * maze.BLOCK_PIXEL, maze.player[0] * maze.BLOCK_PIXEL,
                                  maze.BLOCK_PIXEL, maze.BLOCK_PIXEL))
                    pg.display.flip()  # 刷新
            if event.key == pg.K_LEFT:
                # print("Press LEFT")
                if maze.player[0] == maze.target[0] and maze.player[1] - 1 == maze.target[1]:  # 到了终点退出
                    pg.quit()
                    exit()
                if maze.map[maze.player[0], maze.player[1] - 1] == 1:  # 不是墙
                    pg.draw.rect(maze.screen, "white",
                                 (maze.player[1] * maze.BLOCK_PIXEL, maze.player[0] * maze.BLOCK_PIXEL,
                                  maze.BLOCK_PIXEL, maze.BLOCK_PIXEL))
                    maze.player[1] -= 1
                    pg.draw.rect(maze.screen, "black",
                                 (maze.player[1] * maze.BLOCK_PIXEL, maze.player[0] * maze.BLOCK_PIXEL,
                                  maze.BLOCK_PIXEL, maze.BLOCK_PIXEL))
                    pg.display.flip()  # 刷新
            if event.key == pg.K_RIGHT:
                # print("Press RIGHT")
                if maze.player[0] == maze.target[0] and maze.player[1] + 1 == maze.target[1]:  # 到了终点退出
                    pg.quit()
                    exit()
                if maze.map[maze.player[0], maze.player[1] + 1] == 1:  # 不是墙
                    pg.draw.rect(maze.screen, "white",
                                 (maze.player[1] * maze.BLOCK_PIXEL, maze.player[0] * maze.BLOCK_PIXEL,
                                  maze.BLOCK_PIXEL, maze.BLOCK_PIXEL))
                    maze.player[1] += 1
                    pg.draw.rect(maze.screen, "black",
                                 (maze.player[1] * maze.BLOCK_PIXEL, maze.player[0] * maze.BLOCK_PIXEL,
                                  maze.BLOCK_PIXEL, maze.BLOCK_PIXEL))
                    pg.display.flip()  # 刷新
            agent.online_net.suggest(maze.player)
