# 강화 학습 (Reinforcement learning) 3

> **Migrated from Quartz wiki** (`wiki/study/rokey-ai/intermediate/ai-app-14-dqn.md`, updated 2026-09-26)  
> tags: rokey, ai, notebook, intermediate, study  
> source: raw/2026-09-26-rokey-ai/intermediate/강의_3기_AI응용_14차시__DQN_.ipynb  
> source: https://github.com/weedmo/Rokey-AI  


```python
import random 
import math 

import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn, optim
import torch.nn.functional as F
from collections import deque 

import gymnasium as gym 
```

## Vanilla DQN : Carte-Pole


```python
# 하이퍼파라미터
eposides = 100    
e_start = 0.9  
e_end = 0.05  
e_decay = 200  
gamma = 0.8      
lr = 0.001       
batch_size = 64  
```


```python
class DQN_Cart(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

        self.optimizer = optim.Adam(self.model.parameters(), lr = lr)
        self.step = 0
        self.deque = deque(maxlen=1000)


    def memorize(self, state, action, reward, next_state):
        self.deque.append((state, 
                            action, 
                            torch.FloatTensor([reward]), 
                            torch.FloatTensor([next_state])))
        
    def action(self, state):
        eps_theta = e_end + (e_start - e_end)*np.exp(-(self.step/e_decay))
        self.step += 1
        if np.random.rand() < eps_theta:
            return torch.LongTensor([[np.random.choice([0, 1])]])
        else:
            return self.model(state).data.argmax().view(1, 1) # tensor([[-0.0653, -0.0613]]) tensor([[1]])

    def learning(self):
        if len(self.deque) < batch_size*4:
            return
        batch = random.sample(self.deque, batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.cat(states)
        actions = torch.cat(actions)
        rewards = torch.cat(rewards)
        next_states = torch.cat(next_states)

        # tensor([[ 0.1863, -0.0853],
        # [ 0.1863, -0.0853]], grad_fn=<AddmmBackward0>)
        current_q = self.model(states).gather(1, actions) 
        max_next_q = self.model(next_states).detach().max(1)[0] # tensor([0.1863])
        expected_q = rewards + (gamma * max_next_q)
        
        loss = F.mse_loss(current_q.squeeze(), expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

```


```python
env = gym.make('CartPole-v1') 
agent = DQN_Cart()
scores = []
```


```python
for i in range(1, eposides+1):
    state, _ = env.reset() 
    steps = 0
    while True:
     
        state = torch.FloatTensor([state])
        action = agent.action(state)
        next_state, reward, terminated, truncated, _ = env.step(action.item())

        if terminated:
            reward = -1

        agent.memorize(state, action, reward, next_state)
        agent.learning()

        state = next_state
        steps += 1

        if terminated:
            print("에피소드:{0} 점수: {1}".format(i, steps))
            scores.append(steps)
            break
```

    C:\Users\user\AppData\Local\Temp\ipykernel_34700\204308576.py:6: UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\pytorch\torch\csrc\utils\tensor_new.cpp:257.)
      state = torch.FloatTensor([state])


    에피소드:1 점수: 24
    에피소드:2 점수: 24
    에피소드:3 점수: 15
    에피소드:4 점수: 10
    에피소드:5 점수: 11
    에피소드:6 점수: 12
    에피소드:7 점수: 10
    에피소드:8 점수: 15
    에피소드:9 점수: 12
    에피소드:10 점수: 13
    에피소드:11 점수: 14
    에피소드:12 점수: 16
    에피소드:13 점수: 12
    에피소드:14 점수: 15
    에피소드:15 점수: 11
    에피소드:16 점수: 15
    에피소드:17 점수: 13
    에피소드:18 점수: 9
    에피소드:19 점수: 12
    에피소드:20 점수: 13
    에피소드:21 점수: 10
    에피소드:22 점수: 9
    에피소드:23 점수: 9
    에피소드:24 점수: 11
    에피소드:25 점수: 10
    에피소드:26 점수: 14
    에피소드:27 점수: 9
    에피소드:28 점수: 10
    에피소드:29 점수: 16
    에피소드:30 점수: 10
    에피소드:31 점수: 13
    에피소드:32 점수: 11
    에피소드:33 점수: 13
    에피소드:34 점수: 13
    에피소드:35 점수: 8
    에피소드:36 점수: 10
    에피소드:37 점수: 9
    에피소드:38 점수: 9
    에피소드:39 점수: 16
    에피소드:40 점수: 19
    에피소드:41 점수: 10
    에피소드:42 점수: 77
    에피소드:43 점수: 56
    에피소드:44 점수: 72
    에피소드:45 점수: 274
    에피소드:46 점수: 213
    에피소드:47 점수: 234
    에피소드:48 점수: 264
    에피소드:49 점수: 310
    에피소드:50 점수: 425
    에피소드:51 점수: 473
    에피소드:52 점수: 106
    에피소드:53 점수: 457
    에피소드:54 점수: 421
    에피소드:55 점수: 228
    에피소드:56 점수: 328
    에피소드:57 점수: 284
    에피소드:58 점수: 292
    에피소드:59 점수: 452
    에피소드:60 점수: 790
    에피소드:61 점수: 7682
    에피소드:62 점수: 39
    에피소드:63 점수: 19
    에피소드:64 점수: 48
    에피소드:65 점수: 12
    에피소드:66 점수: 19
    에피소드:67 점수: 34
    에피소드:68 점수: 157
    에피소드:69 점수: 13
    에피소드:70 점수: 208
    에피소드:71 점수: 62
    에피소드:72 점수: 426
    에피소드:73 점수: 38
    에피소드:74 점수: 411
    에피소드:75 점수: 195
    에피소드:76 점수: 405
    에피소드:77 점수: 224
    에피소드:78 점수: 283
    에피소드:79 점수: 309
    에피소드:80 점수: 233
    에피소드:81 점수: 143
    에피소드:82 점수: 20
    에피소드:83 점수: 162
    에피소드:84 점수: 143
    에피소드:85 점수: 140
    에피소드:86 점수: 188
    에피소드:87 점수: 175
    에피소드:88 점수: 163
    에피소드:89 점수: 199
    에피소드:90 점수: 207
    에피소드:91 점수: 203
    에피소드:92 점수: 198
    에피소드:93 점수: 168
    에피소드:94 점수: 177
    에피소드:95 점수: 192
    에피소드:96 점수: 187
    에피소드:97 점수: 181
    에피소드:98 점수: 156
    에피소드:99 점수: 167
    에피소드:100 점수: 161



```python
plt.plot(scores)
plt.ylabel('score')
plt.show()
```


    
*(이미지 없음 / image not preserved: output_7_0.png)*
    


## DQN with Target Network: Carte-Pole


```python
import gymnasium as gym
import matplotlib.pyplot as plt
import math
import random
import time
import torch.nn as nn
import torch.optim as optim
import torch
import matplotlib
import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# env_name = 'MountainCar-v0' 
env_name = 'CartPole-v1'

env = gym.make(env_name)

# 하이퍼파라미터 설정
num_episodes = 300
GAMMA = 0.99  # 감마 (discount factor)
learning_rate = 0.001  # 학습률
hidden_layer = 120  # 은닉층 노드 수
replay_memory_size = 50_000  # 리플레이 메모리 크기
batch_size = 128  # 배치 크기

e_start = 0.9  # 입실론 초기값
e_end = 0.05  # 입실론 최종값
e_decay = 200  # 입실론 감소율

target_nn_update_frequency = 10  # 타겟 네트워크 업데이트 주기
clip_error = False  # 오차 클리핑 여부

device = "cpu"

n_inputs = env.observation_space.shape[0]  # 입력 차원 수 (상태 수)
n_outputs = env.action_space.n  # 출력 차원 수 (액션 수)

# 리플레이 메모리 클래스
class ExperienceReplay:
    def __init__(self, capacity):
        self.capacity = capacity  # 리플레이 메모리의 최대 크기 설정
        self.memory = []  # 경험을 저장할 메모리 리스트 초기화
        self.position = 0  # 현재 저장 위치 초기화

    # 경험 추가 함수
    def push(self, state, action, new_state, reward, done):
        # 주어진 경험(transition)을 메모리에 추가
        transition = (state, action, new_state, reward, done)

        if self.position >= len(self.memory):
            # 메모리에 빈 공간이 있으면 경험 추가
            self.memory.append(transition)
        else:
            # 메모리가 가득 차면 오래된 경험을 덮어쓰기
            self.memory[self.position] = transition
            
        # 저장 위치를 다음으로 이동, 용량을 초과하면 처음으로 돌아감
        self.position = (self.position + 1) % self.capacity

    # 경험 샘플링 함수
    def sample(self, batch_size):
        # 메모리에서 주어진 배치 크기만큼 무작위로 샘플링하여 반환
        return zip(*random.sample(self.memory, batch_size))

    def __len__(self):
        # 현재 메모리에 저장된 경험의 수를 반환
        return len(self.memory)

# 신경망 클래스
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(n_inputs, hidden_layer)
        self.linear2 = nn.Linear(hidden_layer, hidden_layer//2)
        self.linear3 = nn.Linear(hidden_layer//2, n_outputs)

    # 순전파 함수
    def forward(self, x):
        a1 = torch.relu(self.linear1(x))
        a2 = torch.relu(self.linear2(a1))
        output = self.linear3(a2)
        return output

# 액션 선택 함수
def select_action(state, steps_done):
    # 입실론 값 계산
    e_threshold = e_end + (e_start - e_end) * \
        math.exp(-1. * steps_done/e_decay)

    if random.random() > e_threshold:
        # 입실론보다 큰 경우, Q 함수에 따라 행동 선택
        with torch.no_grad():
            state = torch.Tensor(state).to(device)   # 상태를 텐서로 변환하고 장치에 할당
            action_values = Q(state)   # Q 함수를 사용하여 각 행동의 가치 계산
            action = torch.argmax(action_values).item()   # 가장 높은 가치를 갖는 행동 선택
    else: 
        # 입실론보다 작은 경우, 무작위 행동 선택 (탐색)
        action = env.action_space.sample()

    return action

# 리플레이 메모리 초기화
memory = ExperienceReplay(replay_memory_size)

# 타겟 Q함수 초기화 (랜덤 가중치)
target_Q = NeuralNetwork().to(device)

# Q함수 초기화 (랜덤 가중치로 신경망 생성)
Q = NeuralNetwork().to(device)

# 손실 함수 설정 (평균 제곱 오차)
criterion = nn.MSELoss()

# 최적화 알고리즘 설정 (Adam 옵티마이저)
optimizer = optim.Adam(Q.parameters(), lr=learning_rate)

# 타겟 네트워크 업데이트 카운터 초기화
update_target_counter = 0
# 각 에피소드에서 얻은 보상을 저장할 리스트 초기화
reward_history = []
# 총 스텝 수 초기화
total_steps = 0
# 학습 시작 시간 기록
start_time = time.time()

# 에피소드 루프
for episode in range(num_episodes):
    if episode > num_episodes * 0.98:
        env = gym.make(env_name, render_mode="human")
    else:
        env = gym.make(env_name)

    s, _ = env.reset()
    reward = 0
    while True:
        total_steps += 1

        # 액션 선택
        a = select_action(s, total_steps)

        # 환경에서 액션 수행
        s_, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        reward += r

        # 리플레이 메모리에 경험 저장
        memory.push(s, a, s_, r, done)

        if len(memory) >= batch_size:
            # 리플레이 메모리에서 미니배치 샘플링
            states, actions, new_states, rewards, dones = memory.sample(
                batch_size)

            # 샘플링한 데이터를 텐서로 변환하여 장치에 할당
            states = torch.Tensor(states).to(device)
            actions = torch.LongTensor(actions).to(device)
            new_states = torch.Tensor(new_states).to(device)
            rewards = torch.Tensor([rewards]).to(device)
            dones = torch.Tensor(dones).to(device)
            
            # 타겟 Q 네트워크로부터 새로운 상태의 Q 값 계산
            new_action_values = target_Q(new_states).detach()

            # 타겟 값 계산
            y_target = rewards + \
                (1 - dones) * GAMMA * torch.max(new_action_values, 1)[0]
            # 예측 값 계산
            y_pred_pre = Q(states)
            y_pred = y_pred_pre.gather(1, actions.unsqueeze(1))
            
            # 손실 계산 및 역전파
            loss = criterion(y_pred.squeeze(), y_target.squeeze())
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            # 타겟 네트워크 업데이트
            if update_target_counter % target_nn_update_frequency == 0:
                target_Q.load_state_dict(Q.state_dict())

            update_target_counter += 1

        s = s_

        if done:
            reward_history.append(reward)
            print(f"{episode} episode finished after {reward:.2f} rewards")
            break

# 평균 보상 출력
print("Average rewards: %.2f" % (sum(reward_history)/num_episodes))

# 마지막 50 에피소드의 평균 보상 출력
print("Average of last 100 episodes: %.2f" % (sum(reward_history[-50:])/50))

# 하이퍼파라미터 정보 출력
print("---------------------- Hyper parameters --------------------------------------")
print(
    f"GAMMA:{GAMMA}, learning rate: {learning_rate}, hidden layer: {hidden_layer}")
print(f"replay_memory: {replay_memory_size}, batch size: {batch_size}")
print(f"epsilon_start: {e_start}, epsilon_end: {e_end}, " +
      f"epsilon_decay: {e_decay}")
print(
    f"update frequency: {target_nn_update_frequency}, clipping: {clip_error}")

# 경과 시간 출력
elapsed_time = time.time() - start_time
print(f"Time Elapsed : {elapsed_time//60} min {elapsed_time%60:.0} sec")

# 학습 과정의 보상 플롯
plt.bar(torch.arange(len(reward_history)).numpy(), reward_history)
plt.xlabel("episodes")
plt.ylabel("rewards")
plt.title("DQN - Target Network")
plt.show()

```

    0 episode finished after 38.00 rewards
    1 episode finished after 10.00 rewards
    2 episode finished after 10.00 rewards
    3 episode finished after 60.00 rewards
    4 episode finished after 19.00 rewards


    C:\Users\user\AppData\Local\Temp\ipykernel_30392\3248349219.py:155: UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\pytorch\torch\csrc\utils\tensor_new.cpp:257.)
      states = torch.Tensor(states).to(device)


    5 episode finished after 34.00 rewards
    6 episode finished after 10.00 rewards
    7 episode finished after 9.00 rewards
    8 episode finished after 8.00 rewards
    9 episode finished after 12.00 rewards
    10 episode finished after 13.00 rewards
    11 episode finished after 10.00 rewards
    12 episode finished after 13.00 rewards
    13 episode finished after 13.00 rewards
    14 episode finished after 12.00 rewards
    15 episode finished after 11.00 rewards
    16 episode finished after 11.00 rewards
    17 episode finished after 13.00 rewards
    18 episode finished after 12.00 rewards
    19 episode finished after 11.00 rewards
    20 episode finished after 10.00 rewards
    21 episode finished after 15.00 rewards
    22 episode finished after 35.00 rewards
    23 episode finished after 23.00 rewards
    24 episode finished after 34.00 rewards
    25 episode finished after 42.00 rewards
    26 episode finished after 57.00 rewards
    27 episode finished after 42.00 rewards
    28 episode finished after 31.00 rewards
    29 episode finished after 65.00 rewards
    30 episode finished after 57.00 rewards
    31 episode finished after 196.00 rewards
    32 episode finished after 213.00 rewards
    33 episode finished after 299.00 rewards
    34 episode finished after 198.00 rewards
    35 episode finished after 193.00 rewards
    36 episode finished after 183.00 rewards
    37 episode finished after 221.00 rewards
    38 episode finished after 173.00 rewards
    39 episode finished after 170.00 rewards
    40 episode finished after 219.00 rewards
    41 episode finished after 174.00 rewards
    42 episode finished after 208.00 rewards
    43 episode finished after 221.00 rewards
    44 episode finished after 186.00 rewards
    45 episode finished after 212.00 rewards
    46 episode finished after 232.00 rewards
    47 episode finished after 198.00 rewards
    48 episode finished after 213.00 rewards
    49 episode finished after 189.00 rewards
    50 episode finished after 228.00 rewards
    51 episode finished after 190.00 rewards
    52 episode finished after 199.00 rewards
    53 episode finished after 209.00 rewards
    54 episode finished after 290.00 rewards
    55 episode finished after 212.00 rewards
    56 episode finished after 150.00 rewards
    57 episode finished after 252.00 rewards
    58 episode finished after 116.00 rewards
    59 episode finished after 114.00 rewards
    60 episode finished after 165.00 rewards
    61 episode finished after 136.00 rewards
    62 episode finished after 189.00 rewards
    63 episode finished after 223.00 rewards
    64 episode finished after 186.00 rewards
    65 episode finished after 169.00 rewards
    66 episode finished after 132.00 rewards
    67 episode finished after 241.00 rewards
    68 episode finished after 157.00 rewards
    69 episode finished after 263.00 rewards
    70 episode finished after 338.00 rewards
    71 episode finished after 205.00 rewards
    72 episode finished after 158.00 rewards
    73 episode finished after 229.00 rewards
    74 episode finished after 173.00 rewards
    75 episode finished after 193.00 rewards
    76 episode finished after 178.00 rewards
    77 episode finished after 298.00 rewards
    78 episode finished after 234.00 rewards
    79 episode finished after 178.00 rewards
    80 episode finished after 189.00 rewards
    81 episode finished after 202.00 rewards
    82 episode finished after 206.00 rewards
    83 episode finished after 207.00 rewards
    84 episode finished after 185.00 rewards
    85 episode finished after 175.00 rewards
    86 episode finished after 173.00 rewards
    87 episode finished after 200.00 rewards
    88 episode finished after 178.00 rewards
    89 episode finished after 180.00 rewards
    90 episode finished after 188.00 rewards
    91 episode finished after 180.00 rewards
    92 episode finished after 199.00 rewards
    93 episode finished after 246.00 rewards
    94 episode finished after 220.00 rewards
    95 episode finished after 210.00 rewards
    96 episode finished after 214.00 rewards
    97 episode finished after 212.00 rewards
    98 episode finished after 500.00 rewards
    99 episode finished after 292.00 rewards
    100 episode finished after 163.00 rewards
    101 episode finished after 153.00 rewards
    102 episode finished after 172.00 rewards
    103 episode finished after 243.00 rewards
    104 episode finished after 188.00 rewards
    105 episode finished after 213.00 rewards
    106 episode finished after 292.00 rewards
    107 episode finished after 273.00 rewards
    108 episode finished after 500.00 rewards
    109 episode finished after 339.00 rewards
    110 episode finished after 500.00 rewards
    111 episode finished after 307.00 rewards
    112 episode finished after 314.00 rewards
    113 episode finished after 184.00 rewards
    114 episode finished after 294.00 rewards
    115 episode finished after 189.00 rewards
    116 episode finished after 210.00 rewards
    117 episode finished after 221.00 rewards
    118 episode finished after 171.00 rewards
    119 episode finished after 219.00 rewards
    120 episode finished after 195.00 rewards
    121 episode finished after 226.00 rewards
    122 episode finished after 314.00 rewards
    123 episode finished after 210.00 rewards
    124 episode finished after 500.00 rewards
    125 episode finished after 250.00 rewards
    126 episode finished after 296.00 rewards
    127 episode finished after 185.00 rewards
    128 episode finished after 196.00 rewards
    129 episode finished after 153.00 rewards
    130 episode finished after 220.00 rewards
    131 episode finished after 500.00 rewards
    132 episode finished after 500.00 rewards
    133 episode finished after 500.00 rewards
    134 episode finished after 193.00 rewards
    135 episode finished after 500.00 rewards
    136 episode finished after 240.00 rewards
    137 episode finished after 347.00 rewards
    138 episode finished after 500.00 rewards
    139 episode finished after 500.00 rewards
    140 episode finished after 500.00 rewards
    141 episode finished after 173.00 rewards
    142 episode finished after 309.00 rewards
    143 episode finished after 500.00 rewards
    144 episode finished after 500.00 rewards
    145 episode finished after 500.00 rewards
    146 episode finished after 500.00 rewards
    147 episode finished after 500.00 rewards
    148 episode finished after 500.00 rewards
    149 episode finished after 500.00 rewards
    150 episode finished after 500.00 rewards
    151 episode finished after 500.00 rewards
    152 episode finished after 500.00 rewards
    153 episode finished after 500.00 rewards
    154 episode finished after 500.00 rewards
    155 episode finished after 461.00 rewards
    156 episode finished after 109.00 rewards
    157 episode finished after 187.00 rewards
    158 episode finished after 218.00 rewards
    159 episode finished after 314.00 rewards
    160 episode finished after 500.00 rewards
    161 episode finished after 441.00 rewards
    162 episode finished after 244.00 rewards
    163 episode finished after 197.00 rewards
    164 episode finished after 195.00 rewards
    165 episode finished after 500.00 rewards
    166 episode finished after 500.00 rewards
    167 episode finished after 500.00 rewards
    168 episode finished after 500.00 rewards
    169 episode finished after 500.00 rewards
    170 episode finished after 337.00 rewards
    171 episode finished after 500.00 rewards
    172 episode finished after 500.00 rewards
    173 episode finished after 500.00 rewards
    174 episode finished after 500.00 rewards
    175 episode finished after 500.00 rewards
    176 episode finished after 500.00 rewards
    177 episode finished after 444.00 rewards
    178 episode finished after 326.00 rewards
    179 episode finished after 242.00 rewards
    180 episode finished after 245.00 rewards
    181 episode finished after 270.00 rewards
    182 episode finished after 213.00 rewards
    183 episode finished after 405.00 rewards
    184 episode finished after 204.00 rewards
    185 episode finished after 34.00 rewards
    186 episode finished after 211.00 rewards
    187 episode finished after 258.00 rewards
    188 episode finished after 332.00 rewards
    189 episode finished after 500.00 rewards
    190 episode finished after 347.00 rewards
    191 episode finished after 500.00 rewards
    192 episode finished after 301.00 rewards
    193 episode finished after 500.00 rewards
    194 episode finished after 500.00 rewards
    195 episode finished after 374.00 rewards
    196 episode finished after 301.00 rewards
    197 episode finished after 397.00 rewards
    198 episode finished after 309.00 rewards
    199 episode finished after 234.00 rewards
    200 episode finished after 500.00 rewards
    201 episode finished after 395.00 rewards
    202 episode finished after 388.00 rewards
    203 episode finished after 196.00 rewards
    204 episode finished after 500.00 rewards
    205 episode finished after 500.00 rewards
    206 episode finished after 500.00 rewards
    207 episode finished after 500.00 rewards
    208 episode finished after 500.00 rewards
    209 episode finished after 306.00 rewards
    210 episode finished after 289.00 rewards
    211 episode finished after 500.00 rewards
    212 episode finished after 500.00 rewards
    213 episode finished after 500.00 rewards
    214 episode finished after 461.00 rewards
    215 episode finished after 500.00 rewards
    216 episode finished after 500.00 rewards
    217 episode finished after 500.00 rewards
    218 episode finished after 500.00 rewards
    219 episode finished after 500.00 rewards
    220 episode finished after 500.00 rewards
    221 episode finished after 500.00 rewards
    222 episode finished after 500.00 rewards
    223 episode finished after 368.00 rewards
    224 episode finished after 500.00 rewards
    225 episode finished after 500.00 rewards
    226 episode finished after 500.00 rewards
    227 episode finished after 500.00 rewards
    228 episode finished after 500.00 rewards
    229 episode finished after 500.00 rewards
    230 episode finished after 500.00 rewards
    231 episode finished after 469.00 rewards
    232 episode finished after 241.00 rewards
    233 episode finished after 500.00 rewards
    234 episode finished after 500.00 rewards
    235 episode finished after 500.00 rewards
    236 episode finished after 500.00 rewards
    237 episode finished after 391.00 rewards
    238 episode finished after 397.00 rewards
    239 episode finished after 500.00 rewards
    240 episode finished after 470.00 rewards
    241 episode finished after 94.00 rewards
    242 episode finished after 45.00 rewards
    243 episode finished after 92.00 rewards
    244 episode finished after 148.00 rewards
    245 episode finished after 500.00 rewards
    246 episode finished after 500.00 rewards
    247 episode finished after 500.00 rewards
    248 episode finished after 500.00 rewards
    249 episode finished after 500.00 rewards
    250 episode finished after 275.00 rewards
    251 episode finished after 143.00 rewards
    252 episode finished after 500.00 rewards
    253 episode finished after 500.00 rewards
    254 episode finished after 399.00 rewards
    255 episode finished after 238.00 rewards
    256 episode finished after 188.00 rewards
    257 episode finished after 377.00 rewards
    258 episode finished after 500.00 rewards
    259 episode finished after 500.00 rewards
    260 episode finished after 500.00 rewards
    261 episode finished after 500.00 rewards
    262 episode finished after 500.00 rewards
    263 episode finished after 500.00 rewards
    264 episode finished after 500.00 rewards
    265 episode finished after 212.00 rewards
    266 episode finished after 500.00 rewards
    267 episode finished after 500.00 rewards
    268 episode finished after 444.00 rewards
    269 episode finished after 500.00 rewards
    270 episode finished after 500.00 rewards
    271 episode finished after 500.00 rewards
    272 episode finished after 500.00 rewards
    273 episode finished after 500.00 rewards
    274 episode finished after 500.00 rewards
    275 episode finished after 500.00 rewards
    276 episode finished after 264.00 rewards
    277 episode finished after 185.00 rewards
    278 episode finished after 500.00 rewards
    279 episode finished after 500.00 rewards
    280 episode finished after 500.00 rewards
    281 episode finished after 500.00 rewards
    282 episode finished after 500.00 rewards
    283 episode finished after 500.00 rewards
    284 episode finished after 500.00 rewards
    285 episode finished after 500.00 rewards
    286 episode finished after 500.00 rewards
    287 episode finished after 500.00 rewards
    288 episode finished after 500.00 rewards
    289 episode finished after 500.00 rewards
    290 episode finished after 500.00 rewards
    291 episode finished after 500.00 rewards
    292 episode finished after 500.00 rewards
    293 episode finished after 500.00 rewards
    294 episode finished after 500.00 rewards
    295 episode finished after 500.00 rewards
    296 episode finished after 279.00 rewards
    297 episode finished after 500.00 rewards
    298 episode finished after 500.00 rewards
    299 episode finished after 500.00 rewards
    Average rewards: 313.52
    Average of last 100 episodes: 450.08
    ---------------------- Hyper parameters --------------------------------------
    GAMMA:0.99, learning rate: 0.001, hidden layer: 120
    replay_memory: 50000, batch size: 128
    epsilon_start: 0.9, epsilon_end: 0.05, epsilon_decay: 200
    update frequency: 10, clipping: False
    Time Elapsed : 5.0 min 3e+01 sec



    
*(이미지 없음 / image not preserved: output_9_3.png)*
    

