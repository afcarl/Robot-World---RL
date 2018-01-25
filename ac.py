import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np 
from torch.distributions import Categorical 
from collections import namedtuple
import robotWorld as World

Memory = namedtuple('Memory',  ['log_prob', 'estim'])

class AC(nn.Module): 

	def __init__(self, env_infos, hidden = 64 ): 
		nn.Module.__init__(self)
		self.l1 = nn.Linear(env_infos[0],hidden)
		self.policy_head = nn.Linear(hidden, env_infos[1])
		self.action_head = nn.Linear(hidden, 1)

	def forward(self, state_tensor): 

		x = F.relu(self.l1(state_tensor))
		policy = F.softmax(self.policy_head(x))
		value = self.action_head(x)

		return policy, value

class Agent():

	def __init__(self,env_infos, hidden): 
		self.brain = AC(env_infos, hidden= hidden)
		self.adam = optim.Adam(self.brain.parameters(), 5e-3)

		self.memory = []
		self.update = [1,5]

	def think(self, state_tensor): 
		probs, val = self.brain(state_tensor)
		m = Categorical(probs)
		action = m.sample()
		self.memory.append(Memory(m.log_prob(action), val))
		return action.data[0]

	def discountReward(self, r, gamma = 0.99): 
		current = 0
		result = []
		for i in reversed(range(len(r))): 
			current = current*gamma + r[i]
			result.insert(0,current)
		return result

	def train(self, reward): 
		reward = self.discountReward(reward)
		policy_loss, value_loss = [],[]

		reward = torch.Tensor(reward)
		#reward = (reward - reward.mean())/reward.std()

		for (log_prob, value), r in zip(self.memory, reward): 
			r_t = r - value.data[0,0]
			policy_loss.append(-log_prob*r_t)
			value_loss.append(F.smooth_l1_loss(value, Variable(torch.Tensor([r]))))
		
		loss = torch.cat(policy_loss).sum() + torch.cat(value_loss).sum()
		loss.backward()
		del self.memory[:]
		self.apply()


	def apply(self): 
		self.update[0] = (self.update[0] + 1)%self.update[1]
		if self.update[0] == 0: 
			self.adam.step()
			self.adam.zero_grad()
		

env = World.World(robotJoints = 2, robotJointsLength = 0.35,
	randomizeRobot = False, randomizeTarget = False,
	 groundHeight =0.05, targetLimits = [0.05,0.95,0.1,0.65])

env.setTargetPosition([[0.3,0.6]])

env_infos = [6,env.actionSpaceSize()]
hidden = 64
# import gym 
# env = gym.make('Acrobot-v1')
# env_infos = [6,3]

player = Agent(env_infos, hidden)
#player = torch.load('bl.new')

epochs = 1500
info, mean_reward, successes = 100,0,0 
successives_actions = 3
success_hist = []


for epoch in range(epochs): 
	s = env.reset()
	done = False
	steps, reward = 0,0
	r_hist = []

	while not done: 

		if steps%successives_actions == 0:
			state_tensor = Variable(torch.Tensor(s)).unsqueeze(0)
			action = player.think(state_tensor)

		#state_tensor = Variable(torch.Tensor(s.tolist())).unsqueeze(0)
		#action = player.think(state_tensor)


		ns, r, done, success = env.step(action)
		if steps%successives_actions == 0:
			r_hist.append(r)
		s = ns 
		reward += r 
		steps += 1

		if done: 
			mean_reward += 1.*reward/info
			player.train(r_hist)
			successes += success
			if epoch%info == 0: 
				print('It {}/{} - Success {} -- Mean reward {:.2f} '.format(epoch, epochs, successes, mean_reward))
				success_hist.append(successes)
				mean_reward, successes = 0,0
				#torch.save(player, 'bl.plus')

import matplotlib.pyplot as plt 
if len(success_hist) > 10:
	plt.style.use('ggplot')
	plt.plot(np.arange(len(success_hist)),success_hist)
	plt.xlabel('Epochs')
	plt.ylabel('Success')
	plt.ylim(0,101)
	plt.title('2 joints robot with one fixed target')
	plt.pause(0.1)
	raw_input()

env.initRender()
s = env.reset()
steps = 0 
while True: 

	if steps%successives_actions == 0:
		action = player.think(Variable(torch.Tensor(s)).unsqueeze(0))
	n, r, done, _ = env.step(action)
	s = n 
	env.render()
	steps += 1
	if done: 
		s = env.reset()