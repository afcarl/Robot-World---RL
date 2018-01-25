import numpy as np 
import pygame as pg 


class World():

	def __init__(self, robotJoints = 4, robotJointsLength = 0.18, 
				randomizeRobot = False, randomizeTarget = False, 
				groundHeight = 0.05, targetLimits = [0.2,0.8,0.2,0.8], maxSteps = 200):

		self.randomizeRobot = randomizeRobot
		self.randomizeTarget = randomizeTarget
		self.baseHeight = groundHeight
		self.targetLimits = targetLimits
		self.maxSteps = maxSteps

		self.ground = Ground(groundHeight)
		self.robot = Robot(robotJoints,robotJointsLength, baseHeight = groundHeight, randomize = randomizeRobot)
		self.target = Target(self.targetLimits)

		self.listed_positions = False
		self.robotParameters = [robotJoints, robotJointsLength]

		self.steps = 0

		self.currentDistance = (self.target.position - self.robot.points[-1])**2
		self.currentDistance = np.sqrt(self.currentDistance)

	def setTargetPosition(self, pos): 
		#self.target.position = pos
		self.target_positions = pos
		self.target.position = np.array(pos[0])
		self.target_position_iterator = 0
		self.listed_positions = True

	def initRender(self, size = [700,700]):

		pg.init()
		self.screen = pg.display.set_mode(size)
		self.clock = pg.time.Clock()
		self.size = size

	def render(self): 

		time = 30
		self.clock.tick(time)
		self.screen.fill((0,0,0))
		self.draw(self.screen, self.size)

		pg.display.flip() 

	def draw(self,screen, screenSize): 

		self.ground.draw(screen, screenSize)
		self.robot.draw(screen, screenSize)
		self.target.draw(screen, screenSize)

	def observe(self): 

		# ------------------------------------------------------------------
		# State is distance between the effector and the ball in the following form -> [dX_Positive, dX_Negative, dY_Pos, dY_Neg]
		# ------------------------------------------------------------------

		targetPosition = self.target.position
		effectorPosition = self.robot.points[-1]

		#print('effectorPosition: {} '.format(effectorPosition))
		#print('targetPosition: {} '.format(targetPosition))

		vector = targetPosition - effectorPosition
		distance = np.sqrt(np.sum(vector**2))

		#print('distance: {} '.format(distance))
		
		state = []
		pos = self.robot.joints_positions()
		for p in pos: 
			for e in p:
				state.append(e)

		for p in targetPosition: 
			state.append(p)

		#state = [targetPosition[0], targetPosition[1], effectorPosition[0], effectorPosition[1]]

		# ------------------------------------------------------------------
		# ------- Reward -----------
		# ------------------------------------------------------------------
		reward = 0.01/distance
		#reward = 0

		angle_penalty = 0.6*np.abs(self.robot.angles[0]) + 0.1*np.abs(self.robot.angles[1])
		#reward -= 0.002*angle_penalty

		self.currentDistance = distance

		# ------------------------------------------------------------------
		# ------- Completion -------- 
		# ------------------------------------------------------------------
		#reward = -1
		complete = False
		success = 0
		i = 0
		for p in self.robot.points:    # checking whether touching the ground
			if p[1] < self.ground.height: 
				complete = True
				reward = -2

		if distance < 0.03:  # target reached
			complete = True
			success = 1
			reward = 1

		if self.steps > self.maxSteps: 
			complete = True
			reward = -1

		return state, reward, complete, success


	def step(self, action): 

		self.robot.rotate(action)
		self.robot.computePositions()

		self.steps += 1

		return self.observe()
		 

	def randomAction(self): 

		maxActions = self.robot.nbJoints*2
		action = np.random.randint(maxActions)
		return action

	def actionSpaceSize(self): 
		return self.robot.nbJoints*2

	def reset(self): 

		self.steps = 0
		self.robot = Robot(self.robotParameters[0], self.robotParameters[1], baseHeight = self.baseHeight, randomize= self.randomizeRobot)
		if self.randomizeTarget: 
			self.target = Target(self.targetLimits)
		if self.listed_positions: 
			self.target_position_iterator = (self.target_position_iterator+1)%len(self.target_positions)
			self.target.position = np.array(self.target_positions[self.target_position_iterator])


		state,_,__,___ = self.observe()
		return state



class Target(): 

	def __init__(self, limits = [0.2,0.8,0.2,0.7], radius = 15):

		x = np.random.uniform(low = limits[0], high = limits[1])
		y = np.random.uniform(low = limits[2], high = limits[3])

		self.position = np.array([x,y])
		self.radius = radius


	def draw(self, screen, screenSize): 

		pos = self.position.copy()
		pos[1] = screenSize[1]*(1-pos[1])
		pos[0] *= screenSize[0]
		

		pg.draw.circle(screen, (250,0,0), pos.astype(int), self.radius)
		pg.draw.circle(screen, (250,250,250), pos.astype(int), int(self.radius*2/3))
		pg.draw.circle(screen, (250,0,0), pos.astype(int), int(self.radius/3))


class Ground(): 

	def __init__(self, height): 

		self.height = height
		self.color = (150,150,150)

	def draw(self, screen, screenSize, nbL = 10): 

		heightOnScreen = screenSize[1]*(1.-self.height)
		pg.draw.line(screen,self.color, [0, heightOnScreen], [screenSize[0], heightOnScreen])

		inc = screenSize[0]/nbL
		for i in range(nbL): 
			pg.draw.line(screen,self.color, [i*inc, heightOnScreen], [(i+1)*inc, screenSize[1]])
			pg.draw.line(screen,self.color, [i*inc, screenSize[1]], [(i+1)*inc, heightOnScreen])

class Robot(): 

	def __init__(self, nbJoints, LengthJoints, baseHeight = 0.0501, speed = 2, randomize = False): 

		self.nbJoints = nbJoints
		self.uLength = LengthJoints

		self.baseHeight = baseHeight
		self.speed = speed

		self.angles = np.zeros((nbJoints))

		if randomize: 
			self.angles = np.random.randint(low = -80, high = 80, size =(nbJoints))

		self.computePositions()

	def rotate(self, action): 

		jointIndex = action/2
		direction = 1 if action%2 == 0 else -1

		self.angles[int(jointIndex)] += direction*self.speed


	def computePositions(self): 

		self.points = np.zeros((self.nbJoints+1,2))
		for i in range(self.nbJoints+1): 
			if i == 0: 
				self.points[i,:] = np.array([0.5,self.baseHeight])
			else:
				angle = (self.angles[i-1] + 90)%360 
				angle = np.radians(angle)
				self.points[i,:] = self.points[i-1,:] + self.uLength*np.array([np.cos(angle), np.sin(angle)])

	def joints_positions(self): 
		points = []
		for i,p in enumerate(self.points): 
			if i != 0: 
				points.append(p)
		return points 

	def draw(self, screen, screenSize): 

		for j in range(self.nbJoints): 
			
			p0 = self.points[j].copy()
			p1 = self.points[j+1].copy()

			p0[0] = (p0[0]*screenSize[0])
			p0[1] = (screenSize[1]*(1-p0[1]))

			p1[0] = (p1[0]*screenSize[0])
			p1[1] = (screenSize[1]*(1-p1[1]))

			pg.draw.line(screen, (220,150,20), p0.astype(int), p1.astype(int), 5)
			pg.draw.circle(screen, (150,250,250), p0.astype(int), 10)
			pg.draw.circle(screen, (150,250,250), p1.astype(int), 10)


# import PGAgent 
# import QAgent
# import torch.nn as nn 
# import torch
# import torch.optim as optim
# from torch.autograd import Variable

# def discountReward(r, gamma = 0.95): 
	
# 	current = 0 
# 	result = np.zeros_like(r)
# 	for i in reversed(range(r.shape[0])): 
# 		current = current*gamma + r[i]
# 		result[i] = current

# 	return result

# def greed(epoch, e_min = 0.1, e_max = 0.9, e_decay = 10000): 

# 	epsi = e_min + (e_max - e_min)*np.exp(-epoch*1./e_decay)
# 	return epsi


# class LRSetting(): 

# 	def __init__(self, ini = 1e-2, final = 1e-5, size = 5): 
# 		self.ini = ini
# 		self.final = final 
# 		self.buffer = [0 for i in range(size)]

# 		self.current = ini


# 	def add_performance(self, performance_measure): 

# 		self.buffer.insert(0,performance_measure)
# 		self.buffer.pop(len(self.buffer)-1)

# 	def update_LR(self): 

# 		p0 = self.buffer[0]
# 		p1 = self.buffer[-1] # ecrire une fonction ponderee par les coefficients. Les self.buffer les plus proches sont moins importants 
		
# 		if  p0 <= p1: 
# 			self.current -= 0.005
# 			self.current = self.current if self.current > self.final else self.final
# 		else: 
# 			self.current += 0.005
# 			self.current = self.current if self.current < self.ini else self.ini
# 		return self.current

# 	def give_initial_lr(self): 
# 		return self.ini


# env = World(robotJoints = 2, robotJointsLength = 0.35,
# 	randomizeRobot = False, randomizeTarget = False,
# 	 groundHeight =0.05, targetLimits = [0.2,0.8,0.1,0.6])

# helper = LRSetting()

# env.setTargetPosition([[0.2,0.6],[0.8,0.6]])
# epochs = 10000
# info = 100 
# batch_size = 32
# successives_actions = 10

# o_space, a_space = 4,env.actionSpaceSize()
# h1,h2 = 100,50
# model = nn.Sequential(nn.Linear(o_space,h1), nn.ReLU(), nn.Linear(h1,h2), nn.ReLU(), nn.Linear(h2,a_space), nn.Softmax())
# model = nn.Sequential(nn.Linear(o_space,h1), nn.ReLU(), nn.Linear(h1,a_space),nn.Softmax())
# agent = PGAgent.PGAgent(model, o_space,a_space,lr = helper.give_initial_lr())
# #agent = torch.load('Agent.catcher')
# # model = nn.Sequential(nn.Linear(o_space,h1), nn.Sigmoid(), nn.Linear(h1,a_space))
# # agent = QAgent.QAgent(model,o_space,a_space,1500)
# success_hist = []
# successes_in_range = 0 

# for epoch in range(epochs): 
# 	s = env.reset()
# 	done = False
# 	reward = 0 
# 	steps = 0 
# 	ep_history = []

# 	while not done: 

# 		# if steps%successives_actions == 0:
# 		# 	if np.random.random() < greed(epoch): 
# 		# 		action = env.randomAction()
# 		# 	else: 
# 		# 		sTensor = Variable(torch.Tensor(s)).unsqueeze(0)
# 		# 		action = agent.think(sTensor)

# 		sTensor = Variable(torch.Tensor(s)).unsqueeze(0)
# 		action = agent.think(sTensor)

# 		n,r,done,success = env.step(action)
# 		ep_history.append([s,action,r])
# 		#ep_history.append([s,action,r,n])
# 		s = n 
# 		reward += r
# 		steps += 1

# 		if done: 
# 			successes_in_range += success
# 			ep_history = np.array(ep_history)
# 			ep_history[:,2] = discountReward(ep_history[:,2])
# 			history = []
# 			for ep in ep_history: 
# 				history.append([ep[0], ep[1], ep[2]])
# 			agent.train([history, steps])

# 			# for ep in ep_history: 
# 			# 	agent.remember([ep[0], ep[1], ep[2], ep[3]])
# 			# agent.train(batch_size)

# 			if (epoch)%info == 0: 
# 				print('{}/{} - Success: {}/{}'.format((epoch),epochs,successes_in_range,info))
# 				success_hist.append(successes_in_range)
				
# 				torch.save(agent, 'agent.twosides')
# 				helper.add_performance(successes_in_range)
# 				new_learning_rate = helper.update_LR()
# 				agent.change_learning_rate(new_learning_rate)
# 				print('Changed lr to {}'.format(new_learning_rate))
# 				print(helper.buffer)

# 				successes_in_range = 0

# env.initRender()
# s = env.reset()
# steps = 0 
# while True: 

# 	if steps%successives_actions == 0:
# 		if np.random.random() < 0.15: 
# 			action = env.randomAction()
# 		else: 
# 			action = agent.think(Variable(torch.Tensor(s)).unsqueeze(0))
# 	n, r, done, _ = env.step(action)
# 	s = n 
# 	env.render()
# 	steps += 1
# 	if done: 
# 		s = env.reset()


# w = World(robotJoints = 2, robotJointsLength = 0.35, 
# 	randomizeRobot = False, randomizeTarget = False, 
# 	groundHeight =0.05, targetLimits = [0.2,0.8,0.1,0.6])

# w.initRender()

# while True:
# 	state, reward, complete, success = w.step(w.randomAction())
	
# 	if complete: 
# 		w.reset()
# 		raw_input()

# 	w.render()

