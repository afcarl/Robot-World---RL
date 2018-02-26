import pygame as pg 
import numpy as np 


class Robot(): 

	def __init__(self, nbJoints, LengthJoints, speed = 4, randomize = False): 

		self.nbJoints = nbJoints
		self.uLength = LengthJoints

		self.speed = speed
		self.position = np.array([0.5,0.5])
		self.angles = np.zeros((nbJoints))

		if randomize: 
			self.angles = np.random.randint(low = -80, high = 80, size =(nbJoints))

		self.computePositions()

	def rotate(self, action): 

		jointIndex = action/2
		if jointIndex < self.nbJoints:
			direction = 1 if action%2 == 0 else -1
			self.angles[int(jointIndex)] += direction*self.speed
		else: 
			d1 = action%2
			d2 = action - (self.nbJoints + 2)
			vec = d1*np.array([d2,0]) + (1-d1)*np.array([0,d2])
			self.position += vec/100.
			self.position[0] = np.clip(self.position[0],0.05,0.95)
			self.position[1] = np.clip(self.position[1],0.05,0.95)


	def computePositions(self): 

		self.points = np.zeros((self.nbJoints+1,2))
		for i in range(self.nbJoints+1): 
			if i == 0: 
				self.points[i,:] = self.position
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

			if j == 0: 
				a = 30
				pb = p0 - np.array([a,0])
				pc = p0 + np.array([a,0])
				pg.draw.line(screen, (220,150,20), pb.astype(int), pc.astype(int), 20)
				pb = pb + np.array([0,15])
				pc = pc + np.array([0,15])
				pg.draw.circle(screen, (0,150,20), pb.astype(int),10)
				pg.draw.circle(screen, (0,150,20), pc.astype(int),10)

			pg.draw.line(screen, (220,150,20), p0.astype(int), p1.astype(int), 5)
			pg.draw.circle(screen, (150,250,0), p0.astype(int), 10)
			pg.draw.circle(screen, (150,250,250), p1.astype(int), 10) 

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

class World():

	def __init__(self, robotJoints = 2, robotJointsLength = 0.18, 
				randomizeRobot = False, randomizeTarget = False, 
				targetLimits = [0.2,0.8,0.2,0.8], maxSteps = 200):

		self.randomizeRobot = randomizeRobot
		self.randomizeTarget = randomizeTarget
		self.targetLimits = targetLimits
		self.maxSteps = maxSteps

		self.robot = Robot(robotJoints,robotJointsLength, randomize = randomizeRobot)
		self.target = Target(self.targetLimits)

		self.listed_positions = False
		self.robotParameters = [robotJoints, robotJointsLength]

		self.steps = 0

		self.currentDistance = (self.target.position - self.robot.points[-1])**2
		self.currentDistance = np.sqrt(self.currentDistance)

		self.render_ready = False

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

		if not self.render_ready: 
			self.initRender()
			self.render_ready = True

		time = 30
		self.clock.tick(time)
		self.screen.fill((0,0,0))
		self.draw(self.screen, self.size)

		pg.display.flip() 

	def draw(self,screen, screenSize): 

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

		for p in vector: 
			state.append(p)

		#state = [targetPosition[0], targetPosition[1], effectorPosition[0], effectorPosition[1]]

		# ------------------------------------------------------------------
		# ------- Reward -----------
		# ------------------------------------------------------------------
		reward = 0.01/distance
		#reward = 0.

		#angle_penalty = 0.7*np.abs(self.robot.angles[0]) + 0.1*np.abs(self.robot.angles[1])
		#reward -= 0.001*angle_penalty

		#reward *= 0.1
		self.currentDistance = distance

		# ------------------------------------------------------------------
		# ------- Completion -------- 
		# ------------------------------------------------------------------
		#reward = -1
		complete = False
		success = 0
		i = 0

		if distance < self.target.radius/250.:  # target reached
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

		maxActions = self.robot.nbJoints*2 + 4
		action = np.random.randint(maxActions)
		return action

	def actionSpaceSize(self): 
		return self.robot.nbJoints*2

	def obSpaceSize(self): 
		s,_,_,_ = self.observe()
		return len(s)

	def get_env_infos(self): 
		return [self.obSpaceSize(), self.actionSpaceSize()]

	def reset(self): 

		self.steps = 0
		self.robot = Robot(self.robotParameters[0], self.robotParameters[1], randomize= self.randomizeRobot)
		if self.randomizeTarget: 
			self.target = Target(self.targetLimits)
		if self.listed_positions: 
			self.target_position_iterator = (self.target_position_iterator+1)%len(self.target_positions)
			self.target.position = np.array(self.target_positions[self.target_position_iterator])


		state,_,__,___ = self.observe()
		return state


e = World()
e.reset()
e.initRender()

while True: 
	e.render()
	action = e.randomAction()
	e.step(action)