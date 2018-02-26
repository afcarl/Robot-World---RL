import numpy as np 
import pygame as pg 


class World():

	def __init__(self, robotJoints = 4, robotJointsLength = 0.18, 
				randomizeRobot = False, randomizeTarget = False, 
				groundHeight = 0.05, targetLimits = [0.2,0.8,0.2,0.8], 
				maxSteps = 200, state_type = 'angles',
				target_description = 'position'):

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

		self.render_ready = False
		self.state_type = state_type
		self.target_description = target_description

	def first_run(self): 

		if self.state_type == 'angles':
			print('\nInfos: State is [cos(a), sin(a) for a in angles] + [Vector to target]')

		if self.state_type == 'positions': 
			print('\nInfos: State is [v_x, v_y for v in robot_joints] + [Vector to target]')

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

		self.ground.draw(screen, screenSize)
		self.robot.draw(screen, screenSize)
		self.target.draw(screen, screenSize)

	def observe(self): 

		# ------------------------------------------------------------------
		# State is distance between the effector and the ball in the following form -> [dX_Positive, dX_Negative, dY_Pos, dY_Neg]
		# ------------------------------------------------------------------

		targetPosition = self.target.position
		effectorPosition = self.robot.points[-1]

		vector = targetPosition - effectorPosition
		distance = np.sqrt(np.sum(vector**2))
		
		if self.state_type == 'positions': 
			state = []
			pos = self.robot.joints_positions()
			for p in pos: 
				for e in p:
					state.append(e)

		# ------------------------------------------------------------------
		# New state representation using angles
		# ------------------------------------------------------------------
		if self.state_type == 'angles':
			state = []
			for a in self.robot.angles: 
				state.append(np.cos(np.radians(a)))
				state.append(np.sin(np.radians(a)))
		

		
		if self.target_description == 'position':
			for p in targetPosition: 
				state.append(p)
		elif self.target_description == 'vector': 
			for p in vector: 
				state.append(p)
		else: 
			raise KeyError


		# ------------------------------------------------------------------
		# ------- Reward -----------
		# ------------------------------------------------------------------
		reward = -0.01*distance
		#reward = 0.

		angle_penalty = 0.7*np.abs(self.robot.angles[0]) + 0.1*np.abs(self.robot.angles[1])
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
		for p in self.robot.points:    # checking whether touching the ground
			if p[1] < self.ground.height: 
				complete = True
				reward = 0

		if distance < 0.03:  # target reached
			complete = True
			success = 1
			reward = 1.

		if self.steps > self.maxSteps: 
			complete = True
			reward = 0

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

	def obSpaceSize(self): 
		s,_,_,_ = self.observe()
		return len(s)

	def get_env_infos(self): 
		return [self.obSpaceSize(), self.actionSpaceSize()]

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

	def close(self): 
		pg.quit()
		self.render_ready = False



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

	def __init__(self, nbJoints, LengthJoints, baseHeight = 0.0501, speed = 4, randomize = False): 

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
		for a in self.angles: 
			a = np.clip(a,-120,120)

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


