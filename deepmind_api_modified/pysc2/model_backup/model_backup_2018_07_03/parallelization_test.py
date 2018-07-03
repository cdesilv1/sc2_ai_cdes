import random
import math
import os

import numpy as np
import pandas as pd

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features


import tensorflow as tf

import time, random, threading

from keras.models import *
from keras.layers import *
from keras import backend as K

from keras.models import load_model

from pysc2.bin.brain import Brain

_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
_HARVEST_GATHER = actions.FUNCTIONS.Harvest_Gather_screen.id

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_PLAYER_SELF = 1
_PLAYER_HOSTILE = 4
_ARMY_SUPPLY = 5

_TERRAN_COMMANDCENTER = 18
_TERRAN_SCV = 45
_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_BARRACKS = 21
_NEUTRAL_MINERAL_FIELD = 341

_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_ALL = [2]

DATA_FILE = 'a3c'

ACTION_DO_NOTHING = 'donothing'
ACTION_BUILD_SUPPLY_DEPOT = 'buildsupplydepot'
ACTION_BUILD_BARRACKS = 'buildbarracks'
ACTION_BUILD_MARINE = 'buildmarine'
ACTION_ATTACK = 'attack'

smart_actions = [
	ACTION_DO_NOTHING,
	ACTION_BUILD_SUPPLY_DEPOT,
	ACTION_BUILD_BARRACKS,
	ACTION_BUILD_MARINE,
]

for mm_x in range(0, 64):
	for mm_y in range(0, 64):
		if (mm_x + 1) % 32 == 0 and (mm_y + 1) % 32 == 0:
			smart_actions.append(ACTION_ATTACK + '_' + str(mm_x - 16) + '_' + str(mm_y - 16))



	#____________________HYPERPARAMETERS_________________
GAMMA = 0.99
N_STEP_RETURN = 8
GAMMA_N = GAMMA ** N_STEP_RETURN
	#____________________HYPERPARAMETERS_END_________________


frames = 0
class Agent:
	def __init__(self, brain_object, eps_start=0.4, eps_end=0.15, eps_steps=3000, NUM_ACTIONS=8):
		self.eps_start = eps_start
		self.eps_end   = eps_end
		self.eps_steps = eps_steps

		self.mybrain = brain_object

		self.memory = []	# used for n_step return
		self.R = 0.

		self.NUM_ACTIONS = NUM_ACTIONS

	def getEpsilon(self):
		if(frames >= self.eps_steps):
			return self.eps_end
		else:
			return self.eps_start + frames * (self.eps_end - self.eps_start) / self.eps_steps	# linearly interpolate

	def act(self, s, excluded_actions):
		eps = self.getEpsilon()
		global frames; frames = frames + 1

		if random.random() < eps:
			return random.randint(0, self.NUM_ACTIONS-1)

		else:
			s = np.array([s])
			p = self.mybrain.predict_p(s)[0]
			# a = np.argmax(p)
			# print(p)
			while True:
				a = np.random.choice(self.NUM_ACTIONS, p=p)
				if a not in excluded_actions:
					return a

	def train(self, s, a, r, s_):
		def get_sample(memory, n):
			s, a, _, _  = memory[0]
			_, _, _, s_ = memory[n-1]

			return s, a, self.R, s_

		a_cats = np.zeros(self.NUM_ACTIONS)	# turn action into one-hot representation
		a_cats[a] = 1

		self.memory.append( (s, a_cats, r, s_) )

		self.R = ( self.R + r * GAMMA_N ) / GAMMA

		if s_ is None:
			while len(self.memory) > 0:
				n = len(self.memory)
				s, a, r, s_ = get_sample(self.memory, n)
				self.mybrain.train_push(s, a, r, s_)

				self.R = ( self.R - self.memory[0][2] ) / GAMMA
				self.memory.pop(0)

			self.R = 0

		if len(self.memory) >= N_STEP_RETURN:
			s, a, r, s_ = get_sample(self.memory, N_STEP_RETURN)
			self.mybrain.train_push(s, a, r, s_)

			self.R = self.R - self.memory[0][2]
			self.memory.pop(0)

	# possible edge case - if an episode ends in <N steps, the computation is incorrect

class SparseAgent(base_agent.BaseAgent):
	def __init__(self, brain_object):
		super(SparseAgent, self).__init__()

		self.learn = Agent(brain_object = brain_object)
		self.previous_action = None
		self.previous_state = None

		self.cc_y = None
		self.cc_x = None

		self.move_number = 0
		self.mybrain = brain_object
		# if os.path.isfile('/home/anaconda3/lib/python3.6/site-packages/pysc2/bin/a3c.h5'):
		# 	mybrain.model = keras.models.load_model('a3c.h5')

	def transformDistance(self, x, x_distance, y, y_distance):
		if not self.base_top_left:
			return [x - x_distance, y - y_distance]

		return [x + x_distance, y + y_distance]

	def transformLocation(self, x, y):
		if not self.base_top_left:
			return [64 - x, 64 - y]

		return [x, y]

	def splitAction(self, action_id):
		smart_action = smart_actions[action_id]

		x = 0
		y = 0
		if '_' in smart_action:
			smart_action, x, y = smart_action.split('_')

		return (smart_action, x, y)

	def step(self, obs):#env from agent.py is imported as obs
		super(SparseAgent, self).step(obs)

		if obs.last():
			reward = obs.reward
			fd = open('parallelization_results.csv', 'a')
			fd.write(str(reward))
			fd.write('\n')
			fd.close()

			self.learn.train(self.previous_state, self.previous_action, reward, None)
			self.mybrain.model.save('a3c.h5')
			self.previous_action = None
			self.previous_state = None

			self.move_number = 0

			return actions.FunctionCall(_NO_OP, [])

		unit_type = obs.observation['screen'][_UNIT_TYPE]

		if obs.first():
			player_y, player_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
			self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0

			self.cc_y, self.cc_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()

		cc_y, cc_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
		cc_count = 1 if cc_y.any() else 0

		depot_y, depot_x = (unit_type == _TERRAN_SUPPLY_DEPOT).nonzero()
		supply_depot_count = int(round(len(depot_y) / 69))

		barracks_y, barracks_x = (unit_type == _TERRAN_BARRACKS).nonzero()
		barracks_count = int(round(len(barracks_y) / 137))

		supply_used = obs.observation['player'][3]
		supply_limit = obs.observation['player'][4]
		army_supply = obs.observation['player'][5]
		worker_supply = obs.observation['player'][6]

		supply_free = supply_limit - supply_used

		if self.move_number == 0:
			self.move_number += 1

			current_state = np.zeros(12)
			current_state[0] = cc_count
			current_state[1] = supply_depot_count
			current_state[2] = barracks_count
			current_state[3] = obs.observation['player'][_ARMY_SUPPLY]

			hot_squares = np.zeros(4)
			enemy_y, enemy_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_HOSTILE).nonzero()
			for i in range(0, len(enemy_y)):
				y = int(math.ceil((enemy_y[i] + 1) / 32))
				x = int(math.ceil((enemy_x[i] + 1) / 32))

				hot_squares[((y - 1) * 2) + (x - 1)] = 1

			if not self.base_top_left:
				hot_squares = hot_squares[::-1]

			for i in range(0, 4):
				current_state[i + 4] = hot_squares[i]

			green_squares = np.zeros(4)
			friendly_y, friendly_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
			for i in range(0, len(friendly_y)):
				y = int(math.ceil((friendly_y[i] + 1) / 32))
				x = int(math.ceil((friendly_x[i] + 1) / 32))

				green_squares[((y - 1) * 2) + (x - 1)] = 1

			if not self.base_top_left:
				green_squares = green_squares[::-1]

			for i in range(0, 4):
				current_state[i + 8] = green_squares[i]

			if self.previous_action is not None:
				self.learn.train(self.previous_state, self.previous_action, 0, current_state)

			excluded_actions = []
			if supply_depot_count == 7 or worker_supply == 0:
				excluded_actions.append(1)

			if supply_depot_count == 0 or barracks_count == 3 or worker_supply == 0:
				excluded_actions.append(2)

			if supply_free == 0 or barracks_count == 0:
				excluded_actions.append(3)

			if army_supply == 0:
				excluded_actions.append(4)
				excluded_actions.append(5)
				excluded_actions.append(6)
				excluded_actions.append(7)

			rl_action = self.learn.act(current_state, excluded_actions)#is Agent.act() in a3c

			self.previous_state = current_state
			self.previous_action = rl_action

			# print(self.previous_action)

			if self.previous_action == None:
				self.previous_action = 0

			smart_action, x, y = self.splitAction(self.previous_action)

			if smart_action == ACTION_BUILD_BARRACKS or smart_action == ACTION_BUILD_SUPPLY_DEPOT:
				unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()

				if unit_y.any():
					i = random.randint(0, len(unit_y) - 1)
					target = [unit_x[i], unit_y[i]]

					return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

			elif smart_action == ACTION_BUILD_MARINE:
				if barracks_y.any():
					i = random.randint(0, len(barracks_y) - 1)
					target = [barracks_x[i], barracks_y[i]]

					return actions.FunctionCall(_SELECT_POINT, [_SELECT_ALL, target])

			elif smart_action == ACTION_ATTACK:
				if _SELECT_ARMY in obs.observation['available_actions']:
					return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])

		elif self.move_number == 1:
			self.move_number += 1

			smart_action, x, y = self.splitAction(self.previous_action)

			if smart_action == ACTION_BUILD_SUPPLY_DEPOT:
				if supply_depot_count < 7 and _BUILD_SUPPLY_DEPOT in obs.observation['available_actions']:
					if self.cc_y.any():
						if supply_depot_count == 0:
							target = self.transformDistance(round(self.cc_x.mean()), -35, round(self.cc_y.mean()), 0)
						elif supply_depot_count == 1:
							target = self.transformDistance(round(self.cc_x.mean()), -25, round(self.cc_y.mean()), -25)
						elif supply_depot_count == 2:
							target = self.transformDistance(round(self.cc_x.mean()), -15, round(self.cc_y.mean()), 20)
						elif supply_depot_count == 3:
							target = self.transformDistance(round(self.cc_x.mean()), -15, round(self.cc_y.mean()), 25)
						elif supply_depot_count == 4:
							target = self.transformDistance(round(self.cc_x.mean()), -10, round(self.cc_y.mean()), 25)
						elif supply_depot_count == 5:
							target = self.transformDistance(round(self.cc_x.mean()), -10, round(self.cc_y.mean()), 20)
						elif supply_depot_count == 6:
							target = self.transformDistance(round(self.cc_x.mean()), 0, round(self.cc_y.mean()), 20)
						elif supply_depot_count == 7:
							target = self.transformDistance(round(self.cc_x.mean()), 0, round(self.cc_y.mean()), 25)

						return actions.FunctionCall(_BUILD_SUPPLY_DEPOT, [_NOT_QUEUED, target])
			elif smart_action == ACTION_BUILD_BARRACKS:
				if barracks_count < 3 and _BUILD_BARRACKS in obs.observation['available_actions']:
					if self.cc_y.any():
						if  barracks_count == 0:
							target = self.transformDistance(round(self.cc_x.mean()), 15, round(self.cc_y.mean()), -9)
						elif  barracks_count == 1:
							target = self.transformDistance(round(self.cc_x.mean()), 15, round(self.cc_y.mean()), 12)
						elif  barracks_count == 2:
							target = self.transformDistance(round(self.cc_x.mean()), 15, round(self.cc_y.mean()), 18)
						return actions.FunctionCall(_BUILD_BARRACKS, [_NOT_QUEUED, target])

			elif smart_action == ACTION_BUILD_MARINE:
				if _TRAIN_MARINE in obs.observation['available_actions']:
					return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])

			elif smart_action == ACTION_ATTACK:
				do_it = True

				if len(obs.observation['single_select']) > 0 and obs.observation['single_select'][0][0] == _TERRAN_SCV:
					do_it = False

				if len(obs.observation['multi_select']) > 0 and obs.observation['multi_select'][0][0] == _TERRAN_SCV:
					do_it = False

				if do_it and _ATTACK_MINIMAP in obs.observation["available_actions"]:
					x_offset = random.randint(-1, 1)
					y_offset = random.randint(-1, 1)

					return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, self.transformLocation(int(x) + (x_offset * 8), int(y) + (y_offset * 8))])

		elif self.move_number == 2:
			self.move_number = 0

			smart_action, x, y = self.splitAction(self.previous_action)

			if smart_action == ACTION_BUILD_BARRACKS or smart_action == ACTION_BUILD_SUPPLY_DEPOT:
				if _HARVEST_GATHER in obs.observation['available_actions']:
					unit_y, unit_x = (unit_type == _NEUTRAL_MINERAL_FIELD).nonzero()

					if unit_y.any():
						i = random.randint(0, len(unit_y) - 1)

						m_x = unit_x[i]
						m_y = unit_y[i]

						target = [int(m_x), int(m_y)]

						return actions.FunctionCall(_HARVEST_GATHER, [_QUEUED, target])

		return actions.FunctionCall(_NO_OP, [])
