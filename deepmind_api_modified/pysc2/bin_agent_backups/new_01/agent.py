#!/usr/bin/python
# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# AGENT VER 1
"""Run an agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import threading

from future.builtins import range  # pylint: disable=redefined-builtin

from pysc2 import maps
from pysc2.env import available_actions_printer
from pysc2.env import run_loop
from pysc2.env import sc2_env
from pysc2.lib import stopwatch

from absl import app
from absl import flags


import threading
import tensorflow as tf
import time
from keras.models import *
from keras.layers import *
from keras import backend as K
import os

import brain

mybrain = brain.Brain()

FLAGS = flags.FLAGS
flags.DEFINE_bool("render", False, "Whether to render with pygame.")
flags.DEFINE_integer("screen_resolution", 84,
					 "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 64,
					 "Resolution for minimap feature layers.")

flags.DEFINE_integer("max_agent_steps", 9999999, "Total agent steps.")
flags.DEFINE_integer("game_steps_per_episode", 0, "Game steps per episode.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")

flags.DEFINE_string("agent", "pysc2.agents.parallelization_test.SparseAgent",
					"Which agent to run")
flags.DEFINE_enum("agent_race", "T", sc2_env.races.keys(), "Agent's race.")
flags.DEFINE_enum("bot_race", "T", sc2_env.races.keys(), "Bot's race.")
flags.DEFINE_enum("difficulty", "1", sc2_env.difficulties.keys(),
				  "Bot's strength.")

flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
flags.DEFINE_integer("parallel", 1, "How many instances to run in parallel.")

flags.DEFINE_bool("save_replay", False, "Whether to save a replay at the end.")

flags.DEFINE_string("map", "Simple64", "Name of a map to use.")
flags.mark_flag_as_required("map")


class Environment(threading.Thread):
	stop_signal = False

	def __init__(self, agent_cls, map_name, visualize): #needs to take arguments that initialize thread
		threading.Thread.__init__(self)


	def run_thread(self, agent_cls, map_name, visualize):
		global mybrain
		with sc2_env.SC2Env(
			map_name=map_name,
			agent_race=FLAGS.agent_race,
			bot_race=FLAGS.bot_race,
			difficulty=FLAGS.difficulty,
			step_mul=FLAGS.step_mul,
			game_steps_per_episode=FLAGS.game_steps_per_episode,
			screen_size_px=(FLAGS.screen_resolution, FLAGS.screen_resolution),
			minimap_size_px=(FLAGS.minimap_resolution, FLAGS.minimap_resolution),
			visualize=visualize) as env:
			env = available_actions_printer.AvailableActionsPrinter(env)
		agent = agent_cls(mybrain)
		run_loop.run_loop([agent], env, FLAGS.max_agent_steps)
		if FLAGS.save_replay:
			env.save_replay(agent_cls.__name__)

	def runEpisode(self):

		while True:
			time.sleep(THREAD_DELAY) # yield

			my_thread = run_thread(agent_cls, map_name, visualize)

			if done or self.stop_signal:
				break

		print("Total R:", R)

	def run(self):
		while not self.stop_signal:
			self.runEpisode()

	def stop(self):
		self.stop_signal = True


class Optimizer(threading.Thread):
	stop_signal = False

	def __init__(self):
		threading.Thread.__init__(self)

	def run(self):
		while not self.stop_signal:
			mybrain.optimize()

	def stop(self):
		self.stop_signal = True

# def run_thread(agent_cls, map_name, visualize):
#   with sc2_env.SC2Env(
#       map_name=map_name,
#       agent_race=FLAGS.agent_race,
#       bot_race=FLAGS.bot_race,
#       difficulty=FLAGS.difficulty,
#       step_mul=FLAGS.step_mul,
#       game_steps_per_episode=FLAGS.game_steps_per_episode,
#       screen_size_px=(FLAGS.screen_resolution, FLAGS.screen_resolution),
#       minimap_size_px=(FLAGS.minimap_resolution, FLAGS.minimap_resolution),
#       visualize=visualize) as env:
#     env = available_actions_printer.AvailableActionsPrinter(env)
#     agent = agent_cls()
#     run_loop.run_loop([agent], env, FLAGS.max_agent_steps)
#     if FLAGS.save_replay:
#       env.save_replay(agent_cls.__name__)


def main(unused_argv):
	RUN_TIME = 999999
	THREADS = 8
	OPTIMIZERS = 2
	THREAD_DELAY = 0.001

	GAMMA = 0.99

	N_STEP_RETURN = 8
	GAMMA_N = GAMMA ** N_STEP_RETURN

	EPS_START = 0.4
	EPS_STOP  = .15
	EPS_STEPS = 3000

	MIN_BATCH = 32
	LEARNING_RATE = 5e-3

	LOSS_V = .5			# v loss coefficient
	LOSS_ENTROPY = .01 	# entropy coefficient
	"""Run an agent."""

	NUM_STATE = 12
	NUM_ACTIONS = 8
	NONE_STATE = np.zeros(NUM_STATE)

	stopwatch.sw.enabled = FLAGS.profile or FLAGS.trace
	stopwatch.sw.trace = FLAGS.trace

	maps.get(FLAGS.map)  # Assert the map exists.

	agent_module, agent_name = FLAGS.agent.rsplit(".", 1)
	agent_cls = getattr(importlib.import_module(agent_module), agent_name)

	# threads = []
	# for _ in range(FLAGS.parallel - 1):
	#   t = threading.Thread(target=run_thread, args=(agent_cls, FLAGS.map, False))
	#   threads.append(t)
	#   t.start()

	envs = [Environment(agent_cls, FLAGS.map, False) for i in range(THREADS)]
	opts = [Optimizer() for i in range(OPTIMIZERS)]

	for o in opts:
		o.start()

	for e in envs:
		e.start()

	time.sleep(RUN_TIME)

	for e in envs:
		e.stop()

	for e in envs:
		e.join()

	for o in opts:
		o.stop()

	for o in opts:
		o.join()

  # run_thread(agent_cls, FLAGS.map, FLAGS.render)

  # for t in threads:
  #   t.join()

	if FLAGS.profile:
		print(stopwatch.sw)


def entry_point():  # Needed so setup.py scripts work.
	app.run(main)


if __name__ == "__main__":
	app.run(main)
