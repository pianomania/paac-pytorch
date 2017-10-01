# refer to http://matplotlib.org/faq/howto_faq.html#matplotlib-in-a-web-application-server
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle


class Recorder(object):

	def __init__(self, env_id, game_name):

		self.env_id = env_id
		self.history_score = []
		self.score = 0
		self.counts = 0
		self.game_name = game_name

	def record(self, reward, done):

		self.counts += 1
		self.score += reward
		if done:
			self.history_score.append(self.score)
			self.score = 0

		self._save()

	def _save(self):

		if self.counts % 5000 == 0:
			
			plt.figure()
			plt.plot(self.history_score)
			plt.xlabel('Episode')
			plt.ylabel('Score')
			plt.title(self.game_name)
			plt.savefig('./log/env_%d.png' % self.env_id)
			plt.close()
