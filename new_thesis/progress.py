import numpy as np
import tkinter as tk
from tkinter import ttk

from test_suite import Tester

class ProgressPage:
	def __init__(self, parent_root: tk.Tk, test_suite: Tester, reduced_csvs: list[str]) -> None:
		self.root = tk.Toplevel(parent_root)
		self.test_suite = test_suite
		self.reduced_csvs = reduced_csvs
		
		self.root.title('Test Progress')
		self.root.geometry('600x600')
		
		# Start button
		self.start_btn = tk.Button(self.root, text='Start', command=self.start)
		self.start_btn.pack(pady=25)

		# Progress bars
		self.progress_labels = []
		self.progress_bars = []
		for path in reduced_csvs:
			file_name = path.split('/')[-1].split('.')[0]
			label = tk.Label(self.root, text=f'Reduction {file_name} progress')
			progress_bar = ttk.Progressbar(self.root, orient="horizontal", length=200, mode="determinate")

			self.progress_labels.append(label)
			self.progress_bars.append(progress_bar)

			label.pack(pady=(5, 5))
			progress_bar.pack(pady=(0, 25))

			progress_bar['maximum'] = len(test_suite.model_garden.model_generators)
			progress_bar['value'] = 0

	def start(self):
		models_completed = 0
		val_x, val_y = self.test_suite.reduce_dataset(np.random.randint(0, self.test_suite.base.shape[0], 1000))
		for accuracies in self.test_suite.evaluate_reductions_live(self.reduced_csvs, validation=(val_x, val_y)):
			reduction_index = models_completed // len(self.test_suite.model_garden.model_generators)

			# this is the end of the loop, the final `accuracies` holds the list of average validation accuracies for each reduction
			if reduction_index == len(self.progress_bars):
				break

			self.progress_bars[reduction_index]['value'] += 1
			self.root.update_idletasks()
			models_completed += 1

		# display these nicely eventually
		print(accuracies)