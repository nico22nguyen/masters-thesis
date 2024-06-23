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
		
		self.reduction_status_frame=tk.Frame(self.root)
		self.reduction_status_frame.pack(fill='x', pady=(20, 0), padx=20)

		self.reduction_label_vars: list[tk.StringVar] = []
		self.reduction_labels = []
		for path in reduced_csvs:
			var = tk.StringVar(value=f'"{self.get_file_name(path)}" reduction evaluation status:')
			label = tk.Label(self.reduction_status_frame, textvariable=var, font=('TkDefaultFont', 12))
			self.reduction_label_vars.append(var)
			self.reduction_labels.append(label)
			label.pack(side='top', anchor='w')
		
		# Start button
		self.start_btn = tk.Button(self.root, text='Start', command=self.run_test_suite, width=10)
		self.start_btn.pack(pady=25)

		# Progress bars
		self.progress_frame = tk.Frame(self.root)
		self.progress_frame.pack(fill='x')

		self.progress_labels = []
		self.progress_bars = []
		for model_name, _ in test_suite.model_garden.model_generators:
			label = tk.Label(self.progress_frame, text=model_name)
			progress_bar = ttk.Progressbar(self.progress_frame, orient="horizontal", length=400, mode="determinate")

			self.progress_labels.append(label)
			self.progress_bars.append(progress_bar)

			label.pack(pady=(5, 5))
			progress_bar.pack(pady=(0, 25))

			progress_bar['maximum'] = test_suite.epochs
			progress_bar['value'] = 0
	
	def get_file_name(self, path: str) -> str:
		return path.split('/')[-1].split('.')[0]

	def update_reduction_label(self, reduction_index: int, status: str):
		file_name = self.get_file_name(self.reduced_csvs[reduction_index])
		self.reduction_label_vars[reduction_index].set(f'"{file_name}" reduction evaluation status: {status}')

	def run_test_suite(self):
		epochs_completed = 0
		val_x, val_y = self.test_suite.reduce_dataset(np.random.randint(0, self.test_suite.base.shape[0], 1000))
		self.update_reduction_label(0, 'In Progress')
		self.root.update_idletasks()

		for acc in self.test_suite.evaluate_reductions_live(self.reduced_csvs, validation=(val_x, val_y)):
			epoch_index = epochs_completed % self.test_suite.epochs
			model_index = (epochs_completed // self.test_suite.epochs) % len(self.progress_bars)
			reduction_index = epochs_completed // (self.test_suite.epochs * len(self.progress_bars))

			# loop is done, `acc` holds list of average accuracies per reduction
			if reduction_index == len(self.reduced_csvs):
				break
			
			# increment model progress bar
			self.progress_bars[model_index]['value'] += 1

			# last epoch of last model
			if model_index == len(self.progress_bars) - 1 and epoch_index == self.test_suite.epochs - 1:
				# update label with accuracy
				self.update_reduction_label(reduction_index, 'Complete')

				# reset progress bars if this isn't the last reduction
				if reduction_index < len(self.reduced_csvs) - 1:
					# reset reduction progress page
					for progress_bar in self.progress_bars:
						progress_bar['value'] = 0
					
					self.update_reduction_label(reduction_index + 1, 'In Progress')

			epochs_completed += 1
			self.root.update_idletasks()

		results = [(self.get_file_name(path), avg_acc) for path, avg_acc in zip(self.reduced_csvs, acc)]
		results = sorted(results, key=lambda x: x[1], reverse=True)

		self.progress_frame.destroy()
		
		results_frame = tk.Frame(self.root)
		results_frame.pack(fill='x')

		tk.Label(results_frame,  text='Results (avg validation accuracy)', font=('TkDefaultFont', 16)).pack(pady=(0, 10))
		for file_name, avg_acc in results:
			tk.Label(results_frame,  text=f'"{file_name}":\t\t\t{avg_acc*100:.2f}%', font=('TkDefaultFont', 12)).pack(pady=(0, 5))

		self.root.update_idletasks()