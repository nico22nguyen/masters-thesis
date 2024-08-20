from model_interfaces import ModelInterface
from model_garden import ModelGarden

import gc
import numpy as np
import pandas as pd
import tkinter as tk
from keras.utils import to_categorical

class Tester:
	def __init__(self, base_csv: str, shape: tuple[int, ...], custom_model_paths=[], default_models=[], epochs=100, learning_rate=4e-4, batch_size=256, tk_root: tk.Tk=None) -> None:
		'''Expects csv to be in the following format:
			1. One row per sample in dataset
			2. First column specifies class (integer value)
			3. Remaining columns are the features (flat if multidimensional)
			4. **DO NOT** include a header row or an index column in the csv, only data

			leave dataset size off of `original_shape` e.g. (32, 32, 3) instead of (60000, 32, 32, 3)
		'''
		self.base = pd.read_csv(base_csv, header=None)
		self.shape = shape
		self.num_classes = self.base[0].unique().shape[0]
		self.epochs = epochs
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.tk_root = tk_root

		# define models used for evaluation
		self.model_garden = ModelGarden(shape, self.num_classes, default_models, custom_model_paths)
		
	def evaluate_reductions(self, reduced_csvs: list[str], validation: tuple[np.ndarray, np.ndarray]):
		'''Each csv in `reduced_csvs` should be a newline separated list of integers.
			- It is assumed that size(indices) <= size(base)
		'''
		
		accuracies = []
		for index_csv_path in reduced_csvs:
			indices = pd.read_csv(index_csv_path, header=None).to_numpy().squeeze()
			reduced_base = self.base.iloc[indices]
			reduced_x, reduced_y = self.prep_data(reduced_base)

			acclist = []
			for model_name, model_generator in self.model_garden.model_generators:
				model: ModelInterface = model_generator()
				print(f'training {model_name}...')
				
				for acc, valacc in model.get_accuracies(reduced_x, reduced_y, epochs=self.epochs, batch_size=self.batch_size, validation_data=validation):
					print(f'\tacc: {acc}, valacc: {valacc}')
					yield valacc
					if self.tk_root:
						self.tk_root.update()

				acclist.append(valacc)

				# clean up, important for conserving VRAM
				gc.collect()
	
			accuracies.append(np.array(acclist).mean())

		yield accuracies
		
	def prep_data(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
		# separate to x, y
		y = to_categorical(df[0].to_numpy(), self.num_classes) # y is first column
		x = df.drop(0, axis=1).to_numpy() # x is remaining columns

		# ensure x is reshaped to original dimensions
		new_shape = (x.shape[0],) + self.shape
		x = x.reshape(new_shape) 
		
		return x, y

if __name__ == '__main__':
	from model_garden import MODEL
	print('Testing load_base_csv() on cifar csv:\n')
	print('Loading csv...')
	tester = Tester('datasets/cifar_base.csv', (32, 32, 3), default_models=[MODEL.RESNET_34, MODEL.RESNET_34])
	val_x, val_y = tester.prep_data(np.random.randint(0, tester.base.shape[0], 1000))
	accuracies = tester.evaluate_reductions(['datasets/cifar_indices_1.csv', 'datasets/cifar_indices_2.csv', 'datasets/cifar_indices_3.csv'], (val_x, val_y))
	print(accuracies)

	# print(f'reduced x <shape: {reduced_x.shape}>\n', reduced_x, '\n')
	# print(f'reduced y <shape: {reduced_y.shape}>\n', reduced_y)