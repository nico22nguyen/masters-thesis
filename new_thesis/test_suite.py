from model_interfaces import ModelInterface, TensorFlowModel, TorchModel
from model_garden import ModelGarden

import gc
import numpy as np
import pandas as pd
from keras.utils import to_categorical

class Tester:
	def __init__(self, base_csv: str, shape: tuple[int, ...], custom_model_paths=[], default_models=[], epochs=100, learning_rate=4e-4, batch_size=256) -> None:
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

		# define models used for evaluation
		custom_models = self.parse_custom_model_paths(custom_model_paths) if custom_model_paths else []
		self.model_garden = ModelGarden(shape, self.num_classes, default_models, custom_models)

	def evaluate_reductions(self, reduced_csvs: list[str], validation: tuple[np.ndarray, np.ndarray]) -> list[float]:
		'''Each csv in `reduced_csvs` should be a newline separated list of integers.
			- It is assumed that size(indices) <= size(base)
		'''
		
		accuracies = []
		for index_csv_path in reduced_csvs:
			print(f'evaluating: {index_csv_path.split("/")[-1]}...')
			indices = pd.read_csv(index_csv_path, header=None).to_numpy().squeeze()
			reduced_x, reduced_y = self.reduce_dataset(indices)

			acclist = []
			for i, model_generator in enumerate(self.model_garden.model_generators):
				# train model and record training/validation accuracy
				model: ModelInterface = model_generator()
				
				acc, valacc = model.get_accuracies(reduced_x, reduced_y, epochs=self.epochs, batch_size=self.batch_size, validation_data=validation)
				# del model
				# acc = history.history['categorical_accuracy'][-1]
				# valacc = history.history['val_categorical_accuracy'][-1]
				print(f"  MODEL {i + 1}: train acc = {acc:.5f} val acc = {valacc:.5f}")

				acclist.append(valacc)

				# clean up, important for conserving VRAM
				# del history
				gc.collect()
	
			accuracies.append(np.array(acclist).mean())
		
	def reduce_dataset(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		# reduce base by indices
		reduced_base = self.base.iloc[indices]

		# separate to x, y
		y = to_categorical(reduced_base[0].to_numpy(), 10) # y is first column
		x = reduced_base.drop(0, axis=1).to_numpy() # x is remaining columns

		# ensure x is reshaped to original dimensions
		new_shape = (x.shape[0],) + self.shape
		x = x.reshape(new_shape) 
		
		return x, y
	
	def parse_custom_model_paths(self, custom_model_paths: list[str]) -> list[ModelInterface]:
		custom_models = []
		for path in custom_model_paths:
			file_extension = path.split('.')[-1]
			if file_extension == 'keras':
				custom_model = TensorFlowModel.load_model(path)
			elif file_extension == 'pt':
				custom_model = TorchModel.load_model(path)
			else:
				raise ValueError(f'Unexpected file extension: {file_extension}')
			
			custom_models.append(custom_model)
		return custom_models

if __name__ == '__main__':
	print('Testing load_base_csv() on cifar csv:\n')
	print('Loading csv...')
	tester = Tester('datasets/cifar_base.csv', (32, 32, 3))
	val_x, val_y = tester.reduce_dataset(np.random.randint(0, tester.base.shape[0], 1000))
	accuracies = tester.evaluate_reductions(['datasets/cifar_indices_1.csv', 'datasets/cifar_indices_2.csv', 'datasets/cifar_indices_3.csv'], (val_x, val_y))
	print(accuracies)

	# print(f'reduced x <shape: {reduced_x.shape}>\n', reduced_x, '\n')
	# print(f'reduced y <shape: {reduced_y.shape}>\n', reduced_y)