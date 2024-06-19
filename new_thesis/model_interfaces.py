import torch
from numpy import ndarray
import tensorflow as tf

from keras.models import load_model
from torch.utils.data import DataLoader, Dataset

class ModelInterface:
	def __init__(self, model) -> None:
		self.model = model

	def get_accuracies(self, x: ndarray, y: ndarray, epochs: int, batch_size: int, validation_data: tuple[ndarray, ndarray]) -> tuple[float, float]:
		pass

	def load_model(self, path: str):
		pass

class TensorFlowModel(ModelInterface):
	def __init__(self, model: tf.keras.Model) -> None:
		super().__init__(model)

	def get_accuracies(self, x: ndarray, y: ndarray, epochs: int, batch_size: int, validation_data: tuple[ndarray, ndarray]) -> tuple[float, float]:
		history = self.model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=validation_data)
		acc = history.history['categorical_accuracy'][-1]
		valacc = history.history['val_categorical_accuracy'][-1]

		return acc, valacc

	def load_model(self, path: str):
		return load_model(path)
	
class TorchModel(ModelInterface):
	def __init__(self, model: torch.nn.Module) -> None:
		super().__init__(model)
		self.optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
		self.criterion = torch.nn.CrossEntropyLoss()

		if not torch.cuda.is_available():
			self.model = model

		model = model.to('cuda')
		self.model = torch.nn.DataParallel(model)

	def get_accuracies(self, x: ndarray, y: ndarray, epochs: int, batch_size: int, validation_data: tuple[ndarray, ndarray]) -> tuple[float, float]:
		weights_dtype = next(self.model.parameters()).dtype

		train_set = CustomDataset(x, y, x_dtype=weights_dtype)
		train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

		test_set = CustomDataset(validation_data[0], validation_data[1], x_dtype=weights_dtype)
		test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=2)

		# train model
		for _ in range(epochs):
			acc = self.train(train_loader)
			val_acc = self.test(test_loader)
		
		return acc, val_acc

	def train(self, data_loader: DataLoader) -> float:
		self.model.train()
		correct = 0
		total = 0
		for inputs, targets in data_loader:
			inputs, targets = inputs.to('cuda'), targets.to('cuda')
			self.optimizer.zero_grad()
			outputs = self.model(inputs)
			loss = self.criterion(outputs, targets)
			loss.backward()
			self.optimizer.step()

			_, predicted = outputs.max(1)
			total += targets.size(0)
			
			# undo one-hot
			targets_direct = torch.argmax(targets, dim=1)
			correct += predicted.eq(targets_direct).sum().item()

		return correct / total

	def test(self, data_loader: DataLoader) -> float:
		self.model.eval()
		correct = 0
		total = 0
		with torch.no_grad():
			for inputs, targets in data_loader:
				inputs, targets = inputs.to('cuda'), targets.to('cuda')
				outputs = self.model(inputs)

				_, predicted = outputs.max(1)
				total += targets.size(0)
				# undo one-hot
				targets_direct = torch.argmax(targets, dim=1)
				correct += predicted.eq(targets_direct).sum().item()

		return correct / total

	def load_model(self, path: str):
		return torch.jit.load(path)

class CustomDataset(Dataset):
	def __init__(self, x: ndarray, y: ndarray, x_dtype=None, y_dtype=None):
		# convert batch of color images from color channel last (expected) to channel first (torch format)
		if len(x.shape) == 4:
			x = x.reshape((x.shape[0], x.shape[-1], *x.shape[1: -1]))

		x_tensor, y_tensor = torch.from_numpy(x), torch.from_numpy(y)
		if x_dtype:
			x_tensor = x_tensor.type(x_dtype)
		if y_dtype:
			y_tensor = y_tensor.type(y_dtype)

		self.x, self.y = x_tensor, y_tensor

	def __len__(self):
		return len(self.y)

	def __getitem__(self, idx):
		return self.x[idx], self.y[idx]