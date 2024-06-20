from keras.applications import ResNet50
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.losses import CategoricalCrossentropy
from keras.models import Sequential
from torchvision.models import efficientnet_b0, mobilenet_v2

from resnet import ResNet34
from model_interfaces import ModelInterface, TensorFlowModel, TorchModel

class ModelGarden:
	def __init__(self, input_shape: tuple[int, ...], num_classes: int, model_list: list[str] = [], custom_model_list: list[ModelInterface] = []) -> None:
		self.input_shape = input_shape
		self.num_classes = num_classes
		self.model_generators = [*self.parse_model_list(model_list), *self.parse_custom_model_list(custom_model_list)]

	def parse_model_list(self, model_list: list[str]) -> list:
		models = []
		for model_identifier in model_list:
			if model_identifier == 'resnet_50': models.append(self.resnet_50)
			elif model_identifier == 'resnet_simple': models.append(self.sequential)
			elif model_identifier == 'resnet_34': models.append(self.resnet_34)
			elif model_identifier == 'efficient_net': models.append(self.efficient_net)
			elif model_identifier == 'mobile_net': models.append(self.mobile_net)
			else:
				raise ValueError(f'Unsupported model type: {model_identifier}')
		return models
	
	# wrap each custom model in a function so we can call it like the others
	def parse_custom_model_list(self, custom_model_list: list[ModelInterface]) -> list:
		custom_models = []
		for model in custom_model_list:
			custom_models.append(lambda: model)
		return custom_models
	
	def resnet_50(self) -> ModelInterface:
		model = ResNet50(weights=None, input_shape=self.input_shape, pooling='max', classes=self.num_classes, classifier_activation='softmax')
		model.compile(loss=CategoricalCrossentropy(), metrics=['categorical_accuracy'])
		return TensorFlowModel(model)
		
	def sequential(self) -> ModelInterface:
		model = Sequential([
			Conv2D(32, 3, padding='same', input_shape=self.input_shape, activation='relu'),
			Conv2D(32, 3, activation='relu'),
			MaxPooling2D(),
			Dropout(0.25),

			Conv2D(64, 3, padding='same', activation='relu'),
			Conv2D(64, 3, activation='relu'),
			MaxPooling2D(),
			Dropout(0.25),

			Flatten(),
			Dense(512, activation='relu'),
			Dropout(0.5),
			Dense(self.num_classes, activation='softmax'),
		])
		model.compile(loss=CategoricalCrossentropy(), metrics=['categorical_accuracy'])
		return TensorFlowModel(model)
	
	def resnet_34(self) -> ModelInterface:
		net = ResNet34(num_classes=self.num_classes)
		return TorchModel(net)
	
	def efficient_net(self) -> ModelInterface:
		net = efficientnet_b0(weights=None, progress=False, num_classes=self.num_classes)
		return TorchModel(net)
	
	def mobile_net(self) -> ModelInterface:
		net = mobilenet_v2(weights=None, progress=False, num_classes=self.num_classes)
		return TorchModel(net)