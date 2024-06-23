from enum import Enum
from keras.applications import ResNet50
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.losses import CategoricalCrossentropy
from keras.models import Sequential
from torchvision.models import efficientnet_b0, mobilenet_v2

from resnet import ResNet34
from model_interfaces import ModelInterface, TensorFlowModel, TorchModel

class MODEL(Enum):
	RESNET_34 = 'resnet_34'
	RESNET_50 = 'resnet_50'
	RESNET_SIMPLE = 'resnet_simple'
	EFFICIENT_NET = 'efficient_net'
	MOBILE_NET = 'mobile_net'

class ModelGarden:
	def __init__(self, input_shape: tuple[int, ...], num_classes: int, model_list: list[str] = [], custom_model_list: list[tuple[str, ModelInterface]] = []) -> None:
		self.input_shape = input_shape
		self.num_classes = num_classes
		self.model_generators = [*self.parse_model_list(model_list), *self.parse_custom_model_list(custom_model_list)]

	def parse_model_list(self, model_list: list[MODEL]) -> list:
		models = []
		for model_identifier in model_list:
			if model_identifier == MODEL.RESNET_50: model_generator = self.resnet_50
			elif model_identifier == MODEL.RESNET_SIMPLE: model_generator = self.sequential
			elif model_identifier == MODEL.RESNET_34: model_generator = self.resnet_34
			elif model_identifier == MODEL.EFFICIENT_NET: model_generator = self.efficient_net
			elif model_identifier == MODEL.MOBILE_NET: model_generator = self.mobile_net
			else:
				raise ValueError(f'Unsupported model type: {model_identifier}')
			
			models.append((model_identifier.value, model_generator))
		return models
	
	# wrap each custom model in a function so we can call it like the others
	# is this enough? its probably returning the same instance of the model, so it's "resuming" training instead of starting from scratch.
	def parse_custom_model_list(self, custom_model_list: list[ModelInterface]) -> list:
		custom_models = []
		for model_name, model in custom_model_list:
			custom_models.append((model_name, lambda: model))
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