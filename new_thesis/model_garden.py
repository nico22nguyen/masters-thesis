from keras.applications import ResNet50
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.losses import CategoricalCrossentropy
from keras.models import Sequential
from torchvision.models import efficientnet_b0, mobilenet_v2

from resnet import ResNet34
from model_interfaces import ModelInterface, TensorFlowModel, TorchModel

class ModelGarden:
	def __init__(self, input_shape: tuple[int, ...], num_classes: int) -> None:
		self.input_shape = input_shape
		self.num_classes = num_classes
	
	def create_resnet_model(self) -> ModelInterface:
		model = ResNet50(weights=None, input_shape=self.input_shape, pooling='max', classes=self.num_classes, classifier_activation='softmax')
		model.compile(loss=CategoricalCrossentropy(), metrics=['categorical_accuracy'])
		return TensorFlowModel(model)
		
	def create_sequential_model(self) -> ModelInterface:
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
	
	def create_torch_resnet(self) -> ModelInterface:
		net = ResNet34(num_classes=self.num_classes)
		return TorchModel(net)
	
	def create_efficient_net_model(self) -> ModelInterface:
		net = efficientnet_b0(weights=None, progress=False, num_classes=self.num_classes)
		return TorchModel(net)
	
	def create_mobile_net_model(self) -> ModelInterface:
		net = mobilenet_v2(weights=None, progress=False, num_classes=self.num_classes)
		return TorchModel(net)