from keras.applications import ResNet50
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout

class ModelGarden:
	def __init__(self, input_shape: tuple[int, ...], num_classes: int) -> None:
		self.input_shape = input_shape
		self.num_classes = num_classes
	
	def create_resnet_model(self):
		return ResNet50(weights=None, input_shape=self.input_shape, pooling='max', classes=self.num_classes, classifier_activation='softmax')
		
	def create_sequential_model(self):
		return Sequential([
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