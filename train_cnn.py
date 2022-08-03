import argparse
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

parser = argparse.ArgumentParser(
    description='Training Convolutional Neural Network With Keras')

# Params for CNN
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    help='Initial learning rate')

# Train params
parser.add_argument('--b', '--batch-size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--e', '--num-epochs', '--epochs', default=100, type=int,
                    help='The number epochs')
parser.add_argument('--ve', '--validation-epochs', default=50, type=int,
                    help='The number epochs between running validation')
parser.add_argument('--md', '--model-dir', default='models/cnn/model_cnn.h5',
                    help='Directory for saving models')

args = parser.parse_args()

def main():
	train_augmentation = ImageDataGenerator(
					rescale=1./255, 
					rotation_range=40, 
					width_shift_range=0.2, 
					height_shift_range=0.2,
					shear_range=0.2, 
					zoom_range=0.2, 
					horizontal_flip=True, 
					fill_mode='nearest')                  

	val_augmentation = ImageDataGenerator(
					rescale=1./255)

	train_generator = train_augmentation.flow_from_directory(
					'data/images/train/', 
					target_size=(24, 24), 
					batch_size=args.b, 
					class_mode='binary',
					color_mode='grayscale')

	val_generator = val_augmentation.flow_from_directory(
				'data/images/val/', 
				target_size=(24, 24), 
				batch_size=args.b,
				class_mode='binary',
				color_mode='grayscale')

	model = Sequential()
	model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(24, 24, 1)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dropout(0.5))
	model.add(Dense(512, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))

	model.summary()
	print('Batch size: ', args.b)
	print('Learning rate: ', args.lr)
	print('Epochs: ', args.e)

	model.compile(
		loss='binary_crossentropy', 
		optimizer=Adam(learning_rate=args.lr), 
		metrics=['accuracy'])

	history = model.fit(
		train_generator, 
		epochs=args.e, 
		validation_data=val_generator, 
		validation_steps=args.ve)
	
	# model.save(args.md)
	print("Complete training.................................")

	acc = history.history['accuracy']
	val_acc = history.history['val_accuracy']
	loss = history.history['loss']
	val_loss = history.history['val_loss']
	epochs = range(1, len(acc) + 1)
	plt.plot(epochs, acc, 'bo', label='Training acc')
	plt.plot(epochs, val_acc, 'b', label='Validation acc')
	plt.title('Training and validation accuracy')
	plt.legend()
	plt.figure()
	plt.plot(epochs, loss, 'bo', label='Training loss')
	plt.plot(epochs, val_loss, 'b', label='Validation loss')
	plt.title('Training and validation loss')
	plt.legend()
	plt.show()

if __name__ == "__main__":
	main()