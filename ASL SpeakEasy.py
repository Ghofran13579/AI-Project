import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
import pyttsx3

# Paths to the dataset
train_data_dir = r"C:\Users\massa\Downloads\archive (2)\asl_alphabet_train\asl_alphabet_train"
test_data_dir = r"C:\Users\massa\Downloads\archive (2)\asl_alphabet_test\asl_alphabet_test"
img_height, img_width = 64, 64
batch_size = 32

# ImageDataGenerator for data augmentation and rescaling
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
test_datagen = ImageDataGenerator(rescale=1./255)

# Training data generator
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training')

# Validation data generator
validation_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')

# Load and preprocess test images
def load_test_images(test_dir, img_height, img_width):
    test_images = []
    filenames = []
    for fname in os.listdir(test_dir):
        img_path = os.path.join(test_dir, fname)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (img_height, img_width))
            img = img / 255.0
            test_images.append(img)
            filenames.append(fname)
    return np.array(test_images), filenames

# Load test images
test_images, test_filenames = load_test_images(test_data_dir, img_height, img_width)

# Define the CNN model using the Input layer
model = Sequential([
    Input(shape=(img_height, img_width, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=10
)

# Save the trained model in the new Keras format
model.save('sign_language_model.keras')

# Load the trained model
model = load_model('sign_language_model.keras')

# Reinitialize the optimizer
from tensorflow.keras.optimizers import RMSprop
model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])

# Evaluate the model on the test data
test_predictions = model.predict(test_images)
predicted_labels = np.argmax(test_predictions, axis=1)
label_map = {v: k for k, v in train_generator.class_indices.items()}
predicted_labels_text = [label_map[label] for label in predicted_labels]

# Display the test results
for filename, label in zip(test_filenames, predicted_labels_text):
    print(f"File: {filename}, Predicted Label: {label}")

# Initialize webcam
cap = cv2.VideoCapture(0)

# Get the labels from the training data generator
labels = list(train_generator.class_indices.keys())

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Function to convert text to speech
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Real-time prediction loop
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess the frame
    img = cv2.resize(frame, (img_height, img_width))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Make prediction
    prediction = model.predict(img)
    predicted_label = labels[np.argmax(prediction)]
    
    # Display the prediction on the frame
    cv2.putText(frame, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Sign Language Recognition', frame)
    
    # Convert the prediction to speech
    speak(predicted_label)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
