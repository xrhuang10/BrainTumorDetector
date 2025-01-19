import os
import numpy as np
import sklearn as sk
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import json
import pickle

#Load the model
model = tf.keras.models.load_model("model.h5")

# Load the saved history object
with open("history.pkl", "rb") as file:
    history = pickle.load(file)

#Load the categories
categories = np.load("categories.npy")

#Load testing_path
with open("paths.json", "r") as file:
    paths = json.load(file)

testing_path = paths["testing_path"]
training_path = paths["training_path"]

print("Testing Path:", testing_path)
print("Training Path:", training_path)

#Load test_generator
with open("test_generator_config.json", "r") as config_file:
    config = json.load(config_file)

#test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

with open("test_datagen.pkl", "rb") as file:
    test_datagen = pickle.load(file)

# test_generator = test_datagen.flow_from_directory(
#     config["testing_path"],
#     target_size=tuple(config["image_size"]),
#     batch_size=config["batch_size"],
#     class_mode=config["class_mode"],
#     shuffle=config["shuffle"],
# )

test_generator = test_datagen.flow_from_directory(
    testing_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)



# Use the loaded history object
print(history.keys())  # Check available metrics



plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])
plt.show()


loss, accuracy = model.evaluate(test_generator, steps=test_generator.samples // 32)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)


# Make predictions on the test dataset
predictions = model.predict(test_generator)
predicted_categories = np.argmax(predictions, axis=1)
true_categories = test_generator.classes

from sklearn.metrics import classification_report

print(classification_report(true_categories, predicted_categories, target_names=categories))

# Create a confusion matrix
confusion_matrix = tf.math.confusion_matrix(true_categories, predicted_categories)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(ticks=np.arange(len(categories)), labels=categories)
plt.yticks(ticks=np.arange(len(categories)), labels=categories)
plt.show()

# # Plot sample images with their predicted and true labels
test_images = test_generator.filenames
sample_indices = np.random.choice(range(len(test_images)), size=9, replace=False)
sample_images = [test_images[i] for i in sample_indices]
sample_predictions = [categories[predicted_categories[i]] for i in sample_indices]
sample_true_labels = [categories[true_categories[i]] for i in sample_indices]

plt.figure(figsize=(12, 8))
for i in range(9):
    plt.subplot(3, 3, i+1)
    img = plt.imread(os.path.join(testing_path, sample_images[i]))
    plt.imshow(img)
    plt.title(f"Predicted: {sample_predictions[i]}\nTrue: {sample_true_labels[i]}")
    plt.axis("off")
plt.tight_layout()
plt.show()

# # Analyze the sample images and their predictions
plt.figure(figsize=(12, 8))
for i in range(9):
    plt.subplot(3, 3, i+1)
    img = plt.imread(os.path.join(testing_path, sample_images[i]))
    plt.imshow(img)
    if sample_predictions[i] == sample_true_labels[i]:
        plt.title(f"Predicted: {sample_predictions[i]}\nTrue: {sample_true_labels[i]}", color='green')
    else:
        plt.title(f"Predicted: {sample_predictions[i]}\nTrue: {sample_true_labels[i]}", color='red')
    plt.axis("off")
plt.tight_layout()
plt.show()



# Calculate precision, recall, and F1-score from the confusion matrix
precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
f1_score = 2 * (precision * recall) / (precision + recall)

# Print precision, recall, and F1-score for each class
for i, category in enumerate(categories):
    print(f"Class: {category}")
    print(f"Precision: {precision[i]}")
    print(f"Recall: {recall[i]}")
    print(f"F1-Score: {f1_score[i]}")
    print()


model.save("brain_tumor_detection_model.h5")


