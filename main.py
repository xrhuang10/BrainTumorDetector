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
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import 

dataset_path = "tumorsDataset"

# Define paths to the "testing" and "training" subfolders
testing_path = os.path.join(dataset_path, "testing")
training_path = os.path.join(dataset_path, "training")
paths = {
    "testing_path": testing_path,
    "training_path": training_path,
}
with open("paths.json", "w") as file:
    json.dump(paths, file)


categories = ['glioma', 'meningioma', 'notumor', 'pituitary']
np.save("categories.npy", categories)


train_data = []
for category in categories:
    folder_path = os.path.join(training_path, category)
    images = os.listdir(folder_path)
    count = len(images)
    train_data.append(pd.DataFrame({"Image": images, "Category": [category] * count, "Count": [count] * count}))

train_df = pd.concat(train_data, ignore_index=True)

#DISTRIBUTION OF TUMOR TYPES
custom_palette = ['blue', 'red', 'green', 'blue']
plt.figure(figsize=(8, 6))
sns.barplot(data=train_df, x="Category", y="Count", palette=custom_palette)
plt.title("Distribution of Tumor Types")
plt.xlabel("Tumor Type")
plt.ylabel("Count")
plt.show()


#TYPES OF TUMORS EXAMPLES
plt.figure(figsize=(12, 8))
for i, category in enumerate(categories):
    folder_path = os.path.join(training_path, category)
    image_path = os.path.join(folder_path, os.listdir(folder_path)[0])
    img = plt.imread(image_path)
    plt.subplot(2, 2, i+1)
    plt.imshow(img)
    plt.title(category)
    plt.axis("off")
plt.tight_layout()
plt.show()


# Set the image size
image_size = (150, 150)

# Set the batch size for training
batch_size = 32

# Set the number of epochs for training
epochs = 50

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale = 1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest"
)





print(train_datagen)
print(1)
train_generator = train_datagen.flow_from_directory(
    training_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical"
)
print(2)
print(train_generator)

#GPT GENERATED
train_dataset = tf.data.Dataset.from_generator(
    lambda: train_generator,
    output_signature=(
        tf.TensorSpec(shape=(None, *image_size, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, len(categories)), dtype=tf.float32)
    )
).prefetch(tf.data.AUTOTUNE)




train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(buffer_size=1000)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
with open("test_datagen.pkl", "wb") as file:
    pickle.dump(test_datagen, file)
print(3)
print(test_datagen)


test_generator = test_datagen.flow_from_directory(
    testing_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)
print(4)
print(test_generator)

test_generator_config = {
    "testing_path": testing_path,
    "image_size": image_size,
    "batch_size": batch_size,
    "class_mode": "categorical",
    "shuffle": False,
}
# Save the configuration to a JSON file
with open("test_generator_config.json", "w") as config_file:
    json.dump(test_generator_config, config_file)


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(image_size[0], image_size[1], 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(categories), activation="softmax")
])


model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = [
        'accuracy'
    ]
)

model.save("model.h5")


history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=test_generator,
)






# Save the history object to a file
with open("history.pkl", "wb") as file:
    pickle.dump(history.history, file)  # Use history.history to save the dictionary
