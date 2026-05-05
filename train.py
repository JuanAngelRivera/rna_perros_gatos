import tensorflow as tf

img_size = (128, 128)
batch_size = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    'dataset',
    validation_split = 0.2,
    subset = 'training',
    seed = 123,
    image_size = img_size,
    batch_size = batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    'dataset',
    validation_split = 0.2,
    subset = 'validation',
    seed = 123,
    image_size = img_size,
    batch_size = batch_size
)

normalization_layer = tf.keras.layers.Rescaling(1./255)

class_names = train_ds.class_names

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (128, 128, 3)),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, (3, 3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(8, activation = 'softmax')
])

model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

history = model.fit(
    train_ds,
    validation_data = val_ds,
    epochs = 10
)

loss, acc = model.evaluate(val_ds)
print('precisión: ', acc)


import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.legend()
plt.title('Precisión del modelo')
plt.show()


import numpy as np

y_true = []
y_pred = []

for images, labels in val_ds:
    preds = model.predict(images)
    preds = np.argmax(preds, axis = 1)

    y_pred.extend(preds)
    y_true.extend(labels.numpy())


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class_names = class_names
cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = class_names)
disp.plot(cmap = 'Blues')
plt.title("Matriz de Confusión")
plt.show()


from sklearn.metrics import classification_report

report = classification_report(y_true, y_pred, target_names = class_names, output_dict = True)
print(report)


import pandas as pd
precision_por_clase = {}

for i, clase in enumerate(class_names):
    precision_por_clase[i] = report[clase]['precision']

plt.bar(precision_por_clase.keys(), precision_por_clase.values())
plt.xticks(rotation = 45)
plt.title("Precisión por clase")
plt.ylabel("Precisión")
plt.show()