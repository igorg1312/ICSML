import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tensorflow.keras.layers import Concatenate, Dense
np.set_printoptions(precision=3, suppress=True)
import tensorflow as tf

from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

from tensorflow.keras.layers import Embedding, Input, Flatten
from tensorflow.keras.models import Model

print(tf.__version__)

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['Energia', 'Corrente','Voltaggio','ActivePower','Response','Waste']
raw_dataset = pd.read_csv("final.csv", sep=',')
dataset = raw_dataset.copy()



dataset = pd.get_dummies(dataset, prefix_sep='')
print(dataset)
train_dataset = dataset.sample(frac=0.5, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

sns.pairplot(train_dataset[['Energia','Corrente','Voltaggio','ActivePower','Response','Waste']],diag_kind='kde').savefig("Response.png")




print(train_dataset.describe().transpose())

train_features = train_dataset.copy()
test_features = test_dataset.copy()


print(train_dataset.describe().transpose()[['mean', 'std']])
train_labels = train_features.pop('Corrente')
test_labels = test_features.pop('Corrente')


normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(train_features))

print(normalizer.mean.numpy())


first = np.array(train_features[:1])

with np.printoptions(precision=2, suppress=True):
  print('First example:', first)
  print()
  print('Normalized:', normalizer(first).numpy())



horsepower = np.array(train_features['Waste'])

horsepower_normalizer = preprocessing.Normalization(input_shape=[1,])
horsepower_normalizer.adapt(horsepower)


horsepower_model = tf.keras.Sequential([
    horsepower_normalizer,
    layers.Dense(units=1)
])

horsepower_model.summary()


print(horsepower_model.predict(horsepower[:10]))


horsepower_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')


history = horsepower_model.fit(
    train_features['Waste'], train_labels,
    epochs=100,
    # suppress logging
    verbose=0,
    # Calculate validation results on 20% of the training data
    validation_split = 0.2)


hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist)



def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 200])
  plt.xlabel('Epoch')
  plt.ylabel('Error [Waste]')
  plt.legend()
  plt.grid(True)
  plt.savefig("loss.png")

plot_loss(history)


test_results = {}

test_results['horsepower_model'] = horsepower_model.evaluate(
    test_features['Waste'],
    test_labels, verbose=0)


x = tf.linspace(0.0, 250, 251)
y = horsepower_model.predict(x)



def plot_horsepower(x, y):
  plt.scatter(train_features['Waste'], train_labels, label='Data')
  plt.plot(x, y, color='k', label='Predictions')
  plt.xlabel('Waste')
  plt.ylabel('Corrente')
  plt.legend()
  plt.savefig("train1.png")


plot_horsepower(x,y)



linear_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
])



linear_model.predict(train_features[:10])


linear_model.layers[1].kernel

linear_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')



history = linear_model.fit(
    train_features, train_labels, 
    epochs=100,
    # suppress logging
    verbose=0,
    # Calculate validation results on 20% of the training data
    validation_split = 0.2)


plot_loss(history)



test_results['linear_model'] = linear_model.evaluate(
    test_features, test_labels, verbose=0)




def build_and_compile_model(norm):
  model = keras.Sequential([
      norm,
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
  ])

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
  return model




dnn_horsepower_model = build_and_compile_model(horsepower_normalizer)


dnn_horsepower_model.summary()



history = dnn_horsepower_model.fit(
    train_features['Waste'], train_labels,
    validation_split=0.2,
    verbose=0, epochs=100)


plot_loss(history)


x = tf.linspace(0.0, 250, 251)
y = dnn_horsepower_model.predict(x)

plot_horsepower(x, y)

test_results['dnn_horsepower_model'] = dnn_horsepower_model.evaluate(
    test_features['Waste'], test_labels,
    verbose=0)

dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()

history = dnn_model.fit(
    train_features, train_labels,
    validation_split=0.2,
    verbose=0, epochs=100)


plot_loss(history)
test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=0)


print(pd.DataFrame(test_results, index=['Mean absolute error [Current]']).T)



test_predictions = dnn_model.predict(test_features)

print(test_predictions)

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [Current]')
plt.ylabel('Predictions [Current]')
lims = [0, 200]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.savefig("Finale.png")





input_tensor = Input(shape=(2, ))

# Pass it to a Dense layer with 1 unit
output_tensor = Dense(1)(input_tensor)

# Create a model
model = Model(input_tensor, output_tensor)

# Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error')

from sklearn.model_selection import train_test_split

model.fit(train_dataset[['Corrente', 'ActivePower']],
          train_dataset['Energia'],
          epochs=1,
          verbose=True);
print(model.evaluate(test_dataset[['Corrente', 'ActivePower']],
                     test_dataset['Energia'],
                     verbose=True))
