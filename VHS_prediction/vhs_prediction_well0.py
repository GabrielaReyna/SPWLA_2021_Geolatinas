import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os

print(tf.__version__)

#input: NEU, GR, PEF, DENC (skip PEF since well_0 doesn't have it)

#load well3 and well 4

src_path = r"/Users/astromeria/PycharmProjects/SPWLA_2021_Geolatinas/dataset/separated_wells/train/"
col_names_x = ["NEU", "GR", "DEN"]
col_names_y = ["VSH"]
col_names_all = col_names_x + col_names_y
wells_list = []
for i in range(1,9):
    well_data = pd.read_csv(src_path+"well_"+str(i)+".csv")
    well_data= well_data[col_names_all]
    wells_list.append(well_data)
well_all = pd.concat(wells_list)

#test data
well0 = pd.read_csv((src_path+"well_0.csv"))
well0 = well0[col_names_all]
well0 = well0.dropna()

print(well0.describe())

well_all= well_all.dropna()

x_train = well_all[col_names_x]
y_train = well_all[col_names_y]
x_test = well0[col_names_x]
y_test = well0[col_names_y]

#crear modelo
def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(x_train.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
  ])
  optimizer = tf.keras.optimizers.RMSprop(0.001)
  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

model = build_model()

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 100

history = model.fit(
  x_train, y_train,
  epochs=EPOCHS, validation_split = 0.1, verbose=0,
  callbacks=[PrintDot()])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mae'],
           label = 'Val Error')
  plt.ylim([0,5])
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mse'],
           label = 'Val Error')
  plt.ylim([0,0.7])
  plt.legend()
  plt.show()


plot_history(history)

test_predictions = model.predict(x_test).flatten()
predictions_file= os.path.join(os.getcwd(), "predictions_vhs_well0.npy")
np.save(predictions_file, test_predictions)

plt.scatter(y_test, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
plt.show()



#prepare dataset for prediction for the whole well
#test data
well0 = pd.read_csv((src_path+"well_0.csv"))
well0 = well0[col_names_all+["DEPTH"]]
well0 = well0[well0["NEU"].notna()]
well0 = well0[well0["GR"].notna()]
well0 = well0[well0["DEN"].notna()]

#take the columns needed for performing prediction and save npy file

well_0_x = well0[col_names_x]
test_predictions = model.predict(well_0_x).flatten()
predictions_file= os.path.join(os.getcwd(), "predictions_vhs_well0_whole.npy")
np.save(predictions_file, test_predictions)
#predict


fig = plt.figure(figsize=(10, 50))
ax = fig.add_subplot(1,1,1)

#Set up the plot axes

#ax1 = plt.subplot2grid((1,1), (0,0), rowspan=1, colspan = 1) # Gamma-ray



#Title of well logs plot


# As our curve scales will be detached from the top of the track,
# this code adds the top border back in without dealing with splines

print(len(well0["DEPTH"]))
print(len(well0["VSH"]))
print(len(test_predictions))

# Shale volume truth
ax.plot(well0["VSH"], well0["DEPTH"], color = "green", label="True VSH", alpha=0.7)
ax.plot( test_predictions, well0["DEPTH"], color = "blue", label="Predicted VSH", alpha=0.7)
plt.xlim([0,1])
plt.gca().invert_yaxis()
plt.legend()
plt.savefig("well_plot.png")
plt.show()


