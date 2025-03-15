from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Dropout, Attention
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from assistance_functions import DeepLearning

# Set a maximum memory limit
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_logical_device_configuration(gpus[0],[tf.config.LogicalDeviceConfiguration(memory_limit=8044)] ) # Limit to 7.9 gigs approx.

# Input 1
input_1 = Input(shape=(21,6), name='TA')
x1 = LSTM(64, return_sequences=True, name='lstm_TA')(input_1)
x = LSTM(64, return_sequences=False, name='lstm_TA-2')(x1)

# Fully connected layers
x = Dense(32, activation='relu', name='dense_2')(x)
x = Dense(32, activation='relu', name='dense_3')(x)

# Output layer
output = Dense(3, activation='softmax', name='output')(x)

model = Model(inputs=input_1, outputs=output)


model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),loss=keras.losses.categorical_crossentropy,
    metrics=[tf.keras.metrics.Precision(class_id=0, name='precision_buy'),
            tf.keras.metrics.Recall(class_id=0, name='recall_buy'),
            tf.keras.metrics.Precision(class_id=1, name='precision_sell'),
            tf.keras.metrics.Recall(class_id=1, name='recall_sell'),
            tf.keras.metrics.Precision(class_id=2, name='NO_trd_precision'),
            tf.keras.metrics.Recall(class_id=2, name='NO_trd_recall')]
)

model.summary()

early_stopping = EarlyStopping(patience=5, monitor='loss', restore_best_weights=True)


# implement pruning and regularization techs and fix the metrics to be used
y_train = np.load("C:/Users/muyu2/OneDrive/Documents/DeepLearning/eda/tp_10_sl_10_label.npy")
y_train = DeepLearning.label_encode(y_train)
x1_train= np.load("C:/Users/muyu2/OneDrive/Documents/DeepLearning/eda/tp_10_sl_10_timestep.npy", allow_pickle=True)

#correct shape of X1
tech = np.concatenate(x1_train).flatten()
x1_train = tech.reshape(len(x1_train), 21,6).astype(np.float32)

model.fit(x1_train,y_train, batch_size=128, epochs=100 ,verbose=2, callbacks=[early_stopping])#validation split and early call back removed
model.save("C:/Users/muyu2/OneDrive/Documents/DeepLearning/new/cluster_bbands.h5")

#For the test/validation
x1_t = np.load(f'C:/Users/muyu2/OneDrive/Documents/DeepLearning/tp_7_sl__6/timesteps.npy', allow_pickle=True)
y_test = np.load(f'C:/Users/muyu2/OneDrive/Documents/DeepLearning/tp_7_sl__6/label.npy', allow_pickle=True)

# Split to test and validation set( 60/40 split )

tech = np.concatenate(x1_t).flatten()
x1_test= tech.reshape(len(x1_t), 21,6).astype(np.float32)

# Split to test and validation set( 60/40 split )
model.evaluate(x1_test, y_test , verbose=2)



