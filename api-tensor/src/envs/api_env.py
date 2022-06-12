import random
from time import sleep
import gym
import requests
import flatdict
import numpy as np
import pandas as pd
import keras
import tensorflow as tf

from gym.spaces import MultiDiscrete

from pandas import DataFrame

from endpoint import APIEndpoint

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam


from tensorflow import feature_column

class APIEnvModelCallback(keras.callbacks.Callback):
    required_accuracy: float

    def __init__(self, required_accuracy=.95) -> None:
        super().__init__()
        self.required_accuracy = required_accuracy

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None and logs.get('accuracy') >= self.required_accuracy:
            self.model.stop_training = True

class APIEnv(gym.Env):
    endpoint: APIEndpoint
    data: DataFrame = None
    action_history: list = list()

    def _unique_values_in_list_of_lists(self, lst):
        res = []

        for i in lst:
            try:
                indx = res.index(i)
            except: res.append(i)

        return res

    def _create_sub_set(self, action_history: list, labels: list):
        frames = []
        for i in action_history:
            _df = pd.DataFrame(np.array(i).reshape(-1, len(labels)), columns=labels)
            frames.append(_df)

        df = pd.concat(frames)
        labels_ds = df.pop(labels[-1])
        df = tf.data.Dataset.from_tensor_slices((dict(df), labels_ds))
        df = df.batch(1)

        return df

    def _create_set(self, action_history: list, labels: list):
        df = pd.DataFrame([i for i in action_history], columns=labels)
        labels_ds = df.pop(labels[-1])
        df = tf.data.Dataset.from_tensor_slices((dict(df), labels_ds))
        df = df.batch(1)

        return df

    def _create_model(self, feature_layer, optimizer, loss):
        model = Sequential([
            feature_layer,
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=['accuracy'])

        return model

    def _get_processed_data(self, actions: list, db: DataFrame, shuffle=False):
        rows = [i[1] for i in db.iterrows()]

        if shuffle:
            random.shuffle(rows)

        obj_to_use = None
        for row in rows:
            has_all = True
            for i in actions:
                nan = False
                if isinstance(row[i], str):
                    nan = False
                else:
                    nan = np.isnan(row[i])

                if nan == True:
                    has_all = False
                    break

            if has_all:
                obj_to_use = [row[s] for s in actions]
                break

        if obj_to_use is None:
            tmpobj = []

            for i in actions:
                for row in rows:
                    if isinstance(row[i], str):
                        nan = False
                    else:
                        nan = np.isnan(row[i])

                    if nan == False:
                        has_all = False
                        tmpobj.append(row[i])
                        break
                if len(tmpobj) == len(actions):
                    obj_to_use = tmpobj
                    break

        return obj_to_use

    def _execute_endpoint(self, endpoint_input: dict) -> flatdict:
        headers = {
            'Content-type': 'application/json',
            'Accept': 'application/json'
        }

        if self.endpoint.method == 'GET':
            return flatdict.FlatDict(
                requests.get(url=self.endpoint.url, params={}, headers=headers).json())
        else:
            return flatdict.FlatDict(
                requests.post(url=self.endpoint.url, json=endpoint_input, headers=headers).json())

    def __init__(self, endpoint: APIEndpoint, db: DataFrame, goal_callback):
        self.endpoint = endpoint
        self.data = db
        self.action_space = MultiDiscrete(
            [len(db.columns)]*len(endpoint.possible_inputs))
        self.goal_callback = goal_callback
        self.legend = {i:db.columns[i] for i in range(len(db.columns))}

    def Learn(self, n_steps=10000, epochs=15, optimizer='adam', loss='mean_squared_error', force_end_success=0, required_accuracy=.99, delay=0):
        for i in range(n_steps):
            if force_end_success > 0:
                successful_sequences = len(
                    [i for i in self.action_history if i[-1] == 1])

                if successful_sequences >= force_end_success:
                    break
            sleep(delay/1000)

            sample = self.action_space.sample()
            self.step(sample)
            self.reset()

            self.action_history = self.action_history
        
        # Get Unique Action Sequences
        history = self.action_history # self._unique_values_in_list_of_lists(self.action_history)

        labels = ["label_" + str(i)
                  for i in range(len(history[0]))]

        train_ds = self._create_set(
            [i for i in history], labels)

        feature_columns = [
            feature_column.numeric_column(i) for i in labels[:-1]]

        feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

        model = self._create_model(feature_layer, optimizer, loss)
        model.fit(train_ds, epochs=epochs, callbacks=[
                  APIEnvModelCallback(required_accuracy=required_accuracy)])
        model.save('./train/')

    def Run(self, n_steps=5, delay=0):
        model = keras.models.load_model(
            './train/')

        # Get Unique Action Sequences
        history = [] # self._unique_values_in_list_of_lists(self.action_history)

        for i in self.action_history:
            try:
                history.index(i)
            except: history.append(i)

        history = [i for i in history]
        action_history_labels = ["label_" + str(i)
                                    for i in range(len(history[0]))]

        train_ds = self._create_sub_set(
            history, action_history_labels)

        prediction = model.predict(train_ds)
        best_sequence = np.argmax(prediction)
        best_sequence = history[best_sequence]

        print('Best Sequence: {}'.format(best_sequence))

        for i in range(n_steps):
            state, reward, done, info = self.step(best_sequence[:-1])
            assert(info['response']['Success'] == True)
            sleep(delay/1000)

        best_sequence = [self.legend[best_sequence[i]] for i in range(len(best_sequence)-1)]

        return best_sequence[:-1]

    def step(self, action):
        self.state = action
        reward = 0
        processed_data = self._get_processed_data(action, self.data, shuffle=True)
        input = {self.endpoint.possible_inputs[i-1]: [processed_data[i-1]]
                 for i in range(len(processed_data))}

        resp = self._execute_endpoint(input)

        if self.goal_callback(resp) == True:
            reward = 1

        self.action_history.append([a for a in action] + [reward])

        info = {
            'response': resp
        }

        done = False
        return self.state, reward, done, info

    def render(self):
        pass

    def reset(self):
        return self.state