import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from my_answers import NeuralNetwork
import sys
from my_answers import iterations, learning_rate, hidden_nodes, output_nodes
import itertools
import json
import io


def MSE(y, Y):
    return np.mean((y - Y) ** 2)


def main():
    np.random.seed(42)
    data_path = 'Bike-Sharing-Dataset/hour.csv'
    rides = pd.read_csv(data_path)

    dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
    for each in dummy_fields:
        dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
        rides = pd.concat([rides, dummies], axis=1)

    fields_to_drop = ['instant', 'dteday', 'season', 'weathersit',
                      'weekday', 'atemp', 'mnth', 'workingday', 'hr']
    data = rides.drop(fields_to_drop, axis=1)

    quant_features = ['casual', 'registered',
                      'cnt', 'temp', 'hum', 'windspeed']
    # Store scalings in a dictionary so we can convert back later
    scaled_features = {}
    for each in quant_features:
        mean, std = data[each].mean(), data[each].std()
        scaled_features[each] = [mean, std]
        data.loc[:, each] = (data[each] - mean) / std

    # Save data for approximately the last 21 days
    test_data = data[-21 * 24:]

    # Now remove the test data from the data set
    data = data[:-21 * 24]

    # Separate the data into features and targets
    target_fields = ['cnt', 'casual', 'registered']
    features, targets = data.drop(target_fields, axis=1), data[target_fields]
    test_features, test_targets = test_data.drop(
        target_fields, axis=1), test_data[target_fields]

    # Hold out the last 60 days or so of the remaining data as a validation set
    train_features, train_targets = features[:-60 * 24], targets[:-60 * 24]
    val_features, val_targets = features[-60 * 24:], targets[-60 * 24:]

    N_i = train_features.shape[1]

    def train(hidden_nodes, learning_rate, iterations):
        network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)

        losses = {'train': [], 'validation': []}
        for ii in range(iterations):
            # Go through a random batch of 128 records from the training data set
            batch = np.random.choice(train_features.index, size=128)
            X, y = train_features.ix[batch].values, train_targets.ix[batch]['cnt']

            network.train(X, y)

            # Printing out the training progress
            train_loss = MSE(network.run(train_features).T,
                             train_targets['cnt'].values)
            val_loss = MSE(network.run(val_features).T, val_targets['cnt'].values)
            sys.stdout.write("\rProgress: {:2.1f}".format(100 * ii/float(iterations))
                             + "% ... Training loss: " + str(train_loss)[:5]
                             + " ... Validation loss: " + str(val_loss)[:5])
            sys.stdout.flush()

            losses['train'].append(train_loss)
            losses['validation'].append(val_loss)

        sys.stdout.write(f'\nDONE training. Training loss: {str(train_loss)[:5]}, validation loss: {str(val_loss)[:5]}\n')
        sys.stdout.flush()
        return train_loss, val_loss, network, losses

    hidden_nodes_opts = [3, 5, 7, 9, 11, 13]
    learning_rate_opts = [0.0001, 0.001, 0.01, 0.1]
    iterations_opts = [500, 1000, 2000]

    best_params = (None, None, None)
    best_score = 1000

    optimize = False
    if optimize:
        for params in itertools.product(
            hidden_nodes_opts, learning_rate_opts, iterations_opts
        ):
            train_loss, val_loss, _, losses = train(*params)
            print('PARAMS', params, 'SCORE TRAIN', train_loss, 'VALIDATION', val_loss)
            x = '_'.join(map(str, params))
            with io.open(f'result.{x}.json', 'w', encoding='utf-8') as f:
                json.dump(losses, f)
            if val_loss < best_score:
                best_params = params
                best_score = val_loss
        print('BEST PARAMS', best_params)
    else:
        best_params = (7, 0.01, 5000)

    train_loss, val_loss, network, losses = train(*best_params)
    print('SCORE TRAIN', train_loss, 'VALIDATION', val_loss)

    fig, ax = plt.subplots()
    plt.plot(losses['train'], label='Training loss')
    plt.plot(losses['validation'], label='Validation loss')
    plt.legend()
    ax.set_yscale("log")
    plt.ylim()

    plt.show()

    fig, ax = plt.subplots(figsize=(8, 4))

    mean, std = scaled_features['cnt']
    predictions = network.run(test_features).T * std + mean
    ax.plot(predictions[0], label='Prediction')
    ax.plot((test_targets['cnt'] * std + mean).values, label='Data')
    ax.set_xlim(right=len(predictions))
    # ax.legend()

    dates = pd.to_datetime(rides.ix[test_data.index]['dteday'])
    dates = dates.apply(lambda d: d.strftime('%b %d'))
    ax.set_xticks(np.arange(len(dates))[12::24])
    ax.set_xticklabels(dates[12::24], rotation=45)
    plt.show()


if __name__ == '__main__':
    main()
