import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from my_answers import NeuralNetwork
import sys
from my_answers import iterations, learning_rate, hidden_nodes, output_nodes
import itertools
import pickle
import io
import os.path
from multiprocessing import Pool


def MSE(y, Y):
    return np.mean((y - Y) ** 2)


def load_data():
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

    return (
        N_i,
        train_features, train_targets,
        test_features, test_targets,
        val_features, val_targets,
        scaled_features, rides, test_data
    )


def train(hidden_nodes, learning_rate, iterations):
    sys.stdout.write(f'Training: {hidden_nodes} {learning_rate} {iterations}\n')
    fname = f'result.{hidden_nodes}_{learning_rate}_{iterations}.pickle'
    if os.path.exists(fname):
        with io.open(fname, 'rb') as f:
            return pickle.load(f)

    (
        N_i,
        train_features, train_targets,
        test_features, test_targets,
        val_features, val_targets,
        _scaled_features, _rides, _test_data
    ) = load_data()

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
        # sys.stdout.write("\rProgress: {:2.1f}".format(100 * ii/float(iterations))
        #                  + "% ... Training loss: " + str(train_loss)[:5]
        #                  + " ... Validation loss: " + str(val_loss)[:5])
        # sys.stdout.flush()

        losses['train'].append(train_loss)
        losses['validation'].append(val_loss)

    sys.stdout.write(f'DONE training: {hidden_nodes} {learning_rate} {iterations} Training loss: {str(train_loss)[:5]}, validation loss: {str(val_loss)[:5]}\n')
    sys.stdout.flush()

    fname = f'result.{hidden_nodes}_{learning_rate}_{iterations}.pickle'

    result = {
        'train_loss': train_loss,
        'val_loss': val_loss,
        'network': network,
        'losses': losses,
        'hidden_nodes': hidden_nodes,
        'learning_rate': learning_rate,
        'iterations': iterations
    }
    with io.open(fname, 'wb') as f:
        pickle.dump(result, f)
    return result


def main():
    hidden_nodes_opts = [20, 30, 40]
    learning_rate_opts = [0.1, 0.5, 1, 5, 10]
    iterations_opts = [5000, 10000, 20000, 30000]
    opts = itertools.product(
        hidden_nodes_opts, learning_rate_opts, iterations_opts
    )

    with Pool(5) as pool:
        results = pool.starmap(train, opts)

    ordered = sorted(
        results, key=lambda result: result['val_loss'], reverse=True
    )
    for i, result in enumerate(ordered, 1):
        print(
            i,
            result['hidden_nodes'], result['learning_rate'],
            result['iterations'],
            'TRAIN', result['train_loss'], 'VAL', result['val_loss']
        )


def x():
    train_loss, val_loss, network, losses = train(*best_params)
    print('SCORE TRAIN', train_loss, 'VALIDATION', val_loss)

    fig, ax = plt.subplots()
    plt.plot(losses['train'], label='Training loss')
    plt.plot(losses['validation'], label='Validation loss')
    min_validation = min(losses['validation'])
    plt.legend()
    ax.set_yscale("log")
    ax.axhline(min_validation, linestyle='--', color='k')
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
