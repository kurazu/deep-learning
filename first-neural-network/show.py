import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from my_answers import iterations, learning_rate, hidden_nodes
from train import train, load_data


def main():
    (
        N_i,
        train_features, train_targets,
        test_features, test_targets,
        val_features, val_targets,
        scaled_features, rides, test_data
    ) = load_data()
    result = train(hidden_nodes, learning_rate, iterations)
    losses = result['losses']
    network = result['network']

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
