import io
import pickle
import os


def parse_params(filename):
    hidden, lr, iterations = filename[7:-5].split('_')
    return int(hidden), float(lr), int(iterations)


def parse_data(filename):
    with io.open(filename, 'rb') as f:
        return pickle.load(f)


def main():
    filenames = (
        filename for filename in os.listdir('.')
        if filename.startswith('result.') and filename.endswith('.pickle')
    )
    data = (
        parse_data(filename) for filename in filenames
    )
    ordered = sorted(data, key=lambda result: result['val_loss'], reverse=True)
    for i, result in enumerate(ordered, 1):
        print(
            i, result['hidden_nodes'], result['learning_rate'],
            result['iterations'],
            'TRAIN', result['train_loss'],
            'VAL', result['val_loss']
        )


if __name__ == '__main__':
    main()
