import io
import json
import os


def parse_params(filename):
    hidden, lr, iterations = filename[7:-5].split('_')
    return int(hidden), float(lr), int(iterations)


def parse_data(filename):
    with io.open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['train'][-1], data['validation'][-1]


def main():
    filenames = (
        filename for filename in os.listdir('.')
        if filename.startswith('result.') and filename.endswith('.json')
    )
    data = {
        parse_params(filename): parse_data(filename) for filename in filenames
    }
    ordered = sorted(data, key=lambda params: data[params][1], reverse=True)
    best_params = ordered[-1]
    print('BEST PARAMS', best_params, 'RESULT', data[best_params])
    import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()
