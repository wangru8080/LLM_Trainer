import json
import sys
import os
import random
import argparse
import glob


parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='', required=False, help='The input data directory')
parser.add_argument('--output', type=str, required=True, help='The output file')
parser.add_argument('--proportion_config', type=str, required=True, help='The json configuration specifying the proportion of one type of data used in one epoch')
parser.add_argument('--epochs', type=int, required=True, help='The total epochs built for the whole dataset')
parser.add_argument('--seed', type=int, default=42, help='The random seed')
args = parser.parse_args()


def load_samples(dir_path):
    samples = []
    files = glob.glob(f'{dir_path}/*.json')
    if not files:
        print(f'No files match {dir_path}/*.json', file=sys.stderr)
        exit(-1)

    for path in files:
        with open(path) as fin:
            for line in fin:
                obj = json.loads(line)
                assert 'instruction' in obj and 'input' in obj and 'output' in obj and obj['output'], f'Invalid format in {path}:\n{line}'
                samples.append(obj)

    if not samples:
        print(f'No sample data in directory - {dir_path}', file=sys.stderr)
        exit(-1)

    return samples


random.seed(args.seed)

with open(args.proportion_config) as fin:
    config = json.load(fin)

data_dict = {}
offset_dict = {}
for key, val in config.items():
    assert val >=0 and val <= 1.0, f'Invalid ratio - {key}:{val}'

    samples = load_samples(f'{args.input}/{key}')
    random.shuffle(samples)
    data_dict[key] = samples
    offset_dict[key] = 0

fout = open(args.output, 'w')

for epoch in range(args.epochs):
    epoch_data = []
    for key, val in config.items():
        num = int(len(data_dict[key]) * val)

        start = offset_dict[key]
        samples = data_dict[key][start:start + num]
        if len(samples) < num:
            remain = num - len(samples)
            # reuse data in cycle
            samples.extend(data_dict[key][:remain])
            offset_dict[key] = remain
        else:
            offset_dict[key] = start + num 

        epoch_data.extend(samples)

    random.shuffle(epoch_data)

    for obj in epoch_data:
        print(json.dumps(obj, ensure_ascii=False), file=fout, flush=True)

fout.close()
