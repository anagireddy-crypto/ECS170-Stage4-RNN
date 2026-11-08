"""Train and evaluate a configurable Stage 3 CNN experiment."""

import argparse
import csv
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


template_root = Path(__file__).resolve().parents[2]
if str(template_root) not in sys.path:
    sys.path.insert(0, str(template_root))

from local_code.stage_3_code.Dataset_Loader_Stage3 import Dataset_Loader_Stage3
from local_code.stage_3_code.Evaluate_Classification_Stage3 import Evaluate_Classification_Stage3
from local_code.stage_3_code.Method_CNN_Stage3 import Method_CNN_Stage3


def parse_integer_list(value):
    try:
        return tuple(int(part.strip()) for part in value.split(',') if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError('expected comma-separated integers') from error


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['mnist', 'orl', 'cifar'], required=True)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=170)
    parser.add_argument('--conv-channels', type=parse_integer_list, default=(32, 64),
                        help='output channels per convolution, e.g. 32,64,128')
    parser.add_argument('--kernel-sizes', type=parse_integer_list, default=(3,),
                        help='one odd size for all layers or one per layer, e.g. 3 or 3,5')
    parser.add_argument('--hidden-dims', type=parse_integer_list, default=(128,),
                        help='dense hidden dimensions, e.g. 128 or 256,128; use 0 for none')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--adaptive-pool-size', type=int, default=4)
    parser.add_argument('--experiment-name', default='baseline')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        parser.error('epochs, batch-size, and learning-rate must be positive')
    if not 0.0 <= args.dropout < 1.0:
        parser.error('dropout must be in [0, 1)')
    if any(size <= 0 for size in args.conv_channels):
        parser.error('convolution channel sizes must be positive')
    if any(size < 0 for size in args.hidden_dims):
        parser.error('hidden dimensions cannot be negative')
    args.hidden_dims = tuple(size for size in args.hidden_dims if size > 0)
    return args


def main():
    args = parse_arguments()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available')
    device = torch.device(
        'cuda' if args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available())
        else 'cpu'
    )

    result_folder = (
        template_root / 'result' / 'stage_3_result' / args.dataset / args.experiment_name
    )
    result_folder.mkdir(parents=True, exist_ok=True)

    data = Dataset_Loader_Stage3(args.dataset).load()
    train_x = torch.from_numpy(data['train']['X'])
    train_y = torch.from_numpy(data['train']['y'])
    test_x = torch.from_numpy(data['test']['X'])
    training_data = TensorDataset(train_x, train_y)

    print('Dataset:', args.dataset.upper())
    print('Training examples:', len(training_data))
    print('Testing examples:', len(test_y := data['test']['y']))
    print('Input shape:', tuple(data['train']['X'].shape[1:]))
    print('Classes:', data['classes'])
    print('Device:', device)

    model = Method_CNN_Stage3(
        input_channels=data['channels'],
        number_of_classes=data['classes'],
        conv_channels=args.conv_channels,
        kernel_sizes=args.kernel_sizes,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        adaptive_pool_size=args.adaptive_pool_size,
    ).to(device)

    shuffle_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        training_data, batch_size=args.batch_size, shuffle=True, generator=shuffle_generator
    )
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    epoch_losses = []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = loss_function(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_y)

        average_loss = total_loss / len(training_data)
        epoch_losses.append(average_loss)
        print('Epoch', epoch + 1, 'of', args.epochs, '- loss:', round(average_loss, 4))

    model.eval()
    predicted_labels = []
    with torch.no_grad():
        for start in range(0, len(test_x), args.batch_size):
            batch_x = test_x[start:start + args.batch_size].to(device)
            predicted_labels.extend(model(batch_x).argmax(dim=1).cpu().tolist())

    evaluator = Evaluate_Classification_Stage3()
    evaluator.data = {'true_y': test_y, 'pred_y': np.asarray(predicted_labels)}
    scores = evaluator.evaluate()
    print('Test scores:', scores)

    plt.plot(range(1, args.epochs + 1), epoch_losses, marker='o')
    plt.title('{} CNN training loss ({})'.format(args.dataset.upper(), args.experiment_name))
    plt.xlabel('Epoch')
    plt.ylabel('Average training loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(result_folder / 'training_loss.png', dpi=150)
    plt.close()

    with (result_folder / 'training_losses.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['epoch', 'average_training_loss'])
        writer.writerows((epoch + 1, loss) for epoch, loss in enumerate(epoch_losses))

    result = {
        'dataset': args.dataset,
        'experiment': args.experiment_name,
        'conv_channels': ','.join(map(str, args.conv_channels)),
        'kernel_sizes': ','.join(map(str, args.kernel_sizes)),
        'hidden_dims': ','.join(map(str, args.hidden_dims)) or 'none',
        'dropout': args.dropout,
        'adaptive_pool_size': args.adaptive_pool_size,
        **scores,
    }
    with (result_folder / 'experiment_results.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(result))
        writer.writeheader()
        writer.writerow(result)

    with (result_folder / 'experiment_settings.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'random_seed', 'epochs', 'batch_size', 'learning_rate', 'optimizer',
            'loss_function', 'device', 'conv_channels', 'kernel_sizes', 'hidden_dims',
            'dropout', 'adaptive_pool_size',
        ])
        writer.writerow([
            args.seed, args.epochs, args.batch_size, args.learning_rate, 'Adam',
            'CrossEntropyLoss', str(device), result['conv_channels'], result['kernel_sizes'],
            result['hidden_dims'], args.dropout, args.adaptive_pool_size,
        ])

    print('Saved plot and experiment results to:', result_folder)


if __name__ == '__main__':
    main()
