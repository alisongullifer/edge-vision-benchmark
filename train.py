"""CPU CNN experiment with a held-out test set and honest latency measurements."""
import argparse
import copy
import csv
import io
import json 
import platform
import time
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn


class SmallCNN(nn.Module): # creates the CNNs layers (defines model's structure)
    def __init__(self): # initializes the CNN model by defining its layers and structure (builds the CNN architecture)
        super().__init__() # initializes the parent class (nn.Module) to inherit its properties and methods
        
        self.features = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), # 1st convolution
                                      nn.ReLU(), # changes negative values to zero without changing shape
                                      nn.MaxPool2d(2), 
                                      nn.Conv2d(8, 16, 3, padding=1), # 2nd convolution
                                      nn.ReLU(), 
                                      nn.MaxPool2d(2)) 
        # convolutional layers with ReLU activation and max pooling to extract features from the input images

        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))
        # combines the extracted features into a final output layer with 10 classes (digits 0-9)

    def forward(self, x): # passes the input through the CNN layers and returns the output
        return self.head(self.features(x)) 
        # returns logits (unnormalized scores (not yet probabilities)) of the head after passing through the feature extractor


def split_data(seed): 
    # loads and prepares the digits dataset, splits it into training, validation, and test sets, and returns the data and indices
    data = load_digits()
    indices = np.arange(len(data.target))
    train_val, test = train_test_split(indices, test_size=.2, stratify=data.target, random_state=seed)
    train, val = train_test_split(train_val, test_size=.25, stratify=data.target[train_val], random_state=seed)
    # pixel range is specified by the dataset, so this uses no learned test statistics.
    x = torch.tensor(data.images[:, None] / 16.0, dtype=torch.float32) 
    # normalizes the pixel values of the images to the range [0, 1] by dividing by 16.0 and adds a channel dimension (1) to the images
    y = torch.tensor(data.target, dtype=torch.long)
    return x, y, train, val, test


@torch.inference_mode() #runs in inference mode, which disables gradient computation and reduces memory usage
def predict(model, x): #runs the model on the input data and returns the predicted class labels
    return model(x).argmax(1).numpy() 
    # returns the index of the maximum value along dimension 1 (the predicted class) as a numpy array


@torch.inference_mode() 
def latency(model, sample, repeats): #times the model's forward pass on a sample input and returns latency statistics
    for _ in range(30):
        model(sample)
    # repeated rounds; all variants see identical inputs, thread count and timing boundaries.
    rounds = []
    for _ in range(5):
        elapsed = []
        for _ in range(repeats):
            start = time.perf_counter_ns() #records the current time in nanoseconds before the model's forward pass
            model(sample)
            elapsed.append((time.perf_counter_ns() - start) / 1e6) #records the elapsed time in milliseconds after the model's forward pass
        rounds.append(elapsed)
    a = np.array(rounds)
    return {'latency_median_ms': float(np.median(a)), 'latency_p95_ms': float(np.percentile(a, 95)),
            'round_medians_ms': np.median(a, axis=1).tolist()}


def state_bytes(model): # returns the size (measurement) of the model's serialized state dictionary in bytes
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream) #saves the model's state dictionary (weights and biases) to an in-memory byte stream
    return len(stream.getvalue())


def run(args): # coordinates the training, evaluation, quantization, and reporting of the CNN model ( runs whole experiment)
    if args.epochs < 1 or args.repeats < 1:
        raise ValueError('epochs and repeats must be positive')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    x, y, train, val, test = split_data(args.seed)
    model = SmallCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    best_acc = -1
    history = []
    for epoch in range(args.epochs):
        model.train()
        order = torch.tensor(train)[torch.randperm(len(train))]
        losses = []
        for batch in order.split(64):
            optimizer.zero_grad() 
            # clears the gradients of the model's parameters to zero before computing the gradients for the current batch

            loss = nn.functional.cross_entropy(model(x[batch]), y[batch]) 
            # computes the cross-entropy loss between the model's predictions and the true labels for the current batch

            loss.backward() 
            # calculates the gradients of the loss with respect to the model's parameters (backpropagation) uses chain rule

            optimizer.step() # updates
            # adjusts the model's parameters based on the computed gradients to minimize the loss

            losses.append(loss.item())
        model.eval()
        val_acc = accuracy_score(y[val].numpy(), predict(model, x[val]))
        history.append({'epoch': epoch + 1, 'train_loss': float(np.mean(losses)), 'validation_accuracy': val_acc})
        if val_acc > best_acc:
            best_acc, best_epoch, best_state = val_acc, epoch + 1, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    baseline = LogisticRegression(max_iter=1500, random_state=args.seed)
    baseline.fit(x[train].flatten(1).numpy(), y[train].numpy())
    # quantizes ONLY Linear layers; convolution layers remain FP32.
    engines = torch.backends.quantized.supported_engines
    engine = next((e for e in ['x86', 'fbgemm', 'onednn', 'qnnpack'] if e in engines), None)
    if engine is None:
        raise RuntimeError(f'No supported quantization engine: {engines}')
    torch.backends.quantized.engine = engine
    quantized = torch.ao.quantization.quantize_dynamic(copy.deepcopy(model), {nn.Linear}, dtype=torch.qint8)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output / 'cnn_fp32.pt')
    torch.save(quantized.state_dict(), args.output / 'cnn_linear_int8.pt')
    target = y[test].numpy()
    baseline_pred = baseline.predict(x[test].flatten(1).numpy())
    results = []
    matrices = {'logistic_regression': confusion_matrix(target, baseline_pred).tolist()}
    for name, variant in [('cnn_fp32', model), ('cnn_linear_int8', quantized)]:
        pred = predict(variant, x[test])
        row = {'model': name, 'accuracy': accuracy_score(target, pred),
               'macro_f1': f1_score(target, pred, average='macro'),
               'serialized_state_bytes': state_bytes(variant), **latency(variant, x[test[:1]], args.repeats)}
        results.append(row)
        matrices[name] = confusion_matrix(target, pred).tolist()
    # reload test catches artifact/model mismatches without loading external pickle files
    loaded = SmallCNN().eval()
    loaded.load_state_dict(torch.load(args.output / 'cnn_fp32.pt', weights_only=True))
    assert np.array_equal(predict(loaded, x[test]), predict(model, x[test]))
    report = {'dataset': 'sklearn digits: 8x8 grayscale, 1797 samples, 10 classes',
              'seed': args.seed, 'split_counts': {'train': len(train), 'validation': len(val), 'test': len(test)},
              'split_indices': {'train': train.tolist(), 'validation': val.tolist(), 'test': test.tolist()},
              'selected_epoch': best_epoch, 'validation_accuracy': best_acc,
              'baseline': {'model': 'logistic_regression', 'accuracy': accuracy_score(target, baseline_pred),
                           'macro_f1': f1_score(target, baseline_pred, average='macro')},
              'environment': {'platform': platform.platform(), 'processor': platform.processor(),
                              'python': platform.python_version(), 'torch': str(torch.__version__),
                              'numpy': np.__version__, 'sklearn': sklearn.__version__, 'threads': 1,
                              'quantization_engine': engine, 'batch_size': 1, 'warmups': 30,
                              'rounds': 5, 'repeats_per_round': args.repeats},
              'results': results, 'confusion_matrices': matrices, 'training_history': history,
              'limitations': ['Only Linear layers quantized; not a fully INT8 CNN.',
                              'CPU model-forward latency excludes preprocessing and model loading.',
                              'Small educational dataset; no Qualcomm hardware or NPU deployment.',
                              'No guaranteed speedup; quantization overhead can dominate small models.']}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2))
    fields = ['model', 'accuracy', 'macro_f1', 'serialized_state_bytes', 'latency_median_ms', 'latency_p95_ms']
    with (args.output / 'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(results)
    print(json.dumps({'baseline': report['baseline'], 'results': results}, indent=2))
    return report


if __name__ == '__main__': # checks if the script is being run as the main program and not imported as a module
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--repeats', type=int, default=200)
    parser.add_argument('--output', type=Path, default=Path('results/local'))
    run(parser.parse_args())
