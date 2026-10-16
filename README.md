
# Edge Vision Inference Benchmark

An AI-assisted machine learning project exploring convolutional neural networks (CNNs), model evaluation, dynamic quantization, and CPU inference performance.

## Project Purpose

I used this project to develop a practical understanding of machine learning concepts relevant to AI/ML engineering and edge inference.

The project began with code generated with AI assistance. My work focused on setting up and executing the program on my laptop, studying its components, inspecting predictions and errors, running comparisons, interpreting measured results, and documenting what I learned.

This is an educational benchmarking project, not a production application or a deployment to Qualcomm hardware.


## Development Environment

This project was developed and executed locally on my Windows 11 laptop using Visual Studio Code as my integrated development
environment (IDE).

### Development Tools

| Component | Technology |
|---|---|
| Operating System | Windows 11 |
| IDE | Visual Studio Code |
| Programming Language | Python 3.13.14 |
| Python Environment | Virtual environment (.venv) |
| Machine Learning Framework | PyTorch 2.14.1+cpu |
| Numerical Computing | NumPy 2.5.3 |
| Machine Learning Utilities | scikit-learn 1.9.1 |
| Version Control | Git |
| Repository Hosting | GitHub |
| Command-Line Interface | VS Code integrated terminal / PowerShell |

### Local Project Setup

The project was organized in a dedicated folder containing the Python scripts, trained model files, experimental results, and
project documentation.

A Python virtual environment (.venv) was used to manage the project's dependencies separately from other Python projects
on my computer.

Visual Studio Code was used to examine the source code, execute Python scripts, review experiment outputs, and maintain the
project documentation.

The integrated terminal was used to run training and benchmarking commands, while JSON and CSV files were used to review and
compare experimental measurements.

### Development Workflow

My development workflow included:

1. Setting up the project folder and Python virtual environment.
2. Installing and verifying the required Python dependencies.
3. Running the AI-generated machine learning program locally.
4. Examining the CNN architecture and training procedures.
5. Reviewing model predictions and classification errors.
6. Running inference benchmarks and quantization comparisons.
7. Comparing the original CNN with a wider architecture.
8. Organizing experimental results into descriptive directories.
9. Documenting the methodology, measurements, findings, and limitations.
10. Preparing the project for publication through GitHub.

This workflow allowed me to gain hands-on experience with machine learning development tools while studying how model
architecture, quantization, and inference performance relate to practical engineering decisions.


## What the Project Does

- Trains and evaluates a CNN for handwritten-digit classification.
- Compares the CNN against a logistic regression baseline.
- Compares FP32 inference with dynamic INT8 quantization of the CNN's Linear layers.
- Measures serialized model-state size and CPU inference latency.
- Examines individual predictions, confidence scores, and classification errors.
- Compares the original CNN architecture with a wider CNN using validation results.

## Dataset and Evaluation Method

The project uses the scikit-learn Digits dataset, containing 1,797 grayscale images. Each image is 8 × 8 pixels and represents a handwritten digit from 0 through 9.

| Dataset Split | Images | Purpose |
|---|---:|---|
| Training | 1,077 | Learn model weights |
| Validation | 360 | Compare model settings and select a checkpoint |
| Test | 360 | Evaluate the selected model on held-out examples |

The dataset is split using stratification and random seed 42.

The CNN was trained for 30 epochs. The selected checkpoint first achieved its best recorded validation accuracy of 97.22% at epoch 18.

The test dataset was not used to select the checkpoint.

## CNN Model Architecture

The original CNN processes grayscale images using the following architecture:

    Input: 1 × 8 × 8
        |
    Conv2d: 1 to 8 channels
        |
    ReLU
        |
    MaxPool2d
        |
    Conv2d: 8 to 16 channels
        |
    ReLU
        |
    MaxPool2d
        |
    Flatten: 64 features
        |
    Linear: 64 to 64
        |
    ReLU
        |
    Linear: 64 to 10
        |
    Output: 10 class scores

Convolutional filters detect local image features such as edges, strokes, and curves.

Weight sharing allows the same filter to detect a learned pattern at different locations within an image.

The final Linear layers combine extracted features to produce scores for the ten digit classes.

Training uses cross-entropy loss and the Adam optimizer.

## Local Benchmark Results


The following measurements were collected by running the initial training and inference benchmark on my Windows 11
laptop. The results are documented in `results/initial_baseline/report.json`.


### Held-Out Test Performance

| Model | Test Accuracy | Macro F1 | Serialized State Size | Median Latency | p95 Latency |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 95.56% | 0.9546 | Not measured | Not measured | Not measured |
| FP32 CNN | 95.83% | 0.9574 | 27,597 bytes | 0.0807 ms | 0.1066 ms |
| CNN with INT8 Linear Layers | 96.11% | 0.9604 | 14,783 bytes | 0.2684 ms | 0.3076 ms |

These results were measured using a held-out test set containing 360 images.

### Quantization Analysis

Dynamic INT8 quantization was applied to the CNN's Linear layers. The convolutional layers remained in FP32.

Quantization reduced the serialized model-state size from 27,597 bytes to 14,783 bytes, a reduction of approximately 46.4%.

However, median CPU inference latency increased from 0.0807 milliseconds to 0.2684 milliseconds.

The quantized model was approximately 3.33 times slower than the FP32 model in this benchmark.

The quantized CNN achieved 96.11% test accuracy compared with 95.83% for the FP32 CNN.

This difference represents one additional correct prediction out of 360 test images and does not establish that quantization generally improves accuracy.

### Engineering Interpretation

The experiment demonstrated that reducing model precision can decrease storage requirements without necessarily improving inference speed.

For this small CNN, quantization-related overhead may have contributed to the slower CPU inference measurements.

These findings reinforced the importance of measuring performance rather than assuming that a smaller model will run faster.

The results are specific to the tested model, dataset, processor, and software environment.

## Benchmark Environment

| Component | Recorded Configuration |
|---|---|
| Operating System | Windows 11 |
| Processor | Intel64 Family 6 Model 186 Stepping 2 |
| Python | 3.13.14 |
| PyTorch | 2.14.1+cpu |
| NumPy | 2.5.3 |
| scikit-learn | 1.9.1 |
| Quantization Engine | oneDNN |
| CPU Threads | 1 |
| Batch Size | 1 |
| Warmup Calls | 30 |
| Benchmark Rounds | 5 |
| Timed Calls per Round | 200 |

The latency measurements cover CPU model-forward execution.

They exclude image preprocessing, loading the model from disk, and other application-level overhead.

Performance may differ on other processors or software configurations.

## Architecture Comparison

A separate experiment compared the original CNN architecture with a wider CNN containing additional convolutional channels.

### Validation Results

| Architecture | Validation Accuracy |
|---|---:|
| Original CNN | 97.22% |
| Wider CNN | 98.06% |

The wider CNN achieved higher validation accuracy in this experiment.

Increasing the number of channels allows the model to learn additional feature maps, but it also increases the number of learned parameters and computational requirements.

This comparison helped demonstrate the trade-off between model capacity and complexity.

These are validation results, not held-out test results, and should not be directly compared with the test accuracies 
reported in the initial benchmark.

The saved comparison artifacts are organized under `results/architecture_comparison/`.

## Prediction Inspection and Error Analysis

The project includes an inference demonstration for loading a trained CNN and making predictions on individual handwritten-digit images.

The inference workflow involves:

1. Loading the trained model and its saved weights.
2. Preparing an image tensor with the required dimensions.
3. Passing the image through the CNN.
4. Producing ten class scores.
5. Applying softmax to obtain class probabilities.
6. Selecting the class with the highest score.
7. Comparing the prediction with the known label.

During inference, the model uses its previously learned weights without updating them.

The project also includes inspection of incorrectly classified images.

Reviewing these mistakes helped me understand that a model can produce a high softmax probability while still making an incorrect prediction.

Softmax probabilities reflect the model's relative scores and do not guarantee that its prediction is correct.

## Project Files and Results

The experiment outputs are organized into the following directories:

    results/
        initial_baseline/
        inference_benchmark/
        architecture_comparison/


### Initial Baseline

The initial training and evaluation results are stored in `results/initial_baseline/`.

The main report, `report.json`, contains the FP32 and dynamically quantized CNN results, including test accuracy, 
macro F1 scores, serialized model sizes, and CPU inference latency.

### Inference Benchmark

`results/inference_benchmark/`

Contains additional CPU inference benchmark runs used to investigate measurement consistency and performance differences.

These repeated measurements are kept separate from the initial benchmark results.

### Architecture Comparison

`results/architecture_comparison/`

Contains results from comparing the original CNN against a wider CNN architecture.

The comparison uses validation accuracy to evaluate the effect of increasing convolutional channels.

## Running the Project

The project was developed and executed on Windows using Visual Studio Code, a Python virtual environment, and CPU-enabled PyTorch.


### Training and Evaluation

To train and evaluate the CNN, run the following command from the project root in the VS Code terminal:
  
    .\.venv\Scripts\python.exe train.py --epochs 30 --output results/initial_baseline

This command trains the CNN for 30 epochs and saves the evaluation results in `results/initial_baseline/`.

The command assumes the required dependencies are installed in the project's Python virtual environment (`.venv`).

Note: Running this command again may overwrite previously saved results in the specified output directory.


## Key Learning Outcomes

### Training, Validation, and Testing

Training data is used to update model weights.

Validation data is used to evaluate model settings and select a checkpoint.

Test data is reserved for evaluating the selected model on previously unseen examples.

Keeping these roles separate helps prevent test results from influencing model selection.

### Convolution and Weight Sharing

Convolutional filters detect local patterns in images.

Weight sharing allows the same filter to recognize patterns in different image locations without requiring separate learned weights for each position.

### Model Complexity

Increasing convolutional channels can improve a model's ability to learn features.

However, additional channels increase model complexity and may require more computation and memory.

### Quantization

Quantization can reduce model storage requirements, but it does not guarantee faster inference.

Actual performance depends on the architecture, hardware, software implementation, and quantization method.

### Model Evaluation

Accuracy alone does not provide a complete picture of model performance.

Macro F1, model size, inference latency, and classification errors provide additional information about how a model performs.

### Prediction Confidence

A high softmax probability does not guarantee that a prediction is correct.

Inspecting individual mistakes helps reveal limitations that may not be obvious from aggregate accuracy.

## Project Limitations

- The scikit-learn Digits dataset is small and contains low-resolution images.
- Results may not generalize to more complex real-world computer vision tasks.
- Dynamic INT8 quantization was applied only to Linear layers, not the complete CNN.
- CPU inference measurements exclude preprocessing, model loading, and application overhead.
- Benchmarks were performed on a Windows laptop rather than dedicated edge AI hardware.
- The project was not deployed to a Qualcomm processor, NPU, or embedded system.
- Small differences in test accuracy should not be interpreted as proof that one approach consistently generalizes better.
- Benchmark results may vary across hardware and software environments.

## Potential Future Improvements

Future development could include:

- Testing additional CNN architectures and larger datasets.
- Repeating training experiments with multiple random seeds.
- Measuring complete end-to-end inference latency.
- Comparing additional quantization techniques.
- Evaluating memory consumption and power requirements.
- Investigating deployment to supported edge AI hardware.
- Exploring hardware-specific inference optimization.

These are potential improvements and are not presented as completed work.

## AI Assistance and My Contributions

### AI Assistance

This project was developed with substantial assistance from an AI coding assistant.

The initial program, including its machine learning implementation, was generated using AI.

AI was also used to support explanations, troubleshooting, experiment planning, documentation, and interpretation of results.

I do not represent the original code as software I independently wrote from scratch.

### My Contributions

I used the AI-generated program as a structured learning project to develop practical machine learning knowledge.

My work included:

- Configuring the development environment and executing the program on my Windows laptop.
- Studying the CNN architecture, training workflow, and evaluation process.
- Learning how convolution, shared kernels, feature maps, pooling, and Linear layers work.
- Examining the differences between training, validation, testing, and inference.
- Running and reviewing FP32 and dynamically quantized inference experiments.
- Comparing model accuracy, serialized state size, and CPU inference latency.
- Investigating classification mistakes and softmax confidence scores.
- Comparing the original CNN architecture against a wider CNN.
- Interpreting experimental results and identifying performance trade-offs.
- Organizing experiment outputs and documenting findings and limitations.

The purpose of this project was to build my understanding of machine learning engineering concepts through hands-on experimentation.

It demonstrates my ability to work with an existing machine learning program, execute experiments, interpret measured results, and explain the engineering concepts involved.

It is not intended to imply independent authorship of the AI-generated implementation or production-level machine learning deployment experience.

## References

- [scikit-learn Digits Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch Quantization Documentation](https://pytorch.org/docs/stable/quantization.html)
