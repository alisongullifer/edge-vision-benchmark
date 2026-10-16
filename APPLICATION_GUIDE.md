# Qualcomm Application and Seven Day Project Guide

## Seven day build plan

Allow approximately 2–3 focused hours per day, adjusting to your learning pace. Finish the core experiment before adding deployment work.

| Day | Work | Evidence and understanding |
| --- | --- | --- |
| 1 | Set up VS Code and the environment; run tests and the experiment. Read each function. | Record versions and explain inputs, labels, training, validation, and test sets. |
| 2 | Learn a neuron, weights, bias, ReLU, cross-entropy, gradient descent, and backpropagation using this model. | Trace one batch through the model and explain why training updates weights. |
| 3 | Understand convolution, shared kernels, pooling, channels, logits, and tensor shapes. | Draw or document the shape after each layer. Explain why a CNN suits images. |
| 4 | Add your own inference script using the FP32 saved state and a held-out sample; display predicted and actual labels. | Verify model reloading and show several correct and incorrect predictions. |
| 5 | Investigate quantization and benchmarking. Repeat timing without retraining and record background activity. | Explain reduced bytes versus slower inference; label only Linear layers as INT8. |
| 6 | Add error analysis and a validation-only architecture experiment, such as changing channel counts. | Compare validation performance and parameter count; explain the tradeoff and avoid tuning on test results. |
| 7 | Clean the README, commit your contributions, upload your measured results, and practice a short demo. | Be able to explain every resume bullet and identify limitations. |

CNNs are the focus of this project. RNNs and LSTMs process sequences and maintain recurrent state; this project does not demonstrate experience with either. Do not add those skills based only on reading their definitions.

## Resume project entry after you have reproduced and extended it

Replace the general portfolio project entry with this project when it is completed. Put it first within Technical Projects, include the actual repository URL, and add PyTorch/NumPy/scikit-learn only when you can use and explain them.

Suggested title: **Image Classification and CPU Inference Benchmark | Python, PyTorch, NumPy, scikit-learn**

- Implemented and evaluated a compact CNN for handwritten-digit classification, using stratified training, validation, and test splits and a logistic-regression baseline.
- Compared FP32 inference with dynamic INT8 quantization of fully connected layers; measured held-out accuracy, macro F1, serialized model size, and median/p95 CPU latency.
- Documented [your measured accuracy], [your size reduction], and [your latency change] on [your CPU], including error analysis and limitations.

Use “reproduced and extended” instead of “implemented” if that better describes your contributions. Fill brackets with your actual run; do not copy the reference CPU metrics as if you measured them yourself. Never claim Qualcomm device deployment, a fully INT8 CNN, or faster inference unless your evidence supports it.

## Interview preparation

Be ready to answer: Why use a baseline? Why keep a test set separate? What does a convolution learn? What is overfitting? Why does quantization reduce size? Why can a smaller model run slower? What does p95 mean? What changes on an NPU? What did you personally add, and where did AI assist?

One strong week can produce a credible focused project. It will not establish expertise across the posting's entire preferred-skills list. Your strongest application combines honest software foundations, reproducible measurements, and a project you can explain clearly.
