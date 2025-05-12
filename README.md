# Neural Network Architecture Benchmarking for Time Series Classification

![Time Series Classification Datasets](https://www.timeseriesclassification.com/dataset.php)
![ConfigSpace](https://github.com/automl/ConfigSpace/)

This project benchmarks five neural network architectures on multiple time series classification datasets. It produces different hyperparameter configurations for each architecture.

## Overview

Five standard neural network architectures:
1. Fully Connected Neural Network (FCNN)
2. Convolutional Neural Network (CNN)
3. Long Short-Term Memory (LSTM)
4. Gated Recurrent Unit (GRU)
5. Transformer

For each architecture:
- defines a configuration space of hyperparameters
- randomly samples 5 different configurations
- trains and validates each configuration on all datasets
- saves comprehensive results to JSON files

## Datasets

The project uses datasets from the [UCR Time Series Classification Archive](https://www.timeseriesclassification.com):

1. classification_ozone (custom dataset, not included in the website)
2. Adiac
3. ArrowHead
4. Beef
5. BeetleFly
6. BirdChicken
7. Car
8. CBF
9. ChlorineConcentration
10. CinCECGTorso
11. FiftyWords