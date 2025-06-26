# Neural Network Architecture Benchmarking for Time Series Classification

[Time Series Classification Datasets](https://www.timeseriesclassification.com/dataset.php)
[ConfigSpace](https://github.com/automl/ConfigSpace/)

This project benchmarks five neural network architectures on multiple time series classification datasets. It produces 100 different hyperparameter configurations for each dataset and architecture.

## Overview

Five standard neural network architectures:
1. Fully Connected Neural Network (FCNN)
2. Convolutional Neural Network (CNN)
3. Long Short-Term Memory (LSTM)
4. Gated Recurrent Unit (GRU)
5. Transformer

For each architecture:
- defines a configuration space of hyperparameters
- randomly samples 100 different configurations
- trains and validates each configuration on all datasets using 5 folds
- saves comprehensive results to JSON files

## Datasets

The project uses datasets from the [UCR Time Series Classification Archive](https://www.timeseriesclassification.com):

1. classification_ozone (custom dataset, not included in the website)
2. AbnormalHeartbeat
3. Adiac
4. ArrowHead
5. Beef
6. BeetleFly
7. BinaryHeartbeat
8. BirdChicken
9. Car
10. CatsDogs
11. CBF
12. ChlorineConcentration
13. CinCECGTorso
14. CounterMovementJump
15. DucksAndGeese
16. EigenWorms
17. FaultDetectionB
18. FiftyWords
19. HouseTwenty
20. RightWhaleCalls
21. KeplerLightCurves
