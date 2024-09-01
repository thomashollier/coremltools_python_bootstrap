Barebones python script to run inference coreml models on M1 hardware

Production machines are usually large beefy rigs, running windows on intel CPUs and sporting fast NVidia cards. On the other hand, my day to day machine is a first generation apple M1 which continues to impress me years after it came out. Fast, efficient, cheap and elegant... Unfortunately, the biggest hurdle is the inability to run CUDA on it, and all the AI models are written for CUDA.

Apple has come out with coremltools to convert the main families of models and a few ready to use translated models. This script runs some of those models in python nin the quickest simplest way. It's the missing quickstart.

The available apple translated models can be downloaded from their web site: https://developer.apple.com/machine-learning/models/
