# MMNet

## Introduction

## Getting Started

### Download Datasets

* [MIR-1k](https://sites.google.com/site/unvoicedsoundseparation/mir-1k)
* [ADC 2004 & MIREX05](https://labrosa.ee.columbia.edu/projects/melody/)
* [MedleyDB](https://medleydb.weebly.com/)

After downloading the data, use the txt files in the data folder, and process the CFP feature by [feature_extraction.py](feature_extraction.py).

### Overwrite the Configuration

The *config.py* contains all configurations you need to change and set.

### Train and Evaluation

```
python main.py train

python main.py test
```

## Produce the Estimation Digram

Uncomment the write prediction in [mmnet.py](model/mmnet.py)

## Model Checkpoints

## Citing 
```
```

