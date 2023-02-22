# MMNet

## Introduction

The official implementation of "MMNET: Multi-band Multi-scale Network for Singing Melody Extraction from Polyphonic Music, whose manuscript is under submission for interspeech 2023.

We propose a more powerful singing melody extractor named multi-band multi-scale network (MMNet) for polyphonic music. Experimental results show that our proposed MMNet achieves promising performance compared with existing state-of-the-art methods, while keeping with a small number of network parameters.

<p align="center">
<img src="fig/arch.png" align="center" alt="MMNet Architecture" width="100%"/>
</p>

## Getting Started

### Download Datasets

* [MIR-1k](https://sites.google.com/site/unvoicedsoundseparation/mir-1k)
* [ADC 2004 & MIREX05](https://labrosa.ee.columbia.edu/projects/melody/)
* [MedleyDB](https://medleydb.weebly.com/)

After downloading the data, use the txt files in the data folder, and process the CFP feature by [feature_extraction.py](feature_extraction.py).

## Model implementation

Refer to the file: [mmnet.py](model/mmnet.py)

## Result

### Prediction result

The visualization illustrates that our proposed MMNet can avoid the octave errors and reduce the melody detection errors.

<p align="center">
<img src="fig/estimation.png" align="center" alt="Estimation" width="60%"/>
</p>

### Comprehensive result

The scores here are either taken from their respective papers or from the result implemented by us. Experimental results show that our proposed MMNet achieves promising performance compared with existing state-of-the-art methods.

<p align="center">
<img src="fig/result.png" align="center" alt="Result" width="50%"/>
</p>

The model code has been uploaded.
