# MMNet

## Introduction

The official implementation of "[MMNET: MULTI-BAND MULTI-SCALE NETWORK FOR SINGING MELODY EXTRACTION FROM POLYPHONIC MUSIC], in ICASSP 2023

We propose a more powerful singing melody extractor named multi-band multi-scale network (MMNet) for polyphonic music. Experimental results show that our proposed MMNet achieves promising performance compared with existing state-of-the-art methods, while keeping with a small number of network parameters.

## Getting Started

### Download Datasets

* [MIR-1k](https://sites.google.com/site/unvoicedsoundseparation/mir-1k)
* [ADC 2004 & MIREX05](https://labrosa.ee.columbia.edu/projects/melody/)
* [MedleyDB](https://medleydb.weebly.com/)

After downloading the data, use the txt files in the data folder, and process the CFP feature by [feature_extraction.py](feature_extraction.py).

## Model implementation

Refer to the file: [mmnet.py](model/mmnet.py)

