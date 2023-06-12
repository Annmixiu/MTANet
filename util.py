"""

utils file

This file contains useful common methods

"""
import os
import numpy as np
import torch
import mir_eval
import config

def index2centf(seq, centfreq):#seq is 128frame_f0 sequence
    centfreq[0] = 0
    re = np.zeros(len(seq))#re:[0,0,...,0],num_dim=128
    for i in range(len(seq)):# i=0,1,...,127
        for j in range(len(centfreq)):# j=0,1,...,360
            if seq[i] < 0.1:#if label f0<0.1
                re[i] = 0
                break
            elif centfreq[j] > seq[i]:
                re[i] = j
                break
    return re  


def freq2octave(freq):
    if freq < 1.0 or freq > 2050:
        return config.octave_class
    else:
        return int(np.round(69 + 12 * np.log2(freq/440)) // 12) 

def freq2tone(freq):
    if freq < 1.0 or freq > 2050:
        return config.tone_class
    else:
        return int(np.round(69 + 12 * np.log2(freq/440)) % 12) 

def tofreq(tone, octave):
    if tone >= config.tone_class or octave >= config.octave_class or octave < 2:
        return 0.0
    else:
        return 440 * 2 ** ((12 * octave + tone * 12 / config.tone_class - 69) / 12)


def pos_weight(data, freq_bins):
    frames = data.shape[-1]
    non_vocal = float(len(data[data == 0]))
    vocal = float(data.size - non_vocal)
    z = np.zeros((freq_bins, frames))
    z[1:,:] += (non_vocal / vocal)
    z[0,:] += vocal / non_vocal
    print(non_vocal, vocal)
    return torch.from_numpy(z).float()

def freq2octave(freq):
    if freq < 1.0 or freq > 1990: #set the range of frequence
        return 0
    pitch = round(69 + 12 * np.log2(freq / 440)) #semit =  69 + 12*log2(f/440)
    return int(pitch // 12) #octave = semit/12

def compute_roa(pred, gd):
    pred = pred[gd > 0.1]
    gd = gd[gd > 0.1] #the range of frequence
    pred = np.array([freq2octave(d) for d in pred])#频率转换为八度组成的数组(pre)
    gd = np.array([freq2octave(d) for d in gd])#频率转换为对应八度组成的数组(gd)
    return np.sum(pred == gd) / len(pred) #计算预测准确的个数/样本数


def melody_eval(pred, gd):
    ref_time = np.arange(len(gd)) * 0.01
    ref_freq = gd

    est_time = np.arange(len(pred)) * 0.01
    est_freq = pred

    output_eval = mir_eval.melody.evaluate(ref_time,ref_freq,est_time,est_freq) #output_eval={}
    VR = output_eval['Voicing Recall']*100.0 
    VFA = output_eval['Voicing False Alarm']*100.0
    RPA = output_eval['Raw Pitch Accuracy']*100.0
    RCA = output_eval['Raw Chroma Accuracy']*100.0
    ROA = compute_roa(est_freq, ref_freq) * 100.0
    OA = output_eval['Overall Accuracy']*100.0
    eval_arr = np.array([VR, VFA, RPA, RCA, ROA, OA])
    return eval_arr

def tonpy_fn(batch):# integrate bs sets of data into one set of data
    dict_key = batch[0].keys()
    #batch[0]={'cfp':(17/25/14,3,360,128), 'tcfp':(17/25/14,3,360,128), 'gd':(17/25/14,128), 'length':2109/3197/1748}
    #batch[0]：get per_batch data
    output_batch = {}
    for dk in dict_key:# dk=cfp, tcfp, gd, length
        output_batch[dk] = np.array([d[dk] for d in batch])
        #列表生成式:output_batch[cfp] = np.array([d[cfp] for d in batch])
        #d遍历每个batch中的所有数据，将d中cfp(key)对应的value生成为数组并赋给output_batch字典中cfp(key)对应的value
        #将per_batch中的不同key下value分别传给了output_batch对应key下的value实现了多组数据整合为一组数据
    return output_batch

def loss_avg(x_list):
    total = 0
    for i in range(len(x_list)):
        result = total + x_list[i]
        result = result / len((x_list))
    return result
