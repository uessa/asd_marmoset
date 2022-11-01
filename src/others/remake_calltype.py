# -*- coding: utf-8 -*-
import numpy as np
import pathlib
from multiprocessing import Pool
import glob
import re

correct_label = {'B': 'Breath',
                'Bbreath': 'Breath',
                'Brearth': 'Breath',
                'Breasth': 'Breath',
                'Breath': 'Breath',
                'Breathbreath': 'Breath',
                'Breathg': 'Breath',
                'Breathj': 'Breath',
                'Breathy': 'Breath',
                'Breatn': 'Breath',
                'Breatrh': 'Breath',
                'Breatyh': 'Breath',
                'Brerath': 'Breath',
                'Brwath': 'Breath',
                'Btreath': 'Breath',
                'Bvreath': 'Breath',
                'C Ry': 'Cry',
                'Chatter': 'Chatter',
                'Chirp': 'Chirp',
                'Chirpchirp': 'Chirp',
                'Cough': 'Cough',
                'Coughcough': 'Cough',
                'Cry': 'Cry',
                'E': 'Ek',
                'Ek': 'Ek',
                'Ek-Tsik': 'Ek',
                'Ekek': 'Ek',
                'Ekj': 'Ek',
                'Ex': 'Ek',
                'Hee': 'Phee',
                'Intermittentphee': 'Others',
                'Others': 'Others',
                'Phee': 'Phee',
                'Phee-Trill': 'Phee-Trill',
                'Pheecry': 'Phee',
                'PheeCry': 'Phee',
                'Pheee': 'Phee',
                'Pheephee': 'Phee',
                'Phees': 'Phee',
                'Reath': 'Breath',
                'See': 'Others',
                'Sneeze': 'Others',
                'Snooze': 'Others',
                'Treill': 'Trill',
                'Treill-Phee': 'Trill-Phee',
                'Trill': 'Trill',
                'Trill-': 'Trill-Phee',
                'Trill-Hee': 'Trill-Phee',
                'Trill-Phee': 'Trill-Phee',
                'Trill-phee': 'Trill-Phee',
                'trill-Phee': 'Trill-Phee',
                'Trilll': 'Trill',
                'Tsik': 'Tsik',
                'Tsiktsik': 'Tsik',
                'Ttwitter': 'Twitter',
                'Twiitter': 'Twitter',
                'Twiter': 'Twitter',
                'Twitteer': 'Twitter',
                'Twitter': 'Twitter',
                'Twitters': 'Twitter',
                'Twittetr': 'Twitter',
                'Twittrer': 'Twitter',
                'Twittter': 'Twitter',
                'Twitttter': 'Twitter',
                'Unk': 'Unknown',
                'Unkinown': 'Unknown',
                'unknown': 'Unknown',
                'Unknow': 'Unknown',
                'Unknown': 'Unknown',
                'Unknownhow': 'Unknown',
                'Unknownbreath': 'Unknown'}

def find_calltype(list_text):
    type_call = ['Noise']
    for data in list_text:
        ground_truth_data = np.loadtxt(data, dtype=str, delimiter="\t")
        for t in range(ground_truth_data.shape[0]):
            if ground_truth_data[t, 1] == 'Call' or ground_truth_data[t, 1] == 'Calls':
                if not (ground_truth_data[t, 2] in type_call):
                    type_call.append(ground_truth_data[t, 2])
    print("type_call=",type_call)
    print("len(type_call)=",len(type_call))

def remake_calltype(list_text):
    type_call = ['Noise']
    for data in list_text:
        ground_truth_data = np.loadtxt(data, dtype=str, delimiter="\t")
        for t in range(ground_truth_data.shape[0]):
            if ground_truth_data[t, 1] == 'Call' or ground_truth_data[t, 1] == 'Calls':
                # print("pass")
                if not (ground_truth_data[t, 2] in type_call):
                     call = ground_truth_data[t, 2]
                     call = call.replace(' ', '') # "Unk nown " -> "Unknown"
                     call = call.upper() # phee-trill -> PHEE-TRILL
                     call = call.title() # PHEE-TRILL -> Phee-Trill
                     ground_truth_data[t, 2] = correct_label[call]
        np.savetxt(data, ground_truth_data, fmt="%s", delimiter="\t")
        # print("save")
    print("savetxt")
            


if __name__ == "__main__":

    path = "/home/muesaka/projects/marmoset/raw/marmoset_11vpa_text"

    path_text1 = pathlib.Path(path)
    list_text1 = path_text1.glob("*.txt")
    find_calltype(list_text1)

    path_text1 = pathlib.Path(path)
    list_text1 = path_text1.glob("*.txt")
    remake_calltype(list_text1)

    path_text1 = pathlib.Path(path)
    list_text1 = path_text1.glob("*.txt")
    find_calltype(list_text1)



