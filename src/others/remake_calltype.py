# -*- coding: utf-8 -*-
#-------------------------------------#
# 
#
#-------------------------------------#
import numpy as np
import pathlib
from multiprocessing import Pool
import glob
import re

correct_label = {
    # "Noise": " ",
    "Twitter": "Twitter",
    "Trill": "Trill",
    "Trill-phee": "Trill-Phee",
    "Phee": "Phee",
    "Cough": "Cough",
    "Tsik": "Tsik",
    "Cry": "Cry",
    "Unknown": "Unknown",
    "Ek": "Ek",
    "Unknown ": "Unknown",
    "Twitter ": "Twitter",
    "Trill-Phee": "Trill-Phee",
    "Phee-Trill": "Phee-Trill",
    "Twitter  ": "Twitter",
    "Phee-trill": "Phee-Trill",
    "unknown": "Unknown",
    "cough": "Cough",
    "Phees": "Phee",
    "phee": "Phee",
    "twitter": "Twitter",
    "Breath": "Breath",
    "Snooze": "Sneeze",
    "Chatter": "Chatter",
    " Phee": "Phee",
    "Unkinown": "Unknown",
    "breath": "Breath",
    "Twittter": "Twitter",
    "Breatrh": "Breath",
    "trill-Phee": "Trill-Phee",
    "Treill": "Trill",
    "Ex": "Ek",
    "Twitttter": "Twitter",
    "Twittetr": "Twitter",
    "B": "Breath",
    "Ek-Tsik": "Ek-Tsik",
    "Unknow": "Unknown",
    "Breathg": "Breath",
    "Trilll": "Trill",
    "Twittrer": "Twitter",
    "Breath ": "Breath",
    "Phee ": "Phee",
    "Breatn": "Breath",
    "Brwath": "Breath",
    "brerath": "Breath",
    "Intermittent phee": "Intermittent Phee",
    "Ek ": "Ek",
    "Pheee": "Phee",
    "Sneeze": "Sneeze",
    "EkEk": "Ek",
    "Brearth": "Breath",
    "BVreath": "Breath",
    "Breatyh": "Breath",
    "Chirp": "Chirp",
    "tsik": "Tsik",
    "Breasth": "Breath",
    "Ekj": "Ek",
    "ek": "Ek",
    "PheeCry": "Cry",
    "Treill-Phee": "Trill-Phee",
    "UnknownBreath": "Breath",
    "Breathy": "Breath",
    "Trill-": "Trill",
    "Ttwitter": "Twitter",
    "Breathj": "Breath",
    "Trill-hee": "Trill-Phee",
    "Twitters ": "Twitter",
    "EK": "Ek",
    "See": "Phee",
    "reath": "Breath",
    "E": "Ek",
    "PheePhee": "Phee",
    "BreathBreath": "Breath",
    "ChirpChirp": "Chirp",
    "Btreath": "Breath",
    "Twitteer": "Twitter",
    "CoughCough": "Cough",
}

calls = {
    "Twitter",
    "Trill",
    "Trill-Phee",
    "Phee",
    "Cough",
    "Tsik",
    "Cry",
    "Unknown",
    "Ek",
    "Phee-Trill",
    "Breath",
    "Sneeze",
    "Chatter",
    "Ek-Tsik",
    "Intermittent Phee",
    "Chirp",
    }

def find_calltype(list_text):
    type_call = ['Noise']
    for data in list_text:
        ground_truth_data = np.loadtxt(data, dtype=str, delimiter="\t")
        for t in range(ground_truth_data.shape[0]):
            if ground_truth_data[t, 1] == 'Call' or ground_truth_data[t, 1] == 'Calls':
                if not (ground_truth_data[t, 2] in type_call):
                    type_call.append(ground_truth_data[t, 2])
                    # print("\"{}\": \" \",".format(ground_truth_data[t, 2]))
                    # print(ground_truth_data[t,0], ground_truth_data[t, 3])
                    # print(data)
                    # print("")
                    print(ground_truth_data[t,2])        

def remake_calltype(list_text):
    type_call = ['Noise']
    for data in list_text:
        ground_truth_data = np.loadtxt(data, dtype=str, delimiter="\t")
        for t in range(ground_truth_data.shape[0]):
            if ground_truth_data[t, 1] == 'Call' or ground_truth_data[t, 1] == 'Calls':
                # print("pass")
                if not (ground_truth_data[t, 2] in type_call):
                     call = ground_truth_data[t, 2]
                    #  call = call.replace(' ', '') # "Unk nown " -> "Unknown"
                    #  call = call.upper() # phee-trill -> PHEE-TRILL
                    #  call = call.title() # PHEE-TRILL -> Phee-Trill
                     ground_truth_data[t, 2] = correct_label[call]
        np.savetxt(data, ground_truth_data, fmt="%s", delimiter="\t")
        # print("save")
    print("savetxt")
            


if __name__ == "__main__":

    path1 = "/home/muesaka/projects/marmoset/raw/marmoset_23ue_text/nog_remake"
    path2 = "/home/muesaka/projects/marmoset/raw/marmoset_11vpa_text/nog_remake"

    path_text1 = pathlib.Path(path1)
    list_text1 = list(path_text1.glob("*.txt"))
    path_text2 = pathlib.Path(path2)
    list_text2 = list(path_text2.glob("*.txt"))

    find_calltype(list_text1 + list_text2)

    # remake_calltype(list_text1)
    # remake_calltype(list_text2)

    # find_calltype(list_text1 + list_text2)



