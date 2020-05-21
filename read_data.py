import glob, copy, pywt
import scipy
import wfdb
from sklearn import svm
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import Counter
from ecgdetectors import Detectors
all_paths = glob.glob("D:/studying_2020/arythmia/mitdb/*.dat")
#first 23 sample starts with 100 are samples of more common arrythmias other
# 200 are more rare arrythmias

def calculate_entropy(list_values):
    counter_values = Counter(list_values).most_common()
    probabilities = [elem[1]/len(list_values) for elem in counter_values]
    entropy=scipy.stats.entropy(probabilities)
    return entropy

class ECGSample:
    def __init__(self, path):
        self.path = path
        self.file = path.split(".")[0][-3:]

        self.good = {'N': 1, 'L': 1, 'R': 1, 'B': 1, 'A': 2,
                     'a': 2, 'J': 2, 'S': 2, 'V': 3, 'r': 3,
                     'F': 4, 'e': 2, 'j': 2, 'n': 2, 'E': 3,
                     '/': 5, 'f': 5, 'Q': 5, '?': 5}
        self.ann_arr = ["(AB", "(AFIB", "(AFL", "(B", "(BII", "(IVR",
                        "(N", "(NOD", "(P", "(PREX", "(SBR", "(SVTA",
                        "(T", "(VFL", "(VT"]
        self.quality = ["TS", "PSE", "MISSB", "U", "qq", 0]
        self.detectors = Detectors(360)

        self.record1 = None
        self.record2 = None
        self.record_detail = None
        self.n_ann = None # number of annotations
        self.inds = None # indexes of annotation according to 650 000 array values
        self.arrhyt = None # arrythmias names and 0 if no, according to self.inds
        self.beat = None # beat names, according to self.inds

        # read data for the sample
        self.read_record()
        self.read_annot()

        self.good_int = [i for i in range(len(self.ann_arr))] #arrhythmias indexes coresponding value sin self.ann_arr

        self.beats_full = None # 0 for no annotation from 1- 5 name of beat
        self.arrhyt_full = None # -1 for no annotation 0-14 name of arrhythmia

        self.clean_ann_beat()
        self.convert_arr_ann()

        self.r_peaks = None
        self.get_r_peaks()

    def read_record(self):
        record = wfdb.rdsamp(self.path.split(".")[0])
        self.record1 = copy.deepcopy(record[0][:, 0])
        self.record2 = copy.deepcopy(record[0][:, 1])
        self.record_detail = record[1]

    def read_annot(self):

        annotation = wfdb.rdann(self.path.split(".")[0], 'atr')
        arryth = [None for i in range(650000)]
        beat= [None for i in range(650000)]
        self.n_ann =len(annotation.sample)
        self.inds = annotation.sample
        self.arrhyt = annotation.aux_note
        self.beat = annotation.symbol

        c = 0
        n = []

        for i in self.arrhyt:
            if i != "":
                n.append(i)
            else:
                n.append(0)
        self.arrhyt = n

    def plot_record1(self, start, end, R_p, arryth_show = False, move = 0):
        if arryth_show == False:
            plt.plot([i / 360 for i in range(end-start)], self.record1[start:end])

            s = start+move
            colors = ["pink", "red", "blue", "green", "black", "yellow"]
            for i in self.beats_full[start:end]:

                if i != 0:
                    plt.plot([(s-start) / 360], [self.record1[s]], marker='o', markersize=6, color=colors[i])

                s+=1
            red_patch = mpatches.Patch(color='red', label='N,R,L,B (Normal)')
            blue_patch = mpatches.Patch(color='blue', label='AaJSej (SVEB)')
            green_patch = mpatches.Patch(color='green', label='V,E (VEB)')
            black_patch = mpatches.Patch(color='black', label='F (Fusion beat)')
            yellow_patch = mpatches.Patch(color='yellow', label='/,f,Q,? (Unknown beat)')
            plt.legend(handles=[red_patch, blue_patch, green_patch,  yellow_patch, black_patch], loc = 3, fontsize ='xx-small', ncol = 3)
            if R_p == True:
                for l in self.r_peaks:
                    if start <= l <= end:
                        plt.axvline(x=(l-start) / 360, color = "red")
            plt.show()




        if arryth_show == True:
            plt.plot([i / 360 for i in range(end - start)], self.record1[start:end])

            s = start
            colors = ["pink", "red", "blue", "green", "black", "yellow"]
            for i in self.arrhyt_full[start:end]:
                if i != -1:
                    plt.plot([(s - start) / 360], [self.record1[s]], marker='o', markersize=6, color="red")
                    plt.annotate(str(self.ann_arr[i]), ((s - start) / 360, self.record1[s]))
                s += 1


            plt.show()

    def clean_ann_beat(self):
        """

        """
        n = []
        for i in range(len(self.beat)):

            if self.beat[i] in self.good.keys():
                n.append([int(self.inds[i]), self.good[self.beat[i]]])

        ne = [0 for i in range(len(self.record1))]
        for i in n:
            ne[i[0]] = i[1]
        self.beats_full = ne
        return n

    def convert_arr_ann(self):
        n = []
        for i in range(len(self.arrhyt)):

            if self.arrhyt[i] != 0:
                name_a = self.arrhyt[i].rstrip('\x00')
                if name_a not in self.quality:
                    n.append([int(self.inds[i]), self.ann_arr.index(name_a)])

        ne = [-1 for i in range(len(self.record1))]
        for i in n:
            ne[i[0]] = i[1]
        self.arrhyt_full = ne
        return np.array(n)

    def get_r_peaks(self):
        self.r_peaks = self.detectors.two_average_detector(self.record1)

    def gen_features(self):
        data_l = []
        labels = []
        inds = []
        for ind, el in enumerate(self.beats_full):

            if el != 0:
                if ind>=140 and ind +140 < len(self.record1):

                    data_l.append(list(self.record1[ind-140:ind+140]))
                    labels.append(el)
                    inds.append(ind)

        """
        # GLOBAL RR
        inds = np.array(inds)
        global_rr = np.sum(np.ediff1d(inds)) / (len(inds)-1)
        """
        # LOCAL RR
        rr_s = []
        for i, e in enumerate(inds):
            features = [] # previous_rr #next_rr #av_10_rr #entropy

            # if i != 0:
            #     previous_rr = e-inds[i-1]
            # else:
            #     previous_rr = inds[i + 1] - e
            # features.append(previous_rr)
            #
            # if i != len(inds)-1:
            #     next_rr = inds[i + 1] - e
            # else:
            #     next_rr = e - inds[i-1]
            # features.append(next_rr)
            #
            # if i - 5 < 0:
            #     tail1, tail2 = 0, 1
            # else:
            #     tail1, tail2 = i - 5, i - 4
            # if i + 6 >= len(inds):
            #     front = len(inds)
            #     front2 = front - 1
            # else:
            #     front = i+6
            #     front2 = i+5
            #
            # ten = inds[tail2:front]
            # ten_shift = inds[tail1:front2]
            # ten_dif = np.array(ten) - np.array(ten_shift)
            # av_ten_dif = np.sum(ten_dif) / len(ten_dif)
            # features.append(av_ten_dif)
            #
            # features.append(calculate_entropy(data_l[i]))


            coeffs = pywt.wavedec(data_l[i] ,'db1', level=2)

            #features = features + list(coeffs[0])  # for SVM
            features = list(coeffs[0]) # for 1-d CNN
            
            rr_s.append(features)



        return rr_s, labels


train= [ 101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
test = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]


class ECGDataset:
    def __init__(self, directory = "D:/studying_2020/arythmia/mitdb/*.dat"):
        self.directory = directory

        self.raw_samples = None
        self.read_samples()

    def read_samples(self):
        all_paths = glob.glob(self.directory)
        s = []
        for p in all_paths:
            one = ECGSample(p)
            s.append((one))
        self.raw_samples = s

    def get_statistics_arryth(self, start=0, end= 48):
        arr_d ={}
        for i in self.raw_samples[0].good_int:
            arr_d[i] =0

        for s in self.raw_samples[start:end]:
            for j in s.convert_arr_ann():
                arr_d[j[1]] += 1
        return arr_d

    def get_statistics_beat(self, start=0, end= 48):
        beat_d ={}
        for i in [1, 2, 3,4, 5]:
            beat_d[i] =0

        for s in self.raw_samples[start:end]:
            for j in s.clean_ann_beat():
                beat_d[j[1]] += 1
        return beat_d

    def get_train_test_sets(self, train, test, val =None):
        train_x, train_y = [], []
        test_x, test_y = [], []
        val_x, val_y = [], []
        for s in self.raw_samples:
            f, l = s.gen_features()
            if int(s.file) in train:
                train_x += f
                train_y += l
            if int(s.file) in test:
                test_x += f
                test_y += l
            if val:
                if int(s.file) in val:
                    val_x += f
                    val_y += l

        if val:
            return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y), np.array(val_x), np.array(val_y)
        return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)

# s = ECGSample(all_paths[8])
# f, l = s.gen_features()
# dataset = ECGDataset()
# print("loaded")
# train_x, train_y, test_x, test_y = dataset.get_train_test_sets(train, test)
# print(train_x.shape)
# print(train_y.shape)
# print(test_x.shape)
# print(test_y.shape)