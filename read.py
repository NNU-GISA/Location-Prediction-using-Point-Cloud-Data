import pandas as pd
import numpy as np
import cv2
import torch
import pandas as pd
import sys
from sklearn.utils import shuffle
from skimage.transform import resize
import h5py as h5

class Data_Loader:
    input_loc = ""
    im_size = 0
    classes = []
    num_classes = 0
    dataset = {}
    labelset = {}
    def __init__(self, input_loc, im_size):
        self.input_loc = input_loc+"/"
        self.im_size = im_size
    def read_csv(self, phase): 			#phase: training or validation
        dir_loc = self.input_loc+phase+"/directory.csv"
        directory = pd.read_csv(dir_loc)
        directory = directory.sample(frac=1).reset_index(drop=True)
        #print directory
        return directory
    def create_4d(self, phase, fraction=1):
        train_dir = shuffle(self.read_csv(phase))
        self.classes = np.unique(train_dir["label"].values)
        self.num_classes = len(self.classes)
        train_dataset = []
        train_labelset = []
        
        #print self.read_and_join(train_dir["rgb"][0], train_dir["label"][0], "train")
        #print "Following images are imported into {} dataset".format(phase)
        for i in range(len(train_dir)/fraction):
        	d_im = self.read_and_join(train_dir["rgb"][i], train_dir["label"][i], phase)
        	if d_im is not None:
        	    train_dataset.append(d_im)
        	    train_labelset.append(train_dir["label"][i])
        
        train_dataset = self.transform(np.array(train_dataset))
        
        print phase+" set shape: ", train_dataset.shape
        print "Out of {} images {} were skipped".format(len(train_dir), (len(train_dir)-train_dataset.shape[0]))
        print "\n"
        
        #return np.array(train_dataset), np.array(train_labelset)
        #pd.factorize(tl)[0]
        return len(train_dir), torch.from_numpy(np.array(train_dataset)).float(), torch.from_numpy(pd.factorize(train_labelset)[0])
        #return len(train_dir), train_dataset, pd.factorize(train_labelset)[0]

    def transform(self, dataset):
        dataset = np.rollaxis(dataset, 1,0)

        transformed_dataset = []
        for i in range(len(dataset)):
        	transformed_dataset.append(self.normalize(self.zero_center(dataset[i])))
        return np.rollaxis(np.array(transformed_dataset), 1, 0)
    def zero_center(self, dsi):
    	return dsi - np.mean(dsi, axis=0)
    def normalize(self, dsi):
    	return dsi / np.std(dsi, axis=0)
    
    def read_and_join(self, filename, cl, phase):
        rgb_loc = self.input_loc + phase + "/" + cl + "/" + filename
        depth_loc = self.input_loc + phase + "/" + cl + "/depth/" + filename
        #print "\t{}".format(rgb_loc)
        img = cv2.imread(rgb_loc)
        img2 = cv2.imread(depth_loc, 0)
        #img2 = cv2.imread(depth_loc)
        #print img2.shape
        print img.shape
        try:
            img = cv2.resize(img, (self.im_size, self.im_size))
            img2 = cv2.resize(img2, (self.im_size, self.im_size))
            #img = resize(img, (27, 27))
            #img2 = resize(img2, (27, 27))
            #img = img[0:480, 0:480]
            #img2 = img2[0:480, 0:480]
        except:
            print "\tImage cropped:", sys.exc_info()[0]
            
            return
        
        
        return np.rollaxis(np.dstack((img,img2)), 2, 0)
        #return np.rollaxis(img2, 2, 0)
        

if __name__ == "__main__":

    dl = Data_Loader("data", 27)
    fileset = ["size", "dataset", "labelset"]
    #dl.read_csv("train")
    #ts, td, tl = dl.create_4d("train")
    #vs, vd, vl = dl.create_4d("val")
    for phase in ["train", "val"]:
        i = 0
        for file in dl.create_4d(phase):
            filename  = dl.input_loc+"/"+phase+"_"+str(fileset[i])+".h5"
            #print filename
            #h5f = h5.File(filename, 'w')
            #h5f.create_dataset('dataset_1', data=file)
            #h5f.close()
            i += 1



