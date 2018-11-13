import pandas as pd
import numpy as np
import cv2
import torch
import pandas as pd

class Data_Loader:
    input_loc = ""
    classes = []
    num_classes = 0
    train_dataset = []
    def __init__(self, input_loc):
        self.input_loc = input_loc+"/"
    def read_csv(self, phase): 			#phase: training or validation
        dir_loc = self.input_loc+phase+"/directory.csv"
        directory = pd.read_csv(dir_loc)
        return directory
        #print directory
    def create_4d(self, phase):
    	train_dir = self.read_csv(phase)
        self.classes = np.unique(train_dir["label"].values)
        self.num_classes = len(self.classes)
        train_dataset = []
        train_labelset = []
        
        #print self.read_and_join(train_dir["rgb"][0], train_dir["label"][0], "train")
        print "Following images are imported into {} dataset".format(phase)
        for i in range(len(train_dir)):
        	d_im = self.read_and_join(train_dir["rgb"][i], train_dir["label"][i], phase)
        	if d_im is not None:
        	    train_dataset.append(d_im)
        	    train_labelset.append(train_dir["label"][i])
        train_dataset = np.array(train_dataset)
        print phase+" set shape: ", train_dataset.shape
        print "Out of {} images {} were skipped".format(len(train_dir), (len(train_dir)-train_dataset.shape[0]))
        print "\n"
        #return np.array(train_dataset), np.array(train_labelset)
        #pd.factorize(tl)[0]
        return len(train_dir), torch.from_numpy(np.array(train_dataset)).float(), torch.from_numpy(pd.factorize(train_labelset)[0])


    def read_and_join(self, filename, cl, phase):
        rgb_loc = self.input_loc + phase + "/" + cl + "/" + filename
        depth_loc = self.input_loc + phase + "/" + cl + "/depth/" + filename
        print "\t{}".format(rgb_loc)
        img = cv2.imread(rgb_loc)
        img2 = cv2.imread(depth_loc, 0)
        try:
            img = cv2.resize(img, (28, 28))
            img2 = cv2.resize(img2, (28, 28))
        except:
        	print "Image skipped"
        	return
        
        return np.rollaxis(np.dstack((img,img2)), 2, 0)
        

if __name__ == "__main__":

    dl = Data_Loader("data")
    
    td, tl  = dl.create_4d("train")
    vd, vl  = dl.create_4d("val")


