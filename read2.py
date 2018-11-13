from skimage import io
from skimage import color
from skimage.transform import resize
import os
import threading
import numpy as np
import thread
import time
import cv2
import pandas as pd

        	
         
class Toolset:
    def unison_shuffled_copies(self, a, b):
        #assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]
class Data_Loader:
    input_loc = ""
    im_size = 0
    class_set = []
    im_set = {"im": [], "l": []}
    def __init__(self, input_loc="~", im_size=0):

        self.input_loc = os.path.abspath(self.input_loc)+"/"+input_loc+"/"
        self.im_size = im_size
    def create_classset(self):
        for x in os.listdir(self.input_loc+"rgb/"):
            if os.path.isdir(self.input_loc+"rgb/"+x):
                self.class_set.append(x)
        return self.class_set
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
        
        try:
            #img = cv2.resize(img, (self.im_size, self.im_size))
            #img2 = cv2.resize(img2, (self.im_size, self.im_size))
            img = resize(img, (27, 27))
            img2 = resize(img2, (27, 27))
            #img = img[0:480, 0:480]
            #img2 = img2[0:480, 0:480]
        except:
            print "Image skipped"
            return
        
        
        return np.rollaxis(np.dstack((img,img2)), 2, 0)

dl = Data_Loader(input_loc = "data4", im_size = 386)        
class reader(threading.Thread):
    im_loc = ""
    im_name = ""
    def __init__(self, im_loc = "~"):
        super(reader, self).__init__()
        #print im_loc
        #self.dloader = dl_obj
        self.im_loc = im_loc
        self.im_name = im_loc.split('/')[len(im_loc.split('/'))-2]
    def run(self):
        
        collection = io.ImageCollection(self.im_loc + '*.jpg')
        #print np.array(collection).shape
        for im in collection: 
            thread.start_new_thread( self.resize_im, (im, ) )    
        #np.array(collection).shape
    def resize_im(self, im):
        i = len(dl.im_set["im"])
        im = color.rgb2grey(resize(im, (27, 27), mode="constant"))
        dl.im_set["im"].insert(i, im)
        dl.im_set["l"].insert(i, self.im_name)
        #a.insert(len(a), x)
        #print len(self.collection)

class dataset_creator(threading.Thread):
    name = ""
    im_threads = []
    def __init__(self, name = "rgb"):
	    super(dataset_creator, self).__init__()
	    self.name = name + "/"
    def run(self):
        for x in dl.create_classset():
            im_loc = dl.input_loc + self.name + x +"/"
            var = reader(im_loc = im_loc)
            self.im_threads.append(var)
            var.start()

if __name__ == "__main__":
    #rgb = dataset_creator(name = "rgb")
    d = dataset_creator(name = "d")
    #rgb.start()
    d.start()
    while len(threading.enumerate()) != 1:
        a = ""
    tool = Toolset();
    print len(dl.im_set["im"]), len(dl.im_set["l"])
    #dl.im_set["im"], dl.im_set["l"] = tool.unison_shuffled_copies(dl.im_set["im"], pd.factorize(dl.im_set["l"]))
    #for i in dl.im_set["l"]:
    #    print i
    
    




    # rnge = np.random.randint(len(dl.im_set["l"]), size=10)
    # for i in rnge:
    #     cv2.imshow(dl.im_set["l"][i]+str(i),dl.im_set["im"][i])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    

    
    #d = dataset_creator(dl_obj = dl, name = "d")
    #d.start()


    
    
    #d = dataset_creator(dl, name = "d")
    