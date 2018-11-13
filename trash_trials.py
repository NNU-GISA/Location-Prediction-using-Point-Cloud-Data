import numpy as np

a = np.array([
	           [[1,2,3,0], [4,5,6,0], [7,8,9,0]],
	            [[1,2,3,0], [4,5,6,0], [7,8,9,0]],
	            [[1,2,3,0], [4,5,6,0], [7,8,9,0]] 
	            
	           ])

print np.rollaxis(a, 2, 0)
'''
a = np.array([
	           [[[1,2,3,0], [4,5,6,0], [7,8,9,0]],
	            [[1,2,3,0], [4,5,6,0], [7,8,9,0]],
	            [[1,2,3,0], [4,5,6,0], [7,8,9,0]] ],
	            [[[1,2,3,0], [4,5,6,0], [7,8,9,0]],
	            [[1,2,3,0], [4,5,6,0], [7,8,9,0]],
	            [[1,2,3,0], [4,5,6,0], [7,8,9,0]] ]
	           ])
x = np.array([ [ [0,1,2,3], [2,3,4,5] ],
	           [ [4,5,6,7], [6,7,8,9] ] ])
y = np.swapaxes(x, 0, 2)
print x, "\n\n", np.swapaxes(x,0,2)
print "a:{}  x:{}  y:{}".format(a.shape, x.shape, y.shape)

#print type(a), type(a[0][0][0][0]), a.shape
#print type(td), type(td[0][0][0][0]), td.shape
#b = torch.from_numpy(a)
#b = torch.from_numpy(td)
#print type(b)

import torch
import torch.utils.data as utils
import numpy as np

my_x = [np.array([[1.0,2],[3,4]]),np.array([[5.,6],[7,8]])] # a list of numpy arrays
my_y = [np.array([4.]), np.array([2.])] # another list of numpy arrays (targets)


tensor_x = torch.stack([torch.Tensor(i) for i in my_x]) # transform to torch tensors
tensor_y = torch.stack([torch.Tensor(i) for i in my_y])

my_dataset = utils.TensorDataset(tensor_x,tensor_y) # create your datset
my_dataloader = utils.DataLoader(my_dataset) # create your dataloader
'''