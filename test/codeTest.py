from matplotlib import axis
import numpy as np
                         #1st example with 2 channels                #2nd example with 2 channels
                         #1st channel      2nd channel                1st channel      2nd channel
a_prev = np.array([[[[1,2,3],[4,5,6]], [[1,2,2],[1,1,1]]],      [[[5,5,5],[6,6,6]], [[1,2,3],[4,5,6]]]])
a_prev[0,:,:,:]
#print(*a_prev.shape)
np.max(np.array([2,5,8]), keepdims=True)
np.max(a_prev, axis = 0, keepdims=True).shape
print(a_prev.shape)
np.sum(a_prev, axis=(2,3))


a = np.array([[ 1.,  1.,  1.,  1.,  1.], [ 1.,  1.,  1.,  1.,  1.], [ 1.,  1.,  1.,  1.,  1.]])
a
np.pad(a, [(0, 0), (0,0)], mode='constant', constant_values=0)
np.pad(a, [(1, 0), (0,0)], mode='constant', constant_values=0)
np.pad(a, [(0, 1), (0,0)], mode='constant', constant_values=0)
np.pad(a, [(1, 1), (0,0)], mode='constant', constant_values=0)
np.pad(a, [(1, 1), (1,0)], mode='constant', constant_values=0)
np.pad(a, [(1, 1), (0,1)], mode='constant', constant_values=0)
np.pad(a, [(1, 1), (1,1)], mode='constant', constant_values=0)

a = np.array([[1,2,4,5,6,7,8,9]])
i = 2
a.shape
a[0,i:i+1]

a = 2
a *= 5
a
int(4.75)
import numpy as np
np.arange(5)
a_prev = np.array([ [[1,2,3],[4,5,6]],      [[5,5,5],[6,6,6]] ])
a_prev.shape
np.sum(a_prev, axis = 0, keepdims=True).shape

np.clip(a_prev, 0, 3.0)