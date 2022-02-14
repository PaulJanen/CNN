import numpy as np
                         #1st example with 2 channels                #2nd example with 2 channels
                         #1st channel      2nd channel                1st channel      2nd channel
a_prev = np.array([[[[1,2,3],[4,5,6]], [[1,2,2],[1,1,1]]],      [[[5,5,5],[6,6,6]], [[1,2,3],[4,5,6]]]])
a_prev[0,:,:,:]
#print(*a_prev.shape)
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
