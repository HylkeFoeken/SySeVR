'''
This python file is used to convert an 3d array to a 3d numpy array
'''
import numpy as np

def make_dim_same_length(uneven_dim_list, max_length, vector_dim):
    list_len = len(uneven_dim_list)
    even_dim_list = np.zeros((list_len, max_length, vector_dim), dtype=np.float32)

    for a in range(len(uneven_dim_list)):
        for b in range(len(uneven_dim_list[a])):
            for c in range(len(uneven_dim_list[a][b])):
                if a < list_len and b < max_length and c < vector_dim:
                    even_dim_list[a, b, c] = uneven_dim_list[a][b][c]

    return even_dim_list


if __name__ == "__main__":
    x = 2
    y = 3
    z = 2

    m = []
    l = []

    for i in range(x):
        m.append([])

        for j in range(y):
            if j % 2 == 0:
                m[i].append(np.ones(z, dtype=np.float32))
            else:
                m[i].append(np.ones(z // 2, dtype=np.float32))

    m = make_dim_same_length(m)
    l += m

    print('l = length: {}, type: {}:'.format(len(l), type(l)))
    print('l[0] = length: {}, type: {}:'.format(len(l[0]), type(l[0])))
    print('l[0][0] = length: {}, type: {}:'.format(len(l[0][0]), type(l[0][0])))

    np_l = np.array(l)
    
    print('np_l = length: {}, type: {}, shape: {}'.format(len(np_l), type(np_l), np_l.shape))
    print('np_l[0] = length: {}, type: {}'.format(len(np_l[0]), type(np_l[0])))
    print('np_l[0,0] = length: {}, type: {}'.format(len(np_l[0, 0]), type(np_l[0, 0])))

