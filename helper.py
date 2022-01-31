import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def import_files(n_files):
    file_matrixes = []
    for i in range(n_files):
        arr = np.fromfile("matrix_step-{num_file:d}.txt".format(num_file=i), dtype=np.int32)
        n = arr[0]
        arr = np.delete(arr,0)
        matrix = np.reshape(arr,(n,n))
        file_matrixes.append(matrix)
    return file_matrixes,n

def plot_anim(file_matrixes,n_files):
    def animate_func(i):
        im.set_array(file_matrixes[i])
        return [im]

    def init():
        im.set_array(file_matrixes[0])
        return [im]
    fig = plt.figure(figsize=(8,8))
    a = file_matrixes[0]

    im=plt.imshow(a,interpolation = 'none', aspect = 'auto', vmin = 0, vmax = 1)

    anim = animation.FuncAnimation(fig,animate_func,frames = n_files,interval = 1000)
    anim.save('test.mp4')

def check_validity(file_matrixes,size,n_files):
    for i in range(1,n_files):
        prev = file_matrixes[i-1]
        curr = file_matrixes[i]
        for ia in range(size):
            for ja in range(size):
                sum1 = 0
                sum1 += prev[ia][ja]
                sum1 += prev[(ia+1)%size][ja]
                sum1 += prev[ia-1][ja]
                sum1 += prev[ia][ja-1]
                sum1 += prev[ia][(ja+1)%size]
                sign = 1 if sum1 > 0 else -1
                if sign != curr[ia][ja]:
                    return 1
    return 0

n = 15
matrix,size = import_files(n)
plot_anim(matrix, n)
if(check_validity(matrix,size, n)==0):
    print("Model is working")
else:
    print("Model is not working")