import numpy as np
from chainer import cuda
from chainer import function


class Crop(function.Function):
    """
    must refactore. 
    """

    def __init__(self, center, size):
        self.size = size
        self.center = center

    def forward(self, inputs):
        x, = inputs
        xp = cuda.get_array_module(x)
        n_images, ch, h, w = x.shape

        in_size = xp.array([h, w])

        # [-1, 1]^2 -> [0, h_i]x[0, w_i]
        center = 0.5 * (self.center+1) * (in_size+1)
        """
        print("centercentercenter")
        print(center)
        print(self.center)
        """

        top_left = center - 0.5*self.size
        top_left = xp.floor(top_left).astype(np.int32)

        y = xp.zeros((n_images, ch, self.size, self.size), dtype=np.float32)
        for k in range(n_images):
            tl_y, tl_x = top_left[k] # k-th batch

            sl_strt_y = max(0, tl_y)
            sl_strt_x = max(0, tl_x)
            sl_end_y = min(sl_strt_y + self.size, h)
            sl_end_x = min(sl_strt_x + self.size, w)

            patch_h_size = max(0, sl_end_y - sl_strt_y)
            patch_w_size = max(0, sl_end_x - sl_strt_x)

            if patch_h_size==0 or patch_w_size==0:
                continue

            """
            print("-------------")
            print(x.shape)
            print(sl_strt_y, sl_strt_x)
            print(sl_end_y, sl_end_x)
            print(tl_y, tl_x)

            print(patch_h_size, patch_w_size)

            print(x[k, :, sl_strt_y:sl_end_y, sl_strt_x:sl_end_x].shape)
            print(y[k, :, :patch_h_size, :patch_w_size].shape)
            """

            y[k, :, :patch_h_size, :patch_w_size] \
                = x[k, :, sl_strt_y:sl_end_y, sl_strt_x:sl_end_x]
        return y,

    # do not backward (always return 0)
    def backward(self, inputs, grad_outputs):
        x, = inputs
        gy, = grad_outputs
        xp = cuda.get_array_module(x)

        n, c = gy[0].shape[:2]
        h, w = x.shape[2:4]
        gx = xp.zeros_like(x, dtype=np.float32)
        return gx,

def crop(x, center, size):
    return Crop(center, size)(x)
