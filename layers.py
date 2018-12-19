import numpy as np


def fully_connected_forward(x, w, b):
    """
        Forward pass for the fully connected layer.
        Arguments:
            x: numpy array of image input with shape (N, d1, d2,..., dc)
            w: numpy array of weight matrix with shape (D, B)
            b: numpy array of bias with shape (B,)
        Output:
            out: numpy array of output with shape (N, B)
            cache: tuple (x, w, b) for backprop use
    """
    cache = (x, w, b)
    x = x.reshape(x.shape[0], -1)
    out = x.dot(w) + b
    return out, cache


def fully_connected_backward(dout, cache):
    """
        Backward pass for the fully connected layer.
        Arguments:
            dout: numpy array of gradient of output passed from next layer with shape (N, B)
            cache: tuple (x, w, b)
        Output:
            dx: numpy array of gradient for image input with shape (N, d1, d2,..., dc)
            dw: numpy array of gradient for weight matrix with shape (D, B)
            db: numpy array of gradient for bias with shape (B,)
    """
    x, w, b = cache
    dx = dout.dot(w.T)
    dx = dx.reshape(x.shape)
    t = x.reshape(x.shape[0], -1)
    dw = t.T.dot(dout)
    db = np.sum(dout.T, axis=1)
    return dx, dw, db


def relu_forward(x):
    """
        Forward pass for the ReLU function layer.
        Arguments:
            x: numpy array of input with any shape
        Output:
            out: numpy array of output with same shape of x
            cache: tuple (x, w, b) for backprop use
    """
    cache = x
    out = x.copy  # very important, or x will change with out
    out[out < 0] = 0
    return out, cache


def relu_backward(dout, cache):
    """
        Backward pass for the ReLU function layer.
        Arguments:
            dout: numpy array of gradient of output passed from next layer with any shape
            cache: tuple (x)
        Output:
            x: numpy array of gradient for input with same shape of dout
    """
    x = cache
    dx = dout * (x >= 0)
    return dx


def convolution_forward_naive(x, w, b, params):
    """
        A naive implementation of the forward pass of convolution layer.
        Arguments:
            x: numpy array of input image with shape (N, C, H, W)
            w: numpy array of filters with shape (F, C, HH, WW)
            b: numpy array of bias with shape (F,)
            params: dictionary of convolution layer parameters
                - 'stride': integer of stride
                - 'pad': integer of pad
        Output:
            out: numpy array of output with shape (N, F, Hout, Wout)
                - Hout = 1 + (H + 2 * pad - HH) / stride
                - Wout = 1 + (W + 2 * pad - WW) / stride
            cache: tuple (x, w, b, params) for backprop use
    """
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    Hout = 1 + (H + 2 * params['pad'] - HH) // params['stride']
    Wout = 1 + (W + 2 * params['pad'] - WW) // params['stride']
    xpad = np.zeros((N, C, params['pad'] * 2 + H, params['pad'] * 2 + W))
    out = np.zeros((N, F, Hout, Wout))
    for xn in range(N):
        for fn in range(F):
            for cn in range(C):
                xpad[xn, cn, params['pad']:-params['pad'], params['pad']:-params['pad']] = x[xn, cn]
                for i in range(Hout):
                    for j in range(Wout):
                        hh = i * params['stride']
                        ww = j * params['stride']
                        out[xn, fn, i, j] += np.sum(np.multiply(xpad[xn, cn, hh:hh + HH, ww:ww + WW], w[fn, cn]))
            out[xn, fn] += b[fn]
    cache = (xpad, w, b, params)  # notice that the padded input is stored in cache rather than the original input
    return out, cache


def convolution_backward_naive(dout, cache):
    """
        A naive implementation of the backward pass of convolution layer.
        Arguments:
            dout: numpy array of derivative of output with shape (N, F, Hout, Wout)
                - Hout = 1 + (H + 2 * pad - HH) / stride
                - Wout = 1 + (W + 2 * pad - WW) / stride
            cache: tuple (x, w, b, params)
        Output:
            dx: numpy array of gradient of input image with shape (N, C, H, W)
            dw: numpy array of gradient of filters with shape (F, C, HH, WW)
            db: numpy array of gradient of bias with shape (F,)
    """
    x, w, b, params = cache
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    N, F, Hout, Wout = dout.shape
    dx = np.zeros(x.shape)
    dw = np.zeros(w.shape)
    db = np.sum(dout, axis=(0, 2, 3))
    for xn in range(N):
        for fn in range(F):
            for cn in range(C):
                for i in range(Hout):
                    for j in range(Wout):
                        hh = i * params['stride']
                        ww = j * params['stride']
                        dx[xn, cn, hh:hh + HH, ww:ww + WW] += dout[xn, fn, i, j] * w[fn, cn]
                        dw[fn, cn] += x[xn, cn, hh:hh + HH, ww:ww + WW] * dout[xn, fn, i, j]
    dx = dx[:, :, params['pad']:-params['pad'], params['pad']:-params['pad']]
    return dx, dw, db


def max_pooling_forward_naive(x, params):
    """
        A naive implementation of the forward pass of max pooling layer.
        Arguments:
            x: numpy array of input image with shape (N, C, H, W)
            params: dictionary of max pooling layer parameters
                - 'stride': integer of stride
                - 'height': integer of max pooling region height
                - 'width': integer of max pooling region width
        Output:
            out: numpy array of output with shape (N, C, Hp, Wp)
                - Hp = 1 + (H - height) / stride
                - Wp = 1 + (W - width) / stride
            cache: tuple (x, params) for backprop use
    """
    N, C, H, W = x.shape
    Hp = (H - params['height']) // params['stride'] + 1
    Wp = (W - params['width']) // params['stride'] + 1
    out = np.zeros((N, C, Hp, Wp))
    for xn in range(N):
        for cn in range(C):
            for i in range(Hp):
                for j in range(Wp):
                    s = params['stride']
                    hh = params['height']
                    ww = params['width']
                    out[xn, cn, i, j] = np.max(x[xn, cn, i * s:i * s + hh, j * s:j * s + ww])
    cache = (x, params)
    return out, cache


def max_pooling_backward_naive(dout, cache):
    """
        A naive implementation of the backward pass of max pooling layer.
        Arguments:
            dout: numpy array of gradient of output with shape (N, C, Hp, Wp)
            cache: tuple (x, params)
        Output:
            dx: numpy array of gradient of input image with shape (N, C, H, W)
    """
    x, pool_param = cache
    N, C, Hp, Wp = dout.shape
    N, C, H, W = x.shape
    dx = np.zeros(x.shape)
    for xn in range(N):
        for cn in range(C):
            for i in range(Hp):
                for j in range(Wp):
                    s = pool_param['stride']
                    hh = pool_param['height']
                    ww = pool_param['width']
                    x_region = x[xn, cn, i * s:i * s + hh, j * s:j * s + ww]
                    mask = (x_region == np.max(x_region))
                    dx[xn, cn, i * s:i * s + hh, j * s:j * s + ww] = mask * dout[xn, cn, i, j]
    return dx

