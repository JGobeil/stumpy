
import numpy as np
import topomod


@topomod
def polynominal_background_correction(self, order):
    data = self.data

    x_means = np.mean(data, axis=0)
    x_i = np.arange(len(x_means))
    x_fit = np.polyfit(x_i, x_means, order)
    data = data - np.poly1d(x_fit)(x_i)

    y_means = np.mean(data, axis=1)
    y_i = np.arange(len(y_means))
    y_fit = np.polyfit(y_i, y_means, order)
    data = (data.T - np.poly1d(y_fit)(y_i)).T

    return self.clone(data=data)

