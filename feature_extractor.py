import numpy as np
#import pywt
from sklearn.gaussian_process import GaussianProcess


def squared_exponential_periodic_1D(theta, d):
    theta = np.asarray(theta, dtype=np.float)
    d = np.asarray(d, dtype=np.float)
    return np.exp(-theta[0] * np.sum(np.sin(np.abs(d)) ** 2, axis=1))


def fold_time_series(time_point, period, div_period):
    real_period = period / div_period
    return time_point % real_period  # modulo real_period


def unfold_sample(x, color):
    """Operates inplace"""
    real_period = x['period'] / x['div_period']
    phase = (x['time_points_%s' % color] %
             real_period) / real_period * 2 * np.pi
    order = np.argsort(phase)
    x['phase_%s' % color] = phase[order]
    x['light_points_%s' % color] = np.array(
        x['light_points_%s' % color])[order]
    x['error_points_%s' % color] = np.array(
        x['error_points_%s' % color])[order]
    x['time_points_%s' % color] = np.array(x['time_points_%s' % color])[order]


def binify(bins, a, b, c):
    a_dig = np.digitize(a, bins) - 1
    not_empty_bins = np.unique(a_dig)
    a_bin = np.array([np.mean(a[a_dig == i]) for i in not_empty_bins])
    b_bin = np.array([np.mean(b[a_dig == i]) for i in not_empty_bins])
    c_bin = np.array([np.mean(c[a_dig == i]) for i in not_empty_bins])
    return a_bin, b_bin, c_bin


class FeatureExtractor(object):

    def __init__(self):
        pass

    def fit(self, X_dict, y):
        pass

    def transform(self, X_dict):
        n_points_per_period = 200
        bins_per_period = 10
        sampling_rate = n_points_per_period / bins_per_period
        t_test = np.linspace(-2 * np.pi, 4 * np.pi, 3 * n_points_per_period)
        num_gp_bins = 12
        gp_bins = [i * 2 * np.pi / num_gp_bins for i in range(num_gp_bins + 1)]

        X = []
        for ii, x in enumerate(X_dict):
            if ii % 100 == 0:
                print ii
            real_period = x['period'] / x['div_period']
            x_new = [x['magnitude_b'], x['magnitude_r'], real_period, x['asym_b'], x['asym_r']]
            for color in ['r', 'b']:
                unfold_sample(x, color=color)
                x_train = x['phase_' + color]
                y_train = x['light_points_' + color]
                y_sigma = x['error_points_' + color]

                x_train, y_train, y_var = binify(gp_bins, x_train, y_train, y_sigma ** 2)

                gp = GaussianProcess(regr='constant', theta0=1. / 1.0,
                                     thetaL=1. / 80., thetaU=1. / 0.05,
                                     corr=squared_exponential_periodic_1D,
                                     nugget=y_var)
                gp.fit(x_train[:, np.newaxis], y_train)
                
                y_test = gp.predict(t_test[:, np.newaxis])
                length_scale = np.sqrt(2 / gp.theta_[0][0])
                x_new.append(length_scale)

                min_y = min(y_test)
                amplitude = max(y_test) - min_y
                x_new.append(amplitude)

                # first max after t = 0
                imax = n_points_per_period
                imax += np.argmax(y_test[n_points_per_period:2 * n_points_per_period])

                # sample points from second period [0, 2pi]
                gp_samples = y_test[imax: imax + n_points_per_period: sampling_rate]
                
                #kyrtosis in the first period
                kyrtos=np.kurtosis(y_test[imax: imax + (0.5*n_points_per_period): sampling_rate], fisher=True)
                #scipy.stats.kurtosis(y_test[imax: imax + (0.5*n_points_per_period): sampling_rate], axis=0, fisher=True, bias=True)
                x_new.append(kyrtos)
                               
                #finding the difference between red and blue
                if color == 'r':
                    max_r=max(y_test)
                if color == 'b':
                    dif_max=max_r-(max(y_test))
                
                # normalize sampled points between 0 and 1
                gp_samples_normalized = 1 - (gp_samples - min_y) / amplitude
                
                for gp_sample in gp_samples_normalized:
                    x_new.append(gp_sample)
                '''
                (cA, cD) = pywt.dwt(gp_sample,'db',mode='ppd',level=3)
                wavelet_cA=cA
                wavelet_cD=cD
                x_new.append(cA)
                x_new.append(cD)
                '''  
            x_new.append(dif_max)
            X.append(x_new)

        return np.array(X)
