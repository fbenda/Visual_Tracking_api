import numpy as np

import utils
import vot
from pyhog import pyhog


class Padding:
    def __init__(self):
        self.generic = 1.8
        self.large = 1
        self.height = 0.4


class DSSTtracker:

    def __init__(self, image, region, scale_model_max_area=32 * 16, learning_rate=0.05,
                 output_sigma_factor=(1 / float(16)), lamda=1e-2):
        self._learning_rate = learning_rate

        # parameters for translation
        _output_sigma_factor = output_sigma_factor
        self._lambda = lamda

        # parameters for scale variance
        scale_sigma_factor = 1 / float(4)
        self._lambda_scale = 1e-2
        n_scales = 33  # number of scale levels
        scale_model_factor = 1.0
        scale_step = 1.02  # step of one scale level
        self._current_scale_factor = 1.0
        _scale_model_max_area = scale_model_max_area

        # initialize translation model
        self.target_size = np.array([region.height, region.width])
        self.pos = [region.y + region.height / 2, region.x + region.width / 2]
        init_target_size = self.target_size
        output_sigma = np.sqrt(np.prod(self.target_size)) * _output_sigma_factor
        self.sz = utils.get_window_size(self.target_size, image.shape[:2], Padding())

        grid_y = np.arange(np.floor(self.sz[0])) - np.floor(self.sz[0] / 2)
        grid_x = np.arange(np.floor(self.sz[1])) - np.floor(self.sz[1] / 2)
        rs, cs = np.meshgrid(grid_x, grid_y)
        g = np.exp(-0.5 / output_sigma ** 2 * (rs ** 2 + cs ** 2))
        self.gf = np.fft.fft2(g, axes=(0, 1))

        feature_map = utils.get_subwindow(image, self.pos, self.sz, feature='hog')
        self.cos_window = np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1]))
        x_hog = np.multiply(feature_map, self.cos_window[:, :, None])
        xf = np.fft.fft2(x_hog, axes=(0, 1))

        self.x_num = np.multiply(self.gf[:, :, None], np.conj(xf))
        self.x_den = np.real(np.sum(np.multiply(xf, np.conj(xf)), axis=2))

        # initialize scale model
        self.base_target_size = self.target_size / self._current_scale_factor
        scale_sigma = np.sqrt(n_scales) * scale_sigma_factor

        # Gaussian shaped label for scale estimation
        ss = np.arange(n_scales) - np.ceil(n_scales / 2)
        self.scaleFactors = np.power(scale_step, -ss)
        ys = np.exp(-0.5 * (ss ** 2) / scale_sigma ** 2)
        self.ysf = np.fft.fft(ys)

        # scale search preprocess
        if n_scales % 2 == 0:
            self.scale_window = np.hanning(n_scales + 1)
            self.scale_window = self.scale_window[1:]
        else:
            self.scale_window = np.hanning(n_scales)

        self.scaleSizeFactors = self.scaleFactors
        self.min_scale_factor = np.power(scale_step,
                                         np.ceil(np.log(5. / np.min(self.sz)) / np.log(scale_step)))

        self.max_scale_factor = np.power(scale_step,
                                         np.floor(np.log(np.min(np.divide(image.shape[:2],
                                                                          self.base_target_size)))
                                                  / np.log(scale_step)))

        if scale_model_factor * scale_model_factor * np.prod(init_target_size) > _scale_model_max_area:
            scale_model_factor = np.sqrt(_scale_model_max_area / np.prod(init_target_size))

        self.scale_model_sz = np.floor(init_target_size * scale_model_factor)

        s = utils.get_scale_subwindow(image, self.pos, self.base_target_size,
                                      self._current_scale_factor * self.scaleSizeFactors, self.scale_window,
                                      self.scale_model_sz)

        sf = np.fft.fftn(s, axes=[0])

        self.s_num = np.multiply(np.conj(self.ysf[:, None]), sf)
        self.s_den = np.real(np.sum(np.multiply(sf, np.conj(sf)), axis=1))

    def track(self, image):

        test_patch = utils.get_subwindow(image, self.pos, self.sz,
                                         scale_factor=self._current_scale_factor)

        # 1. Extract sample
        hog_feature_t = pyhog.features_pedro(test_patch / 255., 1)
        hog_feature_t = np.lib.pad(hog_feature_t, ((1, 1), (1, 1), (0, 0)), 'edge')

        # 2. Compute correlation scores
        xt = np.multiply(hog_feature_t, self.cos_window[:, :, None])
        xtf = np.fft.fft2(xt, axes=(0, 1))
        response = np.real(np.fft.ifft2(np.divide(np.sum(
            np.multiply(self.x_num, xtf), axis=2), (self.x_den + self._lambda))))

        # 3. Calc the new pos
        v_centre, h_centre = np.unravel_index(response.argmax(), response.shape)
        vert_delta, horiz_delta = \
            [(v_centre - response.shape[0] / 2) * self._current_scale_factor,
             (h_centre - response.shape[1] / 2) * self._current_scale_factor]

        self.pos = [self.pos[0] + vert_delta, self.pos[1] + horiz_delta]

        # 4. Extract sample
        st = utils.get_scale_subwindow(image, self.pos, self.base_target_size,
                                       self._current_scale_factor * self.scaleSizeFactors,
                                       self.scale_window, self.scale_model_sz)

        # 5. Compute correlation scores
        stf = np.fft.fftn(st, axes=[0])
        temp_conj = np.conj(self.s_num)
        temp_num = np.sum(np.multiply(temp_conj, stf), axis=1)
        temp_den = (self.s_den + self._lambda_scale)
        scale_reponse = np.real(np.fft.ifftn(np.divide(temp_num, temp_den)))
        # print(scale_reponse)

        # 6. Calc the new scale
        recovered_scale = np.argmax(scale_reponse)
        self._current_scale_factor = self._current_scale_factor * self.scaleFactors[recovered_scale]

        if self._current_scale_factor < self.min_scale_factor:
            self._current_scale_factor = self.min_scale_factor
        elif self._current_scale_factor > self.max_scale_factor:
            self._current_scale_factor = self.max_scale_factor

        # 7. Extract samples
        update_patch = utils.get_subwindow(image, self.pos, self.sz,
                                           scale_factor=self._current_scale_factor)
        hog_feature_l = pyhog.features_pedro(update_patch / 255., 1)
        hog_feature_l = np.lib.pad(hog_feature_l, ((1, 1), (1, 1), (0, 0)), 'edge')
        xl = np.multiply(hog_feature_l, self.cos_window[:, :, None])
        xlf = np.fft.fft2(xl, axes=(0, 1))
        new_x_num = np.multiply(self.gf[:, :, None], np.conj(xlf))
        new_x_den = np.real(np.sum(np.multiply(xlf, np.conj(xlf)), axis=2))

        sl = utils.get_scale_subwindow(image, self.pos, self.base_target_size,
                                       self._current_scale_factor * self.scaleSizeFactors,
                                       self.scale_window, self.scale_model_sz)
        slf = np.fft.fftn(sl, axes=[0])
        new_s_num = np.multiply(np.conj(self.ysf[:, None]), slf)
        new_s_den = np.real(np.sum(np.multiply(slf, np.conj(slf)), axis=1))

        # 8. Update translation model
        self.x_num = (1 - self._learning_rate) * self.x_num + self._learning_rate * new_x_num
        self.x_den = (1 - self._learning_rate) * self.x_den + self._learning_rate * new_x_den

        # 9. Update scale model
        self.s_num = (1 - self._learning_rate) * self.s_num + self._learning_rate * new_s_num
        self.s_den = (1 - self._learning_rate) * self.s_den + self._learning_rate * new_s_den

        self.target_size = self.base_target_size * self._current_scale_factor

        return vot.Rectangle(self.pos[1] - self.target_size[1] / 2,
                             self.pos[0] - self.target_size[0] / 2,
                             self.target_size[1],
                             self.target_size[0]
                             )
