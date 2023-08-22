import numpy as np
from numpy import matlib
import scipy.io as sio
from scipy.sparse import lil_matrix, csr_matrix
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, ifftshift, fftn, ifftn
import math
from numpy import linalg
import torch
import time
import tal
import cv2
import os
import glob
import shutil


    
class NLOS_numpy:

    def __init__(self, spatial_dim=64, temporal_dim=4096, bin_resolution = 1e-12):
        self.snr = 8e-1
        self.bin_resolution = bin_resolution ###
        self.c = 3e8 ###
        self.z_trim = 0
        self.width = 1.0
        self.N = spatial_dim
        self.M = temporal_dim
        self.range = self.M * self.c * self.bin_resolution

        self.isdiffuse  = False
        self.isbackprop = False

        """ grid_z """
        self.grid_z = np.tile(np.linspace(0, 1, self.M), (self.N, self.N, 1))  # 1024*32*32
        self.grid_z = self.grid_z.transpose(2, 1, 0)
        if self.isdiffuse:
            self.grid_z = self.grid_z ** 4
        else:
            self.grid_z = self.grid_z ** 2

        """ for PSF """
        self.psf, self.fpsf = self.define_psf()
        if self.isbackprop:
            self.invpsf = np.conj(self.fpsf)
        else:
            self.invpsf = np.conj(self.fpsf) / (abs(self.fpsf) ** 2 + 1 / self.snr)

        """ sampling operator """
        self.mtx, self.mtxi = self.resampling_operator()

    def define_psf(self):

        slope = self.width / self.range
        x = np.linspace(-1, 1, 2 * self.N)
        y = np.linspace(-1, 1, 2 * self.N)
        z = np.linspace( 0, 2, 2 * self.M)
        grid_z, grid_y, grid_x = np.meshgrid(z, y, x, indexing='ij')

        psf = np.abs(((4*slope)**2) * (grid_x**2 + grid_y**2) - grid_z)

        psf = psf == np.tile(np.min(psf, axis=0, keepdims=True), (2 * self.M, 1, 1))
        psf = psf.astype(np.float32)
        psf = psf / np.sum(psf[:, self.N, self.N])
        psf = psf / linalg.norm(np.ravel(psf), 2)

        psf = np.roll(psf, self.N, axis=1)
        psf = np.roll(psf, self.N, axis=2)

        fpsf = fftn(psf)

        return psf, fpsf

    def resampling_operator(self):
        mtx = lil_matrix((self.M ** 2, self.M))             # set sparse matrix
        mtx_tmp = lil_matrix((self.M ** 2, self.M ** 2))
        x = np.linspace(1, self.M ** 2, self.M ** 2)

        # set non-zero elements
        for i in range(self.M ** 2):
            mtx[int(x[i])-1, math.ceil(math.sqrt(x[i]))-1] = 1
            mtx_tmp[i, i] = 1 / math.sqrt(x[i])

        # convert lil_matrix to csr_matrix
        mtx = mtx.tocsr()
        mtx_tmp = mtx_tmp.tocsr()
        mtx = mtx_tmp @ mtx
        mtxi = mtx.T

        K = math.log(self.M) / math.log(2)
        for k in range(0, int(K)):
            mtx  = 0.5 * (mtx[::2, :]  + mtx[1::2, :])
            mtxi = 0.5 * (mtxi[:, ::2] + mtxi[:, 1::2])

        return mtx, mtxi

    def transient_to_albedo(self, transient):
        
        if transient.shape[2] != 4096:
            tmp = transient
            K = 4
            for i in range(K):
                current_tmp = tmp
                tmp = np.resize(tmp, (tmp.shape[0], tmp.shape[1], tmp.shape[2] * 2))
                tmp[:, :, ::2] = current_tmp
                tmp[:, :, 1::2] = current_tmp
            transient = tmp
        
        data = transient.transpose(2, 1, 0)

        
        # Step 1: Scale radiometric component
        data = np.multiply(data, self.grid_z)
        data = np.reshape(data, (self.M, self.M))

        # Step 2: Resample time axis and pad result (R_t)
        tdata = np.zeros((2 * self.M, 2 * self.N, 2 * self.N)) # 2048*64*64
        tdata[:self.M, :self.N, :self.N] = np.reshape( self.mtx @ np.reshape(data, (self.M, self.M)), (self.M, self.N, self.N))

        # Step 3: Convolve with inverse filter and unpad result
        ttvol = ifftn(np.multiply(fftn(tdata), self.invpsf))
        tvol = ttvol[0:self.M, 0:self.N, 0:self.N]

        # Step 4: Resample depth axis and clamp results (R_z^{-1}?)
        vol = np.reshape(self.mtxi @ np.reshape(tvol, (self.M, self.M)), (self.M, self.N, self.N))
        vol = vol.real
        albedo = np.flip(vol, axis=2)
        return albedo


