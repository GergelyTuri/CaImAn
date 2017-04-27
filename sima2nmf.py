# -*- coding: utf-8 -*-
from __future__ import print_function
from builtins import str
from builtins import range
"""
Created on Wed Feb 24 18:39:45 2016

@author: Andrea Giovannucci

For explanation consult at
https://github.com/agiovann/Constrained_NMF/releases/download/v0.4-alpha/Patch_demo.zip
and https://github.com/agiovann/Constrained_NMF

"""
try:
    if __IPYTHON__:
        print('Debugging!')
        # this is used for debugging purposes only. allows to reload classes when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    print('Not IPYTHON')
    pass

import sys
import matplotlib
matplotlib.use('Agg')
import numpy as np
import psutil
import glob
import os
import scipy
import argparse
from ipyparallel import Client
# mpl.use('Qt5Agg')
import pylab as pl
pl.ion()

from sima import ImagingDataset
from sima.ROI import ROI
from sima.ROI import ROIList

import caiman as cm
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.components_evaluation import evaluate_components
from caiman.utils.visualization import plot_contours,view_patches_bar
from caiman.utils.visualization import pdf_patches_bar
from caiman.base.rois import extract_binary_masks_blob

sima_files = {
    '765': '/data2/jack/2p/jb10/20151030/TSeries-10302015-1019-002/aligned_TSeries-10302015-1019-002.sima',
    '2052': '/data2/jack/2p/jb40/20160306/TSeries-03062016-1514-001/TSeries-03062016-1514-001.sima'
}
mmap_files = {
    '765': '/data2/jack/2p/jb10/20151030/TSeries-10302015-1019-002/Yr_d1_472_d2_483_d3_1_order_C_frames_500_.mmap',
    '2052': '/data2/jack/2p/jb40/20160306/TSeries-03062016-1514-001/Yr_d1_484_d2_482_d3_1_order_C_frames_27300_.mmap'
}

def exportRois(sima_path, contours):
    ds = ImagingDataset.load(sima_path, contours)
    im_shape = ds.frame_shape[:3]
    rois = np.load(os.path.join('/data2/jack','rois.npz'))['rois']
    roi_list = []
    for roi in contours:
        coords = contours['coordinates']
        coords = np.hstack((coords,np.zeros((coords.shape[0],1))))
        segment_boundaries = np.where(np.isnan(coords[:,0]))[0]
        polys = []
        for i in xrange(1,len(segment_boundaries)):
            polys.append(
                coords[(segment_boundaries[i-1]+1):segment_boundaries[i],:])
        roi_list.append(ROI(polygons = polys, label=roi['neuron_id'],
            id=roi['neuron_id'], im_shape=im_shape))

    ds.add_ROIs(ROIList(roi_list), label='cnmf_rois')

def main(argv):
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-d', '--dendrites', action='store_true',
        help='is dendrites, possibly inconsistant results on somas')
    argParser.add_argument('-p', '--patch_size', action='store', type=int,
        default=200, help='set patch size to run the algorithm on patches, 0 \
        indicates not to use patches, default=200')
    argParser.add_argument('-n', '--neuron_count', action='store',
        type=int, default=20, help='expected neuron count per patch default=20')
    argParser.add_argument('-z', '--z_plane', action='store', type=int,
        default=-1, help='z-plane of the sima file to extract, -1 will \
        compute the average of all planes. default=-1')
    argParser.add_argument(
        "filename", action="store", type=str, default="",
        help=("Process any experiment that has a tSeriesDirectory containing" +
              "'directory'"))
    args = argParser.parse_args()

    c,dview,n_processes = cm.cluster.setup_cluster(backend='local',
        n_processes=None, single_thread=False)

    if args.patch_size:
        is_patches = True
    else:
        is_patches = False
    is_dendrites = args.dendrites

    if is_dendrites == True:
    # THIS METHOd CAN GIVE POSSIBLY INCONSISTENT RESULTS ON SOMAS WHEN NOT USED
    # WITH PATCHES
        init_method = 'sparse_nmf'
        alpha_snmf=10e1  # this controls sparsity
    else:
        init_method = 'greedy_roi'
        alpha_snmf=None #10e2  # this controls sparsity

    ext = os.path.splitext(args.filename)[1]
    if ext == '.sima':
        fname = args.filename
        print(fname)

        # TODO: change back to 300 (?)
        add_to_movie=0 # the movie must be positive!!!
        downsample_factor=1 # use .2 or .1 if file is large and you want a quick answer
        idx_xy=None
        base_name='Yr'
        fname_new=cm.save_memmap_from_sima(fname, base_name=base_name,
            resize_fact=(1, 1, downsample_factor), remove_init=0,idx_xy=idx_xy,
            add_to_movie=add_to_movie, plane=args.z_plane)
    elif ext == '.mmap':
        fname_new = args.filename
        sima_files = glob.glob(os.path.join(os.path.split(args.filename)[0],
            '*.sima'))
        if not len(sima_files):
            raise Exception('sima file not found')
        elif len(sima_files) > 1:
            raise Exception('multiple sima files in TSeries Directory')
        fname = sima_files[0]
        print(fname)
    else:
        raise Exception('filetype not supported')

    Yr, dims, T = cm.load_memmap(fname_new)
    d1, d2 = dims
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    Y = np.reshape(Yr, dims + (T,), order='F')

    if np.min(images)<0:
        raise Exception('Movie too negative, add_to_movie should be larger')
    if np.sum(np.isnan(images))>0:
        raise Exception('Movie contains nan! You did not remove enough borders')

    Cn = cm.local_correlations(Y[:,:,:3000])

    K = args.neuron_count
    #%%
    if not is_patches:
        #%%
        gSig = [10, 10]  # expected half size of neurons
        merge_thresh = 0.8  # merging threshold, max correlation allowed
        p = 2  # order of the autoregressive system
        cnm = cnmf.CNMF(n_processes, method_init=init_method, k=K, gSig=gSig,
            merge_thresh=merge_thresh, p=p, dview=dview, Ain=None,
            method_deconvolution='oasis',skip_refinement = False)
        cnm = cnm.fit(images)
        crd = plot_contours(cnm.A, Cn, thr=0.9)
    #%%
    else:
        # half-size of the patches in pixels. rf=25, patches are 50x50
        rf = int(args.patch_size/2)
        stride = 15  # amounpl.it of overlap between the patches in pixels
        gSig = [10, 10]  # expected half size of neurons
        merge_thresh = 0.8  # merging threshold, max correlation allowed
        p = 1  # order of the autoregressive system
        save_results = False
        #%% RUN ALGORITHM ON PATCHES

        cnm = cnmf.CNMF(n_processes, k=K, gSig=gSig, merge_thresh=0.8, p=0,
            dview=dview, Ain=None, rf=rf, stride=stride, memory_fact=1,
            method_init=init_method, alpha_snmf=alpha_snmf, only_init_patch=True,
            gnb=1,method_deconvolution='oasis')
        cnm = cnm.fit(images)

        A_tot = cnm.A
        C_tot = cnm.C
        YrA_tot = cnm.YrA
        b_tot = cnm.b
        f_tot = cnm.f
        sn_tot = cnm.sn

        print(('Number of components:' + str(A_tot.shape[-1])))
        #%%
        pl.figure()
        crd = plot_contours(A_tot, Cn, thr=0.9)
        #%%
        final_frate = 10# approx final rate  (after eventual downsampling )
        Npeaks = 10
        traces = C_tot + YrA_tot
        fitness_raw, fitness_delta, erfc_raw, erfc_delta, r_values, \
            significant_samples = evaluate_components(Y, traces, A_tot, C_tot,
                b_tot, f_tot, final_frate, remove_baseline=True, N=5,
                robust_std=False, Athresh=0.1, Npeaks=Npeaks,  thresh_C=0.3)

        idx_components_r = np.where(r_values >= .5)[0]
        idx_components_raw = np.where(fitness_raw < -40)[0]
        idx_components_delta = np.where(fitness_delta < -20)[0]

        idx_components = np.union1d(idx_components_r, idx_components_raw)
        idx_components = np.union1d(idx_components, idx_components_delta)
        idx_components_bad = np.setdiff1d(list(range(len(traces))), idx_components)

        print(('Keeping ' + str(len(idx_components)) +
               ' and discarding  ' + str(len(idx_components_bad))))
        #%%
        pl.figure()
        crd = plot_contours(A_tot.tocsc()[:, idx_components], Cn, thr=0.9)
        #%%
        A_tot = A_tot.tocsc()[:, idx_components]
        C_tot = C_tot[idx_components]
        #%%
        save_results = True
        if save_results:
            np.savez('results_analysis_patch2.npz', A_tot=A_tot, C_tot=C_tot,
                YrA_tot=YrA_tot, sn_tot=sn_tot, d1=d1, d2=d2, b_tot=b_tot, f=f_tot)

        #%%
        cnm = cnmf.CNMF(n_processes, k=A_tot.shape, gSig=gSig,
            merge_thresh=merge_thresh, p=p, dview=dview, Ain=A_tot, Cin=C_tot,
            f_in=f_tot, rf=None, stride=None, method_deconvolution='oasis')
        cnm = cnm.fit(images)

    #%%
    A, C, b, f, YrA, sn = cnm.A, cnm.C, cnm.b, cnm.f, cnm.YrA, cnm.sn
    #%%
    final_frate = 10

    Npeaks = 10
    traces = C + YrA
    fitness_raw, fitness_delta, erfc_raw, erfc_delta, r_values, \
        significant_samples = evaluate_components(Y, traces, A, C, b, f,
            final_frate, remove_baseline=True, N=5, robust_std=False, Athresh=0.1,
            Npeaks=Npeaks,  thresh_C=0.3)

    idx_components_r = np.where(r_values >= .95)[0]
    idx_components_raw = np.where(fitness_raw < -100)[0]
    idx_components_delta = np.where(fitness_delta < -100)[0]

    #min_radius = gSig[0] - 2
    #masks_ws, idx_blobs, idx_non_blobs = extract_binary_masks_blob(
    #    A.tocsc(), min_radius, dims, num_std_threshold=1,
    #    minCircularity=0.7, minInertiaRatio=0.2, minConvexity=.5)

    idx_components = np.union1d(idx_components_r, idx_components_raw)
    idx_components = np.union1d(idx_components, idx_components_delta)
    #idx_blobs = np.intersect1d(idx_components, idx_blobs)
    idx_components_bad = np.setdiff1d(list(range(len(traces))), idx_components)

    print(' ***** ')
    print((len(traces)))
    print((len(idx_components)))

    save_results = True
    if save_results:
        np.savez(os.path.join(os.path.split(fname_new)[0], 'results_analysis.npz'),
        Cn=Cn, A=A.todense(), C=C, b=b, f=f, YrA=YrA, sn=sn, d1=d1, d2=d2,
        idx_components=idx_components, idx_components_bad=idx_components_bad)

    crd = plot_contours(A.tocsc(), Cn, thr=0.9,swap_dim=True)

    #%% visualize components
    pl.figure()
    pl.subplot(1, 2, 1)
    crd = plot_contours(A.tocsc()[:, idx_components], Cn, thr=0.9)
    pl.subplot(1, 2, 2)
    crd = plot_contours(A.tocsc()[:, idx_components_bad], Cn, thr=0.9)
    pl.savefig(os.path.join(fname, 'cnmf.pdf'))

    pdf_patches_bar(os.path.join(fname, 'cnmf_plots1.pdf'), Yr,
        scipy.sparse.coo_matrix(A.tocsc()[:, idx_components[:10]]),
        C[idx_components[:10], :], b, f, dims[0], dims[1], YrA=YrA[idx_components[:10], :],
        img=Cn)

    pdf_patches_bar(os.path.join(fname, 'cnmf_plots2.pdf'),
        Yr, scipy.sparse.coo_matrix(A.tocsc()[:, idx_components_bad[:5]]),
        C[idx_components_bad[:5], :], b, f, dims[0], dims[1],
        YrA=YrA[idx_components_bad[:5], :], img=Cn)
    #%% STOP CLUSTER and clean up log files

    cm.stop_server()

    log_files = glob.glob('Yr*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)

    export_rois(fname, crd)

if __name__ == '__main__':
    main(sys.argv[1:])
