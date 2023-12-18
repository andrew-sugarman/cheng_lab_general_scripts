import sys,os,glob

os.environ['NUMEXPR_MAX_THREADS']='272'

import tomopy
import numpy as np
import tifffile
from pathlib import Path
from multiprocessing import Pool, Process, Manager, cpu_count#, shared_memory
from datetime import datetime
from scipy import interpolate
import cv2
import shutil
from scipy import interpolate
from scipy.signal.windows import gaussian
from scipy.ndimage import median_filter
from scipy.ndimage import binary_dilation
# import scipy.fftpack as fft
import pyfftw.interfaces.scipy_fftpack as fft
# import dxchange as dx
def detect_stripe(list_data, snr):
    """
    Locate stripe positions using Algorithm 4 in Ref. [1].
    Parameters
    ----------
    list_data : array_like
        1D array. Normalized data.
    snr : float
        Ratio used to segment stripes from background noise.
    Returns
    -------
    ndarray
        1D binary mask.
    References
    ----------
    .. [1] https://doi.org/10.1364/OE.26.028396
    """
    npoint = len(list_data)
    list_sort = np.sort(list_data)
    listx = np.arange(0, npoint, 1.0)
    ndrop = np.int16(0.25 * npoint)
    (slope, intercept) = np.polyfit(listx[ndrop:-ndrop - 1], list_sort[ndrop:-ndrop - 1], 1)
    y_end = intercept + slope * listx[-1]
    noise_level = np.abs(y_end - intercept)
    noise_level = np.clip(noise_level, 1e-6, None)
    val1 = np.abs(list_sort[-1] - y_end) / noise_level
    val2 = np.abs(intercept - list_sort[0]) / noise_level
    list_mask = np.zeros(npoint, dtype=np.float32)
    if val1 >= snr:
        upper_thresh = y_end + noise_level * snr * 0.5
        list_mask[list_data > upper_thresh] = 1.0
    if val2 >= snr:
        lower_thresh = intercept - noise_level * snr * 0.5
        list_mask[list_data <= lower_thresh] = 1.0
    return list_mask

def remove_unresponsive_and_fluctuating_stripe(sinogram, snr, size, residual=False):
    """
    Remove unresponsive or fluctuating stripes, algorithm 6 in Ref. [1], by:
    locating stripes, correcting using interpolation. Angular direction
    is along the axis 0.
    Parameters
    ----------
    sinogram : array_like
        2D array. Sinogram image.
    snr : float
        Ratio used to segment stripes from background noise.
    size : int
        Window size of the median filter.
    residual : bool, optional
        Removing residual stripes if True.
    Returns
    -------
    ndarray
        2D array. Stripe-removed sinogram.
    References
    ----------
    .. [1] https://doi.org/10.1364/OE.26.028396
    """
    sinogram = np.copy(sinogram)  # Make it mutable
    (nrow, _) = sinogram.shape
    sino_smooth = np.apply_along_axis(uniform_filter1d, 0, sinogram, 10)
    list_diff = np.sum(np.abs(sinogram - sino_smooth), axis=0)
    list_diff_bck = median_filter(list_diff, size)
    list_fact = np.divide(list_diff, list_diff_bck,
                         out=np.ones_like(list_diff), where=list_diff_bck != 0)
    list_mask = detect_stripe(list_fact, snr)
    list_mask = np.float32(binary_dilation(list_mask, iterations=1))
    list_mask[0:2] = 0.0
    list_mask[-2:] = 0.0
    listx = np.where(list_mask < 1.0)[0]
    listy = np.arange(nrow)
    mat = sinogram[:, listx]
    finter = interpolate.interp2d(listx, listy, mat, kind='linear')
    listx_miss = np.where(list_mask > 0.0)[0]
    if len(listx_miss) > 0:
        sinogram[:, listx_miss] = finter(listx_miss, listy)
    if residual is True:
        sinogram = remove_large_stripe(sinogram, snr, size)
    return sinogram

def remove_large_stripe(sinogram, snr=3, size=36, drop_ratio=0.1, norm=True):
    """
    Remove large stripes, algorithm 5 in Ref. [1], by: locating stripes,
    normalizing to remove full stripes, and using the sorting technique
    (Ref. [1]) to remove partial stripes. Angular direction is along the
    axis 0.
    Parameters
    ----------
    sinogram : array_like
        2D array. Sinogram image
    snr : float
        Ratio used to segment stripes from background noise.
    size : int
        Window size of the median filter.
    drop_ratio : float, optional
        Ratio of pixels to be dropped, which is used to reduce the false
        detection of stripes.
    norm : bool, optional
        Apply normalization if True.
    Returns
    -------
    ndarray
        2D array. Stripe-removed sinogram.
    References
    ----------
    .. [1] https://doi.org/10.1364/OE.26.028396
    """
    sinogram = np.copy(sinogram)  # Make it mutable
    drop_ratio = np.clip(drop_ratio, 0.0, 0.8)
    (nrow, ncol) = sinogram.shape
    ndrop = int(0.5 * drop_ratio * nrow)
    sino_sort = np.sort(sinogram, axis=0)
    sino_smooth = median_filter(sino_sort, (1, size))
    list1 = np.mean(sino_sort[ndrop:nrow - ndrop], axis=0)
    list2 = np.mean(sino_smooth[ndrop:nrow - ndrop], axis=0)
    list_fact = np.divide(list1, list2,
                         out=np.ones_like(list1), where=list2 != 0)
    list_mask = detect_stripe(list_fact, snr)
    list_mask = np.float32(binary_dilation(list_mask, iterations=1))
    mat_fact = np.tile(list_fact, (nrow, 1))
    if norm is True:
        sinogram = sinogram / (mat_fact+.00000000000000000001)  # Normalization
    sino_tran = np.transpose(sinogram)
    list_index = np.arange(0.0, nrow, 1.0)
    mat_index = np.tile(list_index, (ncol, 1))
    mat_comb = np.asarray(np.dstack((mat_index, sino_tran)))
    mat_sort = np.asarray(
        [row[row[:, 1].argsort()] for row in mat_comb])
    mat_sort[:, :, 1] = np.transpose(sino_smooth)
    mat_sort_back = np.asarray(
        [row[row[:, 0].argsort()] for row in mat_sort])
    sino_cor = np.transpose(mat_sort_back[:, :, 1])
    listx_miss = np.where(list_mask > 0.0)[0]
    sinogram[:, listx_miss] = sino_cor[:, listx_miss]
    return sinogram
def remove_stripe_based_sorting(sinogram, size=5, dim=1):
    """
    Remove stripe artifacts in a sinogram using the sorting technique,
    algorithm 3 in Ref. [1]. Angular direction is along the axis 0.
    Parameters
    ----------
    sinogram : array_like
        2D array. Sinogram image.
    size : int
        Window size of the median filter.
    dim : {1, 2}, optional
        Dimension of the window.
    Returns
    -------
    ndarray
        2D array. Stripe-removed sinogram.
    References
    ----------
    .. [1] https://doi.org/10.1364/OE.26.028396
    """
    sinogram = np.transpose(sinogram)
    (nrow, ncol) = sinogram.shape
    list_index = np.arange(0.0, ncol, 1.0)
    mat_index = np.tile(list_index, (nrow, 1))
    mat_comb = np.asarray(np.dstack((mat_index, sinogram)))
    mat_sort = np.asarray(
        [row[row[:, 1].argsort()] for row in mat_comb])
    if dim == 2:
        mat_sort[:, :, 1] = median_filter(mat_sort[:, :, 1], (size, size))
    else:
        mat_sort[:, :, 1] = median_filter(mat_sort[:, :, 1], (size, 1))
    mat_sort_back = np.asarray(
        [row[row[:, 0].argsort()] for row in mat_sort])
    return np.transpose(mat_sort_back[:, :, 1])

def remove_stripe_based_interpolation(sinogram, snr=3, size=36, drop_ratio=0.1, norm=True):
    """
    Combination of algorithm 4, 5, and 6 in Ref. [1].
    Remove stripes using a detection technique and an interpolation method.
    Angular direction is along the axis 0.
    Parameters
    ----------
    sinogram : array_like
        2D array. Sinogram image
    snr : float
        Ratio used to segment between useful information and noise.
    size : int
        Window size of the median filter used to detect stripes.
    drop_ratio : float, optional
        Ratio of pixels to be dropped, which is used to to reduce
        the possibility of the false detection of stripes.
    norm : bool, optional
        Apply normalization if True.
    Returns
    -------
    ndarray
        2D array. Stripe-removed sinogram.
    References
    ----------
    .. [1] https://doi.org/10.1364/OE.26.028396
    """
    drop_ratio = np.clip(drop_ratio, 0.0, 0.8)
    sinogram = np.copy(sinogram)
    (nrow, ncol) = sinogram.shape
    ndrop = int(0.5 * drop_ratio * nrow)
    sino_sort = np.sort(sinogram, axis=0)
    sino_smooth = median_filter(sino_sort, (1, size))
    list1 = np.mean(sino_sort[ndrop:nrow - ndrop], axis=0)
    list2 = np.mean(sino_smooth[ndrop:nrow - ndrop], axis=0)
    list_fact = np.divide(list1, list2,
                          out=np.ones_like(list1), where=list2 != 0)
    list_mask = detect_stripe(list_fact, snr)
    list_mask = np.float32(binary_dilation(list_mask, iterations=1))
    mat_fact = np.tile(list_fact, (nrow, 1))
    if norm is True:
        sinogram = sinogram / (mat_fact+.00000000001)
        
    list_mask[0:2] = 0.0
    list_mask[-2:] = 0.0
    listx = np.where(list_mask < 1.0)[0]
    listy = np.arange(nrow)
    matz = sinogram[:, listx]
    finter = interpolate.interp2d(listx, listy, matz, kind='linear')
    listx_miss = np.where(list_mask > 0.0)[0]
    if len(listx_miss) > 0:
        sinogram[:, listx_miss] = finter(listx_miss, listy)
    return sinogram
    

def shared_mem_reader2(args):

    list_index, shm_name, size,multi_proc_pass_list,multi_proc_pass_list_idx,target_proj_y,gain_path,gain_path_post,dark_path,dark_path_post = args
    print(f'landed in shared memory reader2 as index {list_index}')
    

    
    image_paths=multi_proc_pass_list[list_index]
    paths_idx=multi_proc_pass_list_idx[list_index]
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    sino_tiff_dump_mapped = np.ndarray(size, dtype=np.float32, buffer=existing_shm.buf)
    count=0
    incoming_proj_map=tifffile.memmap(image_paths[0], dtype=np.uint16)
    for idx_,image_path in enumerate(image_paths):
        idx=paths_idx[idx_]
        if idx>(sino_tiff_dump_mapped.shape[0]/2):
            gain_path=gain_path_post
            dark_path=dark_path_post
        incoming_gain_map=tifffile.memmap(gain_path, dtype=np.uint16)
        incoming_dark_map=tifffile.memmap(dark_path, dtype=np.uint16)
        incoming_proj_map=tifffile.memmap(image_path, dtype=np.uint16)
        sino_tiff_dump_mapped[paths_idx[idx]:,:]=np.divide(incoming_proj_map[target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:]-incoming_dark_map[target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:],incoming_gain_map[target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:]-incoming_dark_map[target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:]).astype(np.float32)
        if count>2:
            print(f'opened img {idx_} of {len(image_paths)} from list index {list_index} read timing: {(datetime.now()-read_timing_internal)}')
            count=0
            read_timing_internal=datetime.now()
        count+=1
        read_timing_internal=datetime.now()
        
    existing_shm.close()
    print(list_index, 'parent process:', os.getppid())
    print(list_index, 'process id:', os.getpid())
    print(f'existing_shm closed')

def get_image_paths(folder):
    return sorted(glob.glob(os.path.join(folder,'*.tif*')))
def delete_all_psm_here(folder):
    paths=glob.glob(os.path.join(folder,'psm_*'))
    for path in paths:
        print(f'removing {path}')
        try:
            Path.unlink(path)
        except:
            print("didn't own that one")
def get_image_paths_gains(folder):
    # return sorted(glob.glob(os.path.join(folder,'*','*.tif*')))
    return sorted(glob.glob(os.path.join(folder,'*.tif*')))
def center_finding(in_list):
    [idx,[proj_mtx,theta,out_dir,vertical_velocity,idx_,skew_angle_]],pc_center=in_list
    print(idx)
    butts=False
    if butts:
        butterworthpars = [.2,2]
        recon = tomopy.recon(proj_mtx, theta,center=pc_center, algorithm='gridrec',filter_name='butterworth', filter_par=butterworthpars)
    else:
        recon = tomopy.recon(proj_mtx, theta,center=pc_center, algorithm='gridrec')
    tifffile.imwrite(os.path.join(out_dir,str(idx_)+'_'+str(pc_center)+'_PXvelocity_'+str(vertical_velocity)+'_sc_'+skew_angle_+'.tif'),recon)
    del proj_mtx
    del recon
def find_skipped_frames(inlist):
    log_path,trigger_timing_us,proj_per_section=inlist
    logging_dictionary={}
    with open(log_path, 'r') as f:
        next(f)
        lines = f.read().splitlines()
        for idx,line in enumerate(lines):
            if len(line)==0:
                continue
            if idx == 0:
                line_split=line.split('\t')
                # initial_value=int(line_split[-3])
                initial_value=int(line_split[-3])
                # print(initial_value)
                
            line_split=line.split('\t')
            # print(int(line_split[0].split('_')[-1]))
            logging_dictionary[int(line_split[0].split('_')[-1])]=int(line_split[-3])
    # print(logging_dictionary)
    
    sorted_proj_timings={}
    timing_pre=-1
    pre_val=-1
    skipped_frame_count=0
    real_projection_order=[]
    skipped_frames=[]
    skipped_frame_index=[]
    sino_frame_skip_order=[]
    average_timing_list=[]
    count=0
    skipped_frame_index_value={}
    for key, value in sorted(logging_dictionary.items()): # Note the () after items!
        timing=(value-pre_val)
        # print(timing)
        average_timing_list.append(timing)
        pre_val=value
        count+=1
    avg_timing=(sum(average_timing_list)-initial_value)/len(average_timing_list)
    proj_count=0
    for key, value in sorted(logging_dictionary.items()): # Note the () after items!
        sorted_proj_timings[key]=value
        timing=(value-timing_pre)
        real_projection_order.append(timing)
        sino_frame_skip_order.append(int(round(timing-timing_pre))-1)
        # print(key)
        
            
        
        # print(f'avg timing: {avg_timing} difference timing: {timing-timing_pre} timing {timing} timingDiff {timing*.0001} valeu: {value} initValue:{initial_value}')
        if (timing)>avg_timing*1.5 and key>1:
            skipped_frame_count+=1
            skipped_frames.append(key)
            skipped_frame_index.append(proj_count)
            skipped_frame_index_value[key]=round(timing/avg_timing)
        skipped_frame_index_value[key]=round(timing/avg_timing)
            
        timing_pre=value
        proj_count+=1
    missing_proj=[]
    for idx in  range(0,proj_per_section): #make this match logs. i don't actualy use missing_proj though.
        if idx not in sorted_proj_timings:
            missing_proj.append(idx)
    # print(sino_frame_skip_order)
    # print(skipped_frames)
    # print(skipped_frame_index)
    # print(skipped_frame_index_value)
    skipped_frame_index_value[0]=1
    skipped_frame_index_value[1]=1
    return([missing_proj,np.array(sino_frame_skip_order),real_projection_order,skipped_frame_index,skipped_frame_index_value])


def parallel_sectional_write_listdist(in_list):
    [list_index,image_path, npy_map_or_filename_tif ,slope_function,tiff_intermediate_for_sino_gen,already_gain_corrected,sino_frame_skip_order,going_up,target_proj_y,gain_path,gain_path_post,dark_path,dark_path_post,filenames_chunk,filenames_chunk_idx]=in_list
    paths=filenames_chunk[list_index]
    paths_idx=filenames_chunk_idx[list_index]
    
    if tiff_intermediate_for_sino_gen:
        sino_tiff_dump_mapped=tifffile.memmap(npy_map_or_filename_tif, dtype=np.float32)


    # # # # for idx_,image_path in enumerate(paths):
        # # # # idx=paths_idx[idx_]
        # # # # incoming_proj_map=tifffile.memmap(image_path, dtype=np.uint16)
        
        # # # # if idx>(sino_tiff_dump_mapped.shape[0]/2):
            # # # # gain_path=gain_path_post
            # # # # dark_path=dark_path_post
        # # # # if not already_gain_corrected:
            # # # # incoming_gain_map=tifffile.memmap(gain_path, dtype=np.uint16)
            # # # # incoming_dark_map=tifffile.memmap(dark_path, dtype=np.uint16)
    incoming_gain_pre=tifffile.imread(gain_path)
    incoming_dark_pre=tifffile.imread(dark_path)

    incoming_gain_post=tifffile.imread(gain_path_post)
    incoming_dark_post=tifffile.imread(dark_path_post)
    
    
    for idx_,image_path in enumerate(paths):
        idx=paths_idx[idx_]
        incoming_proj_map=tifffile.memmap(image_path, dtype=np.uint16)
        
        incoming_gain_map=incoming_gain_pre
        incoming_dark_map=incoming_dark_pre
        
        if idx>(sino_tiff_dump_mapped.shape[0]/2):
            gain_path=gain_path_post
            dark_path=dark_path_post
            incoming_gain_map=incoming_gain_post
            incoming_dark_map=incoming_dark_post

        if not already_gain_corrected:
            # print(incoming_gain_map.shape)
            
            # print(incoming_dark_map.shape)
            # print(incoming_proj_map.shape)
            # print(sino_tiff_dump_mapped.shape)
            # print(incoming_proj_map[0,target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:].shape)
            # print(incoming_gain_map[target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:].shape)
            # print(incoming_dark_map[target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:].shape)
            # print(np.divide(incoming_proj_map[0,target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:]-incoming_dark_map[target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:],incoming_gain_map[target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:]-incoming_dark_map[target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:]).shape)
            if len(incoming_gain_map.shape)>2 and len(incoming_proj_map.shape)>2:
                sino_tiff_dump_mapped[idx,:,:]=np.divide(incoming_proj_map[0,target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:]-incoming_dark_map[0,target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:],incoming_gain_map[0,target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:]-incoming_dark_map[0,target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:]).astype(np.float32)
            elif len(incoming_proj_map.shape)>2:
                sino_tiff_dump_mapped[idx,:,:]=np.divide(incoming_proj_map[0,target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:]-incoming_dark_map[target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:],incoming_gain_map[target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:]-incoming_dark_map[target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:]).astype(np.float32)
            else:
                sino_tiff_dump_mapped[idx,:,:]=np.divide(incoming_proj_map[target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:]-incoming_dark_map[target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:],incoming_gain_map[target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:]-incoming_dark_map[target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:]).astype(np.float32)

        else:
            sino_tiff_dump_mapped[idx,:,:]=incoming_proj_map[target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:]


def parallel_sectional_write_plusprocessing_listdist(in_list):
    [list_index,image_path, npy_map_or_filename_tif ,slope_function,tiff_intermediate_for_sino_gen,already_gain_corrected,sino_frame_skip_order,going_up,target_proj_y,gain_path,gain_path_post,dark_path,dark_path_post,filenames_chunk,filenames_chunk_idx,skew_angle,d,s1,s2,center_of_rotation]=in_list
    paths=filenames_chunk[list_index]
    paths_idx=filenames_chunk_idx[list_index]
    
    if tiff_intermediate_for_sino_gen:
        sino_tiff_dump_mapped=tifffile.memmap(npy_map_or_filename_tif, dtype=np.float32)
    
    incoming_gain_pre=tifffile.imread(gain_path)
    incoming_dark_pre=tifffile.imread(dark_path)

    incoming_gain_post=tifffile.imread(gain_path_post)
    incoming_dark_post=tifffile.imread(dark_path_post)
    
    
    
    
    count=0
    for idx_,image_path in enumerate(paths):
        idx=paths_idx[idx_]
        incoming_proj_map=tifffile.memmap(image_path, dtype=np.uint16)
        
        incoming_gain_map=incoming_gain_pre
        incoming_dark_map=incoming_dark_pre
        
        # if count>99:
            # print(f'image index {idx}list chunk index: {idx_} list index: {list_index} sino tiff map cutoff {sino_tiff_dump_mapped.shape[0]/2}')
            # count=0
        # count+=1
            
        
        
        if idx>(sino_tiff_dump_mapped.shape[0]/2):
            gain_path=gain_path_post
            dark_path=dark_path_post
            incoming_gain_map=incoming_gain_post
            incoming_dark_map=incoming_dark_post

        if not already_gain_corrected:
            # incoming_gain_map=tifffile.memmap(gain_path, dtype=np.uint16)
            # incoming_dark_map=tifffile.memmap(dark_path, dtype=np.uint16)
            # print(incoming_gain_map.shape)
            
            # print(incoming_dark_map.shape)
            # print(incoming_proj_map.shape)
            # print(sino_tiff_dump_mapped.shape)
            # print(incoming_proj_map[0,target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:].shape)
            # print(incoming_gain_map[target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:].shape)
            # print(incoming_dark_map[target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:].shape)
            # print(np.divide(incoming_proj_map[0,target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:]-incoming_dark_map[target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:],incoming_gain_map[target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:]-incoming_dark_map[target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:]).shape)
            if len(incoming_gain_map.shape)>2 and len(incoming_proj_map.shape)>2:
                image = cv2.bilateralFilter(np.divide(incoming_proj_map[0,target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:]-incoming_dark_map[0,target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:],incoming_gain_map[0,target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:]-incoming_dark_map[0,target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:]).astype(np.float32),d,s1,s2)
                # image = np.divide(incoming_proj_map[0,target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:]-incoming_dark_map[0,target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:],incoming_gain_map[0,target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:]-incoming_dark_map[0,target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:])


                np.log(image, out=image)
                
                if skew_angle:
                    h,w = image.shape[:2]
                    cX,cY = (center_of_rotation,-target_proj_y)
                    M = cv2.getRotationMatrix2D((cX,cY),skew_angle,1)
                    image = cv2.warpAffine(image,M , (w,h))

                sino_tiff_dump_mapped[idx,:,:]=-1*image
            
            
            elif len(incoming_proj_map.shape)>2:
                # image = np.divide(incoming_proj_map[0,target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:]-incoming_dark_map[target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:],incoming_gain_map[target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:]-incoming_dark_map[target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:]).astype(np.float32)
                image = cv2.bilateralFilter(np.divide(incoming_proj_map[0,target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:]-incoming_dark_map[target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:],incoming_gain_map[target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:]-incoming_dark_map[target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:]).astype(np.float32),d,s1,s2)

                np.log(image, out=image)
                
                if skew_angle:
                    h,w = image.shape[:2]
                    cX,cY = (center_of_rotation,-target_proj_y)
                    M = cv2.getRotationMatrix2D((cX,cY),skew_angle,1)
                    image = cv2.warpAffine(image,M , (w,h))
                
                sino_tiff_dump_mapped[idx,:,:]=-1*image
            
            else:

                image = cv2.bilateralFilter(np.divide(incoming_proj_map[target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:]-incoming_dark_map[target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:],incoming_gain_map[target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:]-incoming_dark_map[target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:]).astype(np.float32),d,s1,s2)

                np.log(image, out=image)
                if skew_angle:
                    h,w = image.shape[:2]
                    cX,cY = (center_of_rotation,-target_proj_y)
                    M = cv2.getRotationMatrix2D((cX,cY),skew_angle,1)
                    image = cv2.warpAffine(image,M , (w,h))
                
                sino_tiff_dump_mapped[idx,:,:]=-1*image

        else:
            sino_tiff_dump_mapped[idx,:,:]=incoming_proj_map[target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:]



def parallel_sectional_write(in_list):
    [idx,image_path, npy_map_or_filename_tif ,slope_function,tiff_intermediate_for_sino_gen,already_gain_corrected,sino_frame_skip_order,going_up,target_proj_y,gain_path,gain_path_post,dark_path,dark_path_post]=in_list
    if tiff_intermediate_for_sino_gen:
        sino_tiff_dump_mapped=tifffile.memmap(npy_map_or_filename_tif, dtype=np.float32)

    incoming_proj_map=tifffile.memmap(image_path, dtype=np.uint16)
    
    if idx>(sino_tiff_dump_mapped.shape[0]/2):
        gain_path=gain_path_post
        dark_path=dark_path_post
    if not already_gain_corrected:
        incoming_gain_map=tifffile.memmap(gain_path, dtype=np.uint16)
        incoming_dark_map=tifffile.memmap(dark_path, dtype=np.uint16)
        # print(incoming_gain_map.shape)
        
        # print(incoming_dark_map.shape)
        # print(incoming_proj_map.shape)
        # print(sino_tiff_dump_mapped.shape)
        # print(incoming_proj_map[0,target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:].shape)
        # print(incoming_gain_map[target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:].shape)
        # print(incoming_dark_map[target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:].shape)
        # print(np.divide(incoming_proj_map[0,target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:]-incoming_dark_map[target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:],incoming_gain_map[target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:]-incoming_dark_map[target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:]).shape)
        if len(incoming_gain_map.shape)>2 and len(incoming_proj_map.shape)>2:
            sino_tiff_dump_mapped[idx,:,:]=np.divide(incoming_proj_map[0,target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:]-incoming_dark_map[0,target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:],incoming_gain_map[0,target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:]-incoming_dark_map[0,target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:]).astype(np.float32)
        elif len(incoming_proj_map.shape)>2:
            sino_tiff_dump_mapped[idx,:,:]=np.divide(incoming_proj_map[0,target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:]-incoming_dark_map[target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:],incoming_gain_map[target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:]-incoming_dark_map[target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:]).astype(np.float32)
        else:
            sino_tiff_dump_mapped[idx,:,:]=np.divide(incoming_proj_map[target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:]-incoming_dark_map[target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:],incoming_gain_map[target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:]-incoming_dark_map[target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:]).astype(np.float32)

    else:
        sino_tiff_dump_mapped[idx,:,:]=incoming_proj_map[target_proj_y:target_proj_y+sino_tiff_dump_mapped.shape[1],:]

        
def parallel_helical_write(in_list):
    #grab inlist and reparse into unlisted variables
    # [idx,image_path, filename_tif ,slope_function]=in_list
    [idx,image_path, npy_map_or_filename_tif ,slope_function,tiff_intermediate_for_sino_gen,already_gain_corrected,sino_frame_skip_order,going_up,target_proj_y]=in_list
    
    #interpolate where in the image to start grabbing your projection chunk
    interpolated_ystart_pos=int(round(float(slope_function(idx+np.sum(sino_frame_skip_order[:idx])))))

    if tiff_intermediate_for_sino_gen:
        #memmap the 3D tiff you set up earlier to dump data into
        sino_tiff_dump_mapped=tifffile.memmap(npy_map_or_filename_tif, dtype=np.float32)

    #memmap the projection and gain data based on the interpolated y start position
    if already_gain_corrected:
        incoming_proj_map=tifffile.memmap(image_path, dtype=np.float32)
    else:
        incoming_proj_map=tifffile.memmap(image_path, dtype=np.uint16)
        incoming_gain_map=tifffile.memmap(gain_path, dtype=np.uint16)
    
    #you can pull too many projections that you cannot complete using the chunk size if you oversample in theta. keep it to 360d worth
    
    try:
        if tiff_intermediate_for_sino_gen:
            if going_up:
                if already_gain_corrected:
                    sino_tiff_dump_mapped[idx,:,:]=incoming_proj_map[interpolated_ystart_pos-sino_tiff_dump_mapped.shape[1]:interpolated_ystart_pos,:]
                else:
                    sino_tiff_dump_mapped[idx,:,:]=np.divide(incoming_proj_map[interpolated_ystart_pos-sino_tiff_dump_mapped.shape[1]:interpolated_ystart_pos,:],incoming_gain_map[interpolated_ystart_pos-sino_tiff_dump_mapped.shape[1]:interpolated_ystart_pos,:]).astype(np.float32)
            else:
                if already_gain_corrected:
                    sino_tiff_dump_mapped[idx,:,:]=incoming_proj_map[interpolated_ystart_pos:interpolated_ystart_pos+sino_tiff_dump_mapped.shape[1],:]
                else:
                    sino_tiff_dump_mapped[idx,:,:]=np.divide(incoming_proj_map[interpolated_ystart_pos:interpolated_ystart_pos+sino_tiff_dump_mapped.shape[1],:],incoming_gain_map[interpolated_ystart_pos:interpolated_ystart_pos+sino_tiff_dump_mapped.shape[1],:]).astype(np.float32)
        else:
            if already_gain_corrected:
                npy_map_or_filename_tif[:]=sino_tiff_dump_mapped[idx,:,:]=incoming_proj_map[interpolated_ystart_pos:interpolated_ystart_pos+sino_tiff_dump_mapped.shape[1],:]
            else:
                npy_map_or_filename_tif[:]=np.divide(incoming_proj_map[interpolated_ystart_pos:interpolated_ystart_pos+npy_map_or_filename_tif.shape[0],:],incoming_gain_map[interpolated_ystart_pos:interpolated_ystart_pos+npy_map_or_filename_tif.shape[0],:]).astype(np.float32)
            print(npy_map_or_filename_tif[0])
            npy_map_or_filename_tif.flush()
                
    except:
        print('idx %d'%idx)
        print(incoming_proj_map.shape)
        
        
# def parallel_memmap_SR(inlist):
    # list_index,filename_tif,skew_angle,vertical_velocity,proj_start,s2,s1,d,filenames_chunk,filenames_chunk_idx=inlist
    
    # paths=filenames_chunk[list_index]
    # paths_idx=filenames_chunk_idx[list_index]
    
    # sino_tiff_dump_mapped=tifffile.memmap(filename_tif, dtype=np.float32)
    # for idx_,image_path in enumerate(paths):
        # idx=paths_idx[idx_]
        # image=sino_tiff_dump_mapped[idx,:,:]
        # image = cv2.bilateralFilter(image,d,s1,s2)
        # sino_tiff_dump_mapped[idx,:,:]=image
def parallel_memmap_stripes_filter(inlist):
    list_index,filename_tif,skew_angle,vertical_velocity,proj_start,s2,s1,d,filenames_chunk,filenames_chunk_idx=inlist
    paths=filenames_chunk[list_index]
    paths_idx=filenames_chunk_idx[list_index]

    sino_tiff_dump_mapped=tifffile.memmap(filename_tif, dtype=np.float32)
    
    for idx_,image_path in enumerate(paths):
        idx=paths_idx[idx_]
        # sino_tiff_dump_mapped[:,idx,:]=remove_stripe_based_interpolation(sino_tiff_dump_mapped[:,idx,:])
        # sino_tiff_dump_mapped[:,idx,:]=remove_large_stripe(np.rot90(remove_stripe_based_sorting(np.rot90(remove_stripe_based_sorting(remove_stripe_based_interpolation(sino_tiff_dump_mapped[:,idx,:])),k=1,axes=(0, 1)),size=9),k=1,axes=(1,0)))

        sino_tiff_dump_mapped[:,idx,:]=remove_stripe_based_sorting(sino_tiff_dump_mapped[:,idx,:],size=31)

def parallel_memmap_proj_filter(inlist):
    list_index,filename_tif,skew_angle,vertical_velocity,proj_start,s2,s1,d,filenames_chunk,filenames_chunk_idx=inlist
    
    paths=filenames_chunk[list_index]
    paths_idx=filenames_chunk_idx[list_index]
    
    sino_tiff_dump_mapped=tifffile.memmap(filename_tif, dtype=np.float32)
    for idx_,image_path in enumerate(paths):
        idx=paths_idx[idx_]
        image=sino_tiff_dump_mapped[idx,:,:]
        image = cv2.bilateralFilter(image,d,s1,s2)
        sino_tiff_dump_mapped[idx,:,:]=image
    
def parallel_memmap_center_rotate(inlist):
    list_index,filename_tif,skew_angle,vertical_velocity,proj_start,y_dim,y_,center_of_rotation,filenames_chunk,filenames_chunk_idx=inlist
        
    paths=filenames_chunk[list_index]
    paths_idx=filenames_chunk_idx[list_index]
        
        
    sino_tiff_dump_mapped=tifffile.memmap(filename_tif, dtype=np.float32)
    for idx_,image_path in enumerate(paths):
        idx=paths_idx[idx_]
        image=sino_tiff_dump_mapped[idx,:,:]
        h,w = image.shape[:2]
        if not vertical_velocity:
            cX,cY = (center_of_rotation,-proj_start)
        else:
            # y_=5300 #sorta the bottom of the actual recon space
            pitch=1.04 #sorta the bottom of the actual recon space
            #  cX,cY = (center_of_rotation,9600*1.04-proj_start*1.04)
            #  cX,cY = (center_of_rotation,y_*1.04-(proj_start+h/2)*1.04)
            #   cX,cY = (center_of_rotation,y_*1.04-(proj_start-idx)*1.04)
            cX,cY = (center_of_rotation,y_*pitch-(proj_start-idx)*pitch)
            #  cX,cY = (center_of_rotation,y_*pitch-(proj_start)*pitch)
            #   cX,cY = (center_of_rotation,y_dim-proj_start+h/2)

            #   cX,cY = (center_of_rotation,(proj_start+h/2)*1.04-y_*1.04)
            #   cX,cY = (center_of_rotation,5000-proj_start*1.04)
        M = cv2.getRotationMatrix2D((cX,cY),skew_angle,1)
        image = cv2.warpAffine(image,M , (w,h))     #,flags = cv2.INTER_CUBIC) #INTER_NEAREST , INTER_LINEAR , and INTER_CUBIC
        sino_tiff_dump_mapped[idx,:,:]=image
       

def parallel_memmap_center_rotate_round2(inlist):
    idx,filename_tif,skew_angle,vertical_velocity,proj_start,y_dim,y_,center_of_rotation=inlist
        
    sino_tiff_dump_mapped=tifffile.memmap(filename_tif, dtype=np.float32)
    
    image=sino_tiff_dump_mapped[idx,:,:]
    h,w = image.shape[:2]
    if not vertical_velocity:
        cX,cY = (center_of_rotation,y_dim-proj_start+h/2)
    else:
        skew_angle=.23
        # y_=5300 #sorta the bottom of the actual recon space
        pitch=1.04 #sorta the bottom of the actual recon space
        #  cX,cY = (center_of_rotation,9600*1.04-proj_start*1.04)
        #  cX,cY = (center_of_rotation,y_*1.04-(proj_start+h/2)*1.04)
        #   cX,cY = (center_of_rotation,y_*1.04-(proj_start-idx)*1.04)
        #   cX,cY = (center_of_rotation,y_-(proj_start+h/2))
        cX,cY = (center_of_rotation,y_*pitch-(proj_start)*pitch)
        #  cX,cY = (center_of_rotation,y_*pitch-(proj_start)*pitch)
        #   cX,cY = (center_of_rotation,y_dim-proj_start+h/2)

        #   cX,cY = (center_of_rotation,(proj_start+h/2)*1.04-y_*1.04)
        #   cX,cY = (center_of_rotation,5000-proj_start*1.04)
    M = cv2.getRotationMatrix2D((cX,cY),skew_angle,1)
    image = cv2.warpAffine(image,M , (w,h))
    sino_tiff_dump_mapped[idx,:,:]=image


start_full_script=datetime.now()
proj_dir=os.path.dirname(sys.argv[0])
target_drive=os.path.join(proj_dir)


'''INPUT DIR SUFFIX'''
folder_you_want_to_reconstruct='projections'
#IGNORE SCAN LOG BY PRETENDING IT DOESN'T EXIST
log_txt_path=os.path.join(target_drive,'scanlog_.txt')

'''gains should live in a 'gains' folder and be named 'gain' and 'gain_post', they can live with other gains too.'''
# gain_path=os.path.join(target_drive,'gains'+project_drive_suffixfix,'gain.tif')
# gain_path_post=os.path.join(target_drive,'gains'+project_drive_suffixfix,'gain_post.tif')

'''B5_ Hard code the Gain directories - you can use windows copy dir url here'''
gain_folder_name='gains'
gain_path=os.path.join(proj_dir,gain_folder_name,'gain.tif')
gain_path_post=os.path.join(proj_dir,gain_folder_name,'gain_post.tif')

dark_path=os.path.join(proj_dir,gain_folder_name,'dark.tif')
dark_path_post=os.path.join(proj_dir,gain_folder_name,'dark_post.tif')
exp=360000





try:
    gain_test=tifffile.memmap(gain_path_post)
except:
    try:
        #B6 setup
        gain_dir=os.path.join(proj_dir,gain_folder_name)
        Path(gain_dir).mkdir(parents=True, exist_ok=True)
        print(f'no gains found! greating pre and post gains, make sure you have the appropriate data along side the scan directory.')
        # surrounding_folders = os.listdir(Path(proj_dir).parent.absolute())
        # current_folder=os.path.normpath(proj_dir).split(os.sep)[-1]
        
        
        # current_folder_index=surrounding_folders.index(current_folder)
        gf=os.path.join(proj_dir,f'pregains_{exp}us')
        pgf=os.path.join(proj_dir,f'postgains_{exp}us')
        df=os.path.join(proj_dir,f'predarks_{exp}us')
        pdf=os.path.join(proj_dir,f'postdarks_{exp}us')
        print(f'gain folder {gf}')
        print(f'post gain folder {pgf}')
        print(f'dark folder {df}')
        print(f'post dark folder {pdf}')
        correction_folder_list=[gf,pgf,df,pdf]
        correction_name=['gain','gain_post','dark','dark_post']
        
        correction_image_list={}
        for idx,target_correction in enumerate(correction_folder_list):
            images=get_image_paths_gains(os.path.join(Path(proj_dir).parent.absolute(),target_correction))
            print(f'processing {correction_name[idx]}')
            if idx<2:
                image_stack=tifffile.imread(images)
                image_stack=np.median(image_stack,axis=0)
                correction_image_list[correction_name[idx]]=image_stack.copy()
                tifffile.imwrite(os.path.join(gain_dir,f'{correction_name[idx]}.tif'),np.squeeze(image_stack.astype(np.float32)))
            else:
                image_stack=tifffile.imread(images[1:])
                image_stack=np.median(image_stack,axis=0)
                correction_image_list[correction_name[idx]]=image_stack.copy()
                tifffile.imwrite(os.path.join(gain_dir,f'{correction_name[idx]}.tif'),np.squeeze(image_stack.astype(np.float32)))
        preview_dir=os.path.join(proj_dir,'gc_previews')
        Path(preview_dir).mkdir(parents=True, exist_ok=True)
        proections_dir=os.path.join(proj_dir,f'{folder_you_want_to_reconstruct}_{exp}us')
        
        
        print(f'projections coming from this folder: {proections_dir}')
        images_nshit=get_image_paths(proections_dir)
        
        projection_0=tifffile.imread(images_nshit[0]) #the first one and shit
        projection_0=np.subtract(projection_0,correction_image_list['dark'])
        projection_0=np.divide(projection_0,correction_image_list['gain'],dtype=np.float32)
        tifffile.imwrite(os.path.join(preview_dir,'proj_0_gain_corrected.tif'),np.squeeze(projection_0.astype(np.float32)))
        projection_last=tifffile.imread(images_nshit[-1]) #the first one and shit
        projection_last=np.subtract(projection_last,correction_image_list['dark_post'])
        projection_last=np.divide(projection_last,correction_image_list['gain_post'],dtype=np.float32)
        tifffile.imwrite(os.path.join(preview_dir,'proj_last_gain_corrected.tif'),np.squeeze(projection_last.astype(np.float32)))
        
        projection_halfish=tifffile.imread(images_nshit[len(images_nshit)//2]) #the first one and shit
        projection_halfish=np.subtract(projection_halfish,correction_image_list['dark'])
        projection_halfish=np.divide(projection_halfish,correction_image_list['gain'],dtype=np.float32)
        tifffile.imwrite(os.path.join(preview_dir,'proj_halfish_gain_corrected.tif'),np.squeeze(projection_halfish.astype(np.float32)))
        
        projection_last=tifffile.imread(images_nshit[-1]) #the first one and shit
        projection_last=np.subtract(projection_last,correction_image_list['dark'])
        projection_last=np.divide(projection_last,correction_image_list['gain'],dtype=np.float32)
        tifffile.imwrite(os.path.join(preview_dir,'proj_last_FIRST_gain_corrected.tif'),np.squeeze(projection_last.astype(np.float32)))
        berk_mode=True
    #A3 SETUP
    except:
        try:
            gain_dir=os.path.join(proj_dir,gain_folder_name)
            Path(gain_dir).mkdir(parents=True, exist_ok=True)
            print(f'no gains found! greating pre and post gains, make sure you have the appropriate data along side the scan directory.')
            surrounding_folders = os.listdir(Path(proj_dir).parent.absolute())
            surrounding_folders.sort()
            current_folder=os.path.normpath(proj_dir).split(os.sep)[-1]
            print(f'surrounding folders {surrounding_folders}')
            print(f'target gain folder {current_folder}')
            current_folder_index=surrounding_folders.index(current_folder)
            gf=surrounding_folders[current_folder_index-1]
            pgf=surrounding_folders[current_folder_index+1]
            df=surrounding_folders[current_folder_index-2]
            pdf=surrounding_folders[current_folder_index+2]
            print(f'gain folder {gf}')
            print(f'post gain folder {pgf}')
            print(f'dark folder {df}')
            print(f'post dark folder {pdf}')
            correction_folder_list=[gf,pgf,df,pdf]
            correction_name=['gain','gain_post','dark','dark_post']
            correction_image_list={}
            for idx,target_correction in enumerate(correction_folder_list):
                # print(f'processing correction_name[idx]
                images=get_image_paths_gains(os.path.join(Path(proj_dir).parent.absolute(),target_correction))
                if idx<2:
                    image_stack=tifffile.imread(images)
                    image_stack=np.median(image_stack,axis=0)
                    tifffile.imwrite(os.path.join(gain_dir,f'{correction_name[idx]}.tif'),image_stack.astype(np.float32))
                else:
                    image_stack=tifffile.imread(images[1:])
                    image_stack=np.median(image_stack,axis=0)
                    tifffile.imwrite(os.path.join(gain_dir,f'{correction_name[idx]}.tif'),image_stack.astype(np.float32))
                correction_image_list[correction_name[idx]]=image_stack
            
            preview_dir=os.path.join(proj_dir,'gc_previews')
            Path(preview_dir).mkdir(parents=True, exist_ok=True)
            
            images_nshit=get_image_paths(os.path.join(proj_dir,folder_you_want_to_reconstruct))
            projection_0=tifffile.imread(images_nshit[0]) #the first one and shit
            projection_0=np.subtract(projection_0,correction_image_list['dark'])
            projection_0=np.divide(projection_0,correction_image_list['gain'],dtype=np.float32)
            tifffile.imwrite(os.path.join(preview_dir,'proj_0_gain_corrected.tif'),np.squeeze(projection_0.astype(np.float32)))
            projection_last=tifffile.imread(images_nshit[-1]) #the first one and shit
            projection_last=np.subtract(projection_last,correction_image_list['dark_post'])
            projection_last=np.divide(projection_last,correction_image_list['gain_post'],dtype=np.float32)
            tifffile.imwrite(os.path.join(preview_dir,'proj_last_gain_corrected.tif'),np.squeeze(projection_last.astype(np.float32)))
            
            projection_halfish=tifffile.imread(images_nshit[len(images_nshit)//2]) #the first one and shit
            projection_halfish=np.subtract(projection_halfish,correction_image_list['dark'])
            projection_halfish=np.divide(projection_halfish,correction_image_list['gain'],dtype=np.float32)
            tifffile.imwrite(os.path.join(preview_dir,'proj_halfish_gain_corrected.tif'),np.squeeze(projection_halfish.astype(np.float32)))
            
            projection_last=tifffile.imread(images_nshit[-1]) #the first one and shit
            projection_last=np.subtract(projection_last,correction_image_list['dark'])
            projection_last=np.divide(projection_last,correction_image_list['gain'],dtype=np.float32)
            tifffile.imwrite(os.path.join(preview_dir,'proj_last_FIRST_gain_corrected.tif'),np.squeeze(projection_last.astype(np.float32)))
        except:
            print('gain structure broken, critical error!')
try:
    current_node=os.environ['SLURM_JOB_NODELIST']
    print(f'hpc run: {current_node}')
    is_hpc=True

except:
    # print('not running on SLURM!')
    is_hpc=False
if __name__ == '__main__':
    try:
        Path.unlink(Path('/dev/shm/allinwonderful'))
    except:
        print('shared memory already clean!')
    delete_all_psm_here('/dev/shm/')
        
    '''this changes per computer'''
    threads_to_use_for_sino_gen=24
    #npy intermediate is apparently not working as of 6/4/2022. cannot read back in non-zero values. look into passing the mapped arrays.
    shared_mem_read=False
    tiff_intermediate_for_sino_gen=not shared_mem_read
    list_mode_read=True
    combine_processes_with_sinogen=True
    save_temp_file=True
    
    '''setup directories'''
    dir_=os.path.join(proj_dir,folder_you_want_to_reconstruct)
    '''B5_ hard link the projection directory'''
    '''B5_ hard link the projection directory'''
    berk_mode=False
    if berk_mode:
        dir_=os.path.join(proj_dir,f'{folder_you_want_to_reconstruct}_{exp}us')
    all_of_the_stripe_removal=berk_mode
    
    
    # dir_=Path(r"./projections")
    
    print(os.path.normpath(dir_).split(os.sep))
    start_full=datetime.now()
    unique_id=os.path.normpath(dir_).split(os.sep)[-2]
    sample_name=os.path.normpath(dir_).split(os.sep)[-3]
    project_name=os.path.normpath(dir_).split(os.sep)[-4]
    print(sample_name)
    # output_target=os.path.join('O:','B5 Transfer','DV_Recons','phantom')
    out_dir=os.path.join(proj_dir,'reconstructed',sample_name)


    
    # out_dir_preview=os.path.join(proj_dir,'reconstructed','previews')
    # Path(out_dir).mkdir(parents=True, exist_ok=True)
    # Path(out_dir_preview).mkdir(parents=True, exist_ok=True)
    print(f'projections reading from {dir_}')
    #grab paths of all of the images in the target folder
    image_paths=get_image_paths(dir_)
    
    how_many_angles_to_recon=len(image_paths)
    how_many_angles_to_recon=len(image_paths)//2
    we_are_subsetting_angles=False
    if len(image_paths)!=how_many_angles_to_recon:
        we_are_subsetting_angles=True
        #0-180
        indexes_0_180 = np.round(np.linspace(0, len(image_paths)//2 - 1, how_many_angles_to_recon)).astype(int)
        #180-360
        indexes_180_360 = np.round(np.linspace(len(image_paths)//2,len(image_paths)-1,  how_many_angles_to_recon)).astype(int)

        image_paths=[image_paths[x] for x in indexes_0_180]
        print(len(image_paths))
        
    # multi_proc_pass_list=np.array_split(image_paths,threads_to_use_for_sino_gen)
    # multi_proc_pass_list_idx=np.array_split(range(0,len(image_paths)),threads_to_use_for_sino_gen)
    # print(image_paths)
    ''''B5_ this assumes no frame loss but you're stuck like this so whatever'''
    # proj_per_section=7873 #1969
    
    #STEP TWO, IGNORE HARD CODING PROJECTIONS AND ASSUME EVERYTHING IS FINE
    proj_per_section=len(image_paths) #1969
    print(f'projections per section {proj_per_section}')
    '''memory map the first projection and find its shape, readable style'''
    first_mmap_image=tifffile.memmap(image_paths[0])
    print(first_mmap_image.shape)
    
    if len(first_mmap_image.shape)>2:
        poop,y_dim,x_dim=first_mmap_image.shape
    else:
        y_dim,x_dim=first_mmap_image.shape
    
    #STEP THREE, IGNORE THIS BECAUSE WE ARE IGNORING LOGS
    trigger_timing_us=194900
    going_up=True
    degrees_per_section=180-(180/proj_per_section)*8
    degrees_per_section=360
    if we_are_subsetting_angles:
        degrees_per_section=180 #for argonne halvening
    if berk_mode:
        degrees_per_section=180
        # degrees_per_section=180-(180/proj_per_section)*8
    reconstruction_angle_offset=-(61.37+30)
    reconstruction_angle_offset=-6
    
    phase_ret=False
    proj_filter=True
    d,s1,s2=[3,1,1]
    # d,s1,s2=[25,25,25]

    
    #STEP 4 THIS DOES STUFF BUT WHATEVER
    sinogram_chunk_size=400 #pretty safe for shitty rotation data
    
    
    
    '''
    if section recon, proj_start is the y position (0 is top of ) of sinogram to recon according to projection y dimension.
    if helical, idx of what projection to start sinogram generation on
    '''
    #(might be a negative skew)
    #STEP 5 THIS IS IMPORTANT WHAT SLICE DO YOU WANT TO RECON FROM

    proj_start=0 #1225 center  @ skew_angle_mtx=[-0.0566]              #.107, 2899 center at 2902+50 still appears to be
    # proj_start=3000 #1225 center  @ skew_angle_mtx=[-0.0566]              #.107, 2899 center at 2902+50 still appears to be
    # proj_start=3488 #1223 center  @ skew_angle_mtx=[-0.0566]  1222*-.06             #.107, 2899 center at 2902+50 still appears to be
    # proj_start=3488 #1223 center  @ skew_angle_mtx=[-0.0525]  1222*-.06             #.107, 2899 center at 2902+50 still appears to be
    # proj_start=3488 #1227.5 center  @ 0             #.107, 2899 center at 2902+50 still appears to be
    # proj_start=1200 #1226.5 center  @ 0             #.107, 2899 center at 2902+50 still appears to be
    # proj_start=0 #6489 center               #.107, 2899 center at 2902+50 still appears to be
    # proj_start=3920 #6489 center               #.107, 2899 center at 2902+50 still appears to be
    # proj_start=2500 #4608 center               #.107, 2899 center at 2902+50 still appears to be
    # proj_start=3050 #2165@0 2167@0.0566     center               #.107, 2899 center at 2902+50 still appears to be
    # proj_start=300 #.107, 2899 center at 2902+50 still appears to be
    # proj_start=0 #.107, 2899 center at 2902+50 still appears to be
    # proj_start=2248 #.107, 2899 center at 2902+50 still appears to be
    # proj_start=2500 #.107, 2899 center at 2902+50 still appears to be
    # proj_start=5190 #center at 2902+50 still appears to be 

    # skew corrected centers: 5256.75 each*
    y_=y_dim #end of onion tip #this is when to end recon
    # y_=1500+1150 #end of onion tip #this is when to end recon
    # y_=proj_start+400 #this is when to end recon
    # proj_start=328 #center at 2615 -.23d rotation: 5100
    # omatidia height 



    '''for rotation testing when you have only rotated or bined a subset of a full scan not starting at 0.
    a value of 11000 would mean that the projections you are reading in start proj_11000. automated. be careful.'''
    #grabs the projection number from the first projection in the target folder in case there is an offset as described above
    # stack_subset_offset=int(''.join(filter(str.isdigit, os.path.basename(image_paths[0]))))
    # if stack_subset_offset:
        # print('\n\n\nsubstack detected, starting with projection number %d\n\n' %stack_subset_offset)

    '''semi-auto center finding params'''
    #STEP 6 THIS IS IMPORTANT.
    find_center=False
    #THIS OPERATES ON 180 SO I GOTTA CHANGE SOME SHIT DOWN STAIRS
    find_center_pc=False #THIS WILL OVERRIDE LINE 453
    find_center_vo=False
    '''B5_ '''
    center_find_range=1
    center_find_step=1
    
    
    full_recon=True
    final_recon_crop=np.s_[:,:]
    # final_recon_crop=np.s_[450:3000,650:3000]
    
    chunk_overlap=0
    if sinogram_chunk_size<=2*chunk_overlap:
        chunk_overlap=0
    
    '''B5_ center'''
    #
    center_of_rotation_px_matx=[x_dim/2]
    center_of_rotation_px_matx=[6487] # @2700 Mememoma
    
    print(f'pixel center of image: {center_of_rotation_px_matx}')

    # center after first pass of skew correction = 5256.75
    
    # center_of_rotation_px_matx=[5485] # Second pass of skew correction - taking 5257 as center guess # @200 Mememoma
    
    # center_of_rotation_px_matx=[773] #unrotated and -.23
    # center_of_rotation_px_matx=[3095] #.23 rotated
    
    center_interp_function = interpolate.interp1d([2000,4700], [2632.5,2621.5], fill_value='extrapolate')
    center_interp_function = False
    # center_of_rotation_px_matx=[2590] #.23 rotated
    
    

    
    
    '''if you rotate your stack you get black edges. have to trim. probably safe to just leave this on'''
    edge_buff=True
    edge_buffers_1=50
    edge_buffers_2=-50
    ''' if you rotate your stack you get black beginning and ends as well. theta clip those out (remove beginning and ends)'''
    theta_clip=False
    theta_buffer_pre=50
    theta_buffer_post=500
    
    '''skew correction'''
    skew_angle_mtx=np.arange(-.2,-.26,-.01).round(3)
    skew_angle_mtx=[.01,.015,.02]
    # skew_angle_mtx=[.1015]
    skew_angle_mtx=[-.035] #B5 tilt
    skew_angle_mtx=[-0.0566] #that's the FIRST B6 tilt (or very close to). second b6 tilt is like. .115 :,(
    skew_angle_mtx=[-0.065] #that's the FIRST B6 tilt (or very close to). second b6 tilt is like. .115 :,(
    skew_angle_mtx=[-0.025] #that's the FIRST B6 tilt (or very close to). second b6 tilt is like. .115 :,(

    skew_angle_mtx=[0]
    
    
    # y_=22200-11000
    print(skew_angle_mtx)
    
    # adjust center according to edge buffer



    '''figure out what frames are lost based on buffer.timestamp_ns logging'''
    
    if os.path.isfile(log_txt_path): 
        skipped_frame_list,sino_frame_skip_order,real_projection_order,skipped_frame_index,skipped_frame_index_value=find_skipped_frames([log_txt_path,trigger_timing_us,proj_per_section])

        print(skipped_frame_index)
        theta_step_degrees=degrees_per_section/proj_per_section
        print(theta_step_degrees)
        
        theta_step_rad=np.deg2rad(theta_step_degrees)    
        real_theta_step=np.array(real_projection_order)
        print(real_projection_order)
        
        real_theta_matrix_logbased=real_theta_step*theta_step_rad
        real_theta_matrix_encoder_based=[]
        running_total=0
        # for idx in range(0,proj_per_section-len(skipped_frame_index)):
            # # variable_= idx in skipped_frame_index
            # running_total+=(theta_step_rad*skipped_frame_index_value[idx])
            # if skipped_frame_index_value[idx]>1:
                # print(idx)
                # print(skipped_frame_index_value[idx])
            # # print(running_total)
            # real_theta_matrix_encoder_based.append(running_total)
        # sino_frame_skip_order=sino_frame_skip_order[stack_subset_offset+proj_start:stack_subset_offset+proj_start+proj_per_section]
        
        '''set up theta. this will be more complicated when we deal with dropped frames. we are dealing with dropped frames.'''
        # theta = np.linspace(0,np.deg2rad(degrees_per_section),proj_per_section,False)
        # theta = real_theta_matrix_encoder_based[stack_subset_offset+proj_start:stack_subset_offset+proj_start+proj_per_section]
        theta = real_theta_matrix_encoder_based
        proj_per_section_=proj_per_section-len(skipped_frame_index)
        print(f'actual projections per section {proj_per_section}')
        if len(skipped_frame_index)<1:
            print('no frames skipped! using linspace')
            # theta = np.linspace(0,np.deg2rad(degrees_per_section),proj_per_section,False)
            theta = np.linspace(0+np.deg2rad(reconstruction_angle_offset),np.deg2rad(degrees_per_section+reconstruction_angle_offset),proj_per_section,True)
        else:
            theta = np.linspace(0+np.deg2rad(reconstruction_angle_offset),np.deg2rad(degrees_per_section+reconstruction_angle_offset),proj_per_section,True)
            
            print(theta.shape)

            # print(skipped_frame_index_value)
            print(skipped_frame_index)
            for frame_idx, frame in enumerate(skipped_frame_index):
                for skip_idx,skips in enumerate(range(0,skipped_frame_index_value[frame]-1)):
                    # print(skips)
                    # print(frame-frame_idx+skip_idx)
                    # asd
                    theta=np.delete(theta, frame-frame_idx+skip_idx-1)
                    proj_per_section-=1
            
    
    else:
        ###BERKELEY ^ ANGLE INHERITANCE:
        berk_mode=False
        if berk_mode:
            log_txt_path=os.path.join(target_drive,'scanlog.txt')
            logging_dictionary={}
            with open(log_txt_path, 'r') as f:
                next(f)
                lines = f.read().splitlines()
                for idx,line in enumerate(lines):
                    if len(line)==0:
                        continue
                    if idx == 0:
                        line_split=line.split('\t')
                        # initial_value=int(line_split[-3])
                        initial_value=int(line_split[-3])
                        # print(initial_value)
                        
                    line_split=line.split('\t')
                    # print(int(line_split[0].split('_')[-1]))
                    logging_dictionary[int(line_split[0].split('_')[-1])]=int(line_split[-3])-initial_value
                last_value=int(line_split[-3])-initial_value
            print(f'first stamp: {initial_value}')
            print(f'laststamp: {last_value}')
            print(f'random percentage: {logging_dictionary[443]/last_value}')
            theta_list=[]
            for angle_idx in range(0,proj_per_section):
                theta_list.append(np.deg2rad(logging_dictionary[angle_idx]/last_value*degrees_per_section)+np.deg2rad(reconstruction_angle_offset))
            theta=np.asarray(theta_list)
            # print(theta)
            # asd
            # berk_mode=True
    
        else:
            theta = np.linspace(0+np.deg2rad(reconstruction_angle_offset),np.deg2rad(degrees_per_section+reconstruction_angle_offset),proj_per_section,False)

        sino_frame_skip_order=np.zeros((proj_per_section))
        
        if we_are_subsetting_angles:
            theta=theta[indexes_0_180]

    print(theta.shape)
    print(proj_per_section)
    
    '''
    vertical_velocitiy how many pixels the sample tavels per section in the y dimension. 
    it's an array because i tend to iterate through this in steps of 10 or 50 or 100 to get the correct value.
    sectional scans are zero
    '''
    # vertical_velocities=np.arange(696,697,1) 
    
    vertical_velocities=[0]
    # vertical_velocities=np.arange(2900,3300,100)
    count=0
    for idx,proj_target_range in enumerate(range(proj_start,y_,(sinogram_chunk_size-2*chunk_overlap))):
        loop_timing=datetime.now()
    # for idx,proj_target_range in enumerate(range(proj_start,y_,sinogram_chunk_size-chunk_overlap)):
        proj_target_range=range(proj_target_range,proj_target_range+sinogram_chunk_size)
        if list(proj_target_range)[0]+sinogram_chunk_size>y_:
            proj_target_range=range(list(proj_target_range)[0],y_)
            sinogram_chunk_size=len(list(proj_target_range))
        
        
        print(f'processing chunk {proj_target_range}')
        already_gain_corrected=False
        if first_mmap_image.dtype=='float32':
            print('data is already gain corrected (float32)')
            already_gain_corrected=True
        for skew_angle in skew_angle_mtx:
            for vertical_velocity in vertical_velocities:
                # if len(skew_angle_mtx)>1:
                    # print('resetting angle space')
                    # theta = np.linspace(0,np.deg2rad(degrees_per_section),proj_per_section,False)
                '''
                figuring out how many degrees the sample actually spins per fov. This value should the same as degrees_per_section if sectional.
                '''
                Vertvelocity_VertDim_ratio=vertical_velocity/y_dim
                if not vertical_velocity : Vertvelocity_VertDim_ratio=1 
                actuaual_degrees_per_section=degrees_per_section/Vertvelocity_VertDim_ratio

                
                projections_per_fov=int(round(proj_per_section/Vertvelocity_VertDim_ratio))
                print(f'\ncurrent y slice: {list(proj_target_range)[0]}')
                print('\ndegrees per section: %f' %degrees_per_section)
                print('actual degrees per fov: %f' %actuaual_degrees_per_section)
                print('\nprojections per section: %d' %proj_per_section)
                print('projections per fov: %d' %projections_per_fov)
                print('\nydim projections: %d' %y_dim)
                print('xdim projections: %d' %x_dim)
                print('\nypixels traveled per section: %d' %vertical_velocity)
                print('\npitch: %f' %(y_dim/projections_per_fov))
               
                '''now we start to set up the creation of the helical sinogram'''
                if going_up:
                    slope_function = interpolate.interp1d([0,projections_per_fov],[y_dim+1,0], kind='linear')
                else:
                    slope_function = interpolate.interp1d([0,projections_per_fov],[0,y_dim+1], kind='linear')
                
                if shared_mem_read:
                    try:
                        shm_name='allinwonderful'
                        existing_shm = shared_memory.SharedMemory(name=shm_name)
                        print(existing_shm)
                        existing_shm.close()
                        existing_shm.unlink()
                        print(f'deleted shared memory: {shm_name}')
                    except:
                        print(f'shared memory: {shm_name} not found to delete')
                    print(f'Time for shared memory read mode 2!')
                    sino_tiff_dump=np.empty(shape=(proj_per_section,sinogram_chunk_size,x_dim), dtype=np.float32)
                    print(f'array shape: {sino_tiff_dump.shape}')
                    print(f'array size: {sino_tiff_dump.size}')
                    print(f'array itemsize: {sino_tiff_dump.itemsize}')
                    print(f'array in bytes: {sino_tiff_dump.nbytes}')     
                    print(f'array bytes calculation: {proj_per_section*sinogram_chunk_size*x_dim*4}')     
                    sino_tiff_dump_bytes=sino_tiff_dump.nbytes
                    del sino_tiff_dump     
                    shm = shared_memory.SharedMemory(create=True, size=proj_per_section*sinogram_chunk_size*x_dim*4,name='allinwonderful')
                    print(f'shared memory array buffer created! at {shm}')
                    projection_matrix = np.ndarray(shape=(proj_per_section,sinogram_chunk_size,x_dim), dtype=np.float32, buffer=shm.buf)
                    print(f'ndarray with shm buffer created!')

                    projection_matrix[:,:,:] = np.zeros(shape=(proj_per_section,sinogram_chunk_size,x_dim), dtype=np.float32)
                    print(f'ndarray with shm buffer initialized with zeros! JK SKIPPED IT Starting Pool')
                    multi_proc_pass_list=np.array_split(image_paths,threads_to_use_for_sino_gen)
                    multi_proc_pass_list_idx=np.array_split(range(0,len(image_paths)),threads_to_use_for_sino_gen)
                    pass_list_shm=[]
                    for idx in range(0,threads_to_use_for_sino_gen):
                        pass_list_shm.append([idx,shm.name,projection_matrix.shape,multi_proc_pass_list,multi_proc_pass_list_idx,proj_target_range[0],gain_path,gain_path_post,dark_path,dark_path_post])
                    with Pool(threads_to_use_for_sino_gen) as p:
                        p.map(shared_mem_reader2,pass_list_shm)
                    print(f'done pool, array shape is {projection_matrix.shape}')
                    shm.close()
                    shm.unlink()    
                else:
                    if tiff_intermediate_for_sino_gen:
                        '''set up temporary intermediate file to write and read from'''
                        if is_hpc:
                            tmp_dir=os.environ['TMPDIR']
                            print(f'using temp directory {tmp_dir}')
                            tmp_dir=os.path.join(tmp_dir,'VANSELOW_BABYYY')
                            Path(tmp_dir).mkdir(parents=True, exist_ok=True)
                            filename_tif=os.path.join(tmp_dir,'temp.tif')
                        else:
                            filename_tif=os.path.join(proj_dir,'temp.tif')
                            # print('filtering projections')
                        
                        start_memmapCreation=datetime.now()
                        
                        try:
                            sino_tiff_dump=tifffile.memmap(filename_tif)
                            if not sino_tiff_dump.shape==(proj_per_section,sinogram_chunk_size,x_dim):
                                del sino_tiff_dump
                                sino_tiff_dump=tifffile.memmap(filename_tif,shape=(proj_per_section,sinogram_chunk_size,x_dim),dtype=np.float32)
                        except:
                            sino_tiff_dump=tifffile.memmap(filename_tif,shape=(proj_per_section,sinogram_chunk_size,x_dim),dtype=np.float32)
                        print(f'memory mapped tiff shape is {sino_tiff_dump.shape}')
                        print(f'\nmemory map creation timing:{(datetime.now()-start_memmapCreation)}')
                        del sino_tiff_dump

                        if not vertical_velocity:
                            if list_mode_read:
                                filenames_chunk=np.array_split(image_paths,threads_to_use_for_sino_gen)
                                filenames_chunk_idx=np.array_split(range(0,len(image_paths)),threads_to_use_for_sino_gen)
                                multi_proc_pass_list = [[idx,image_paths[target_proj_idx], filename_tif, slope_function,tiff_intermediate_for_sino_gen,already_gain_corrected,sino_frame_skip_order,going_up,proj_target_range[0],gain_path,gain_path_post,dark_path,dark_path_post,filenames_chunk,filenames_chunk_idx] for idx,target_proj_idx in enumerate(range(0,threads_to_use_for_sino_gen))]

                            else:
                                # multi_proc_pass_list = [[idx,image_paths[target_proj_idx], filename_tif, slope_function,tiff_intermediate_for_sino_gen,already_gain_corrected,sino_frame_skip_order,going_up,proj_target_range[0],gain_path,gain_path_post,dark_path,dark_path_post] for idx,target_proj_idx in enumerate(range(0,proj_per_section))]
                                multi_proc_pass_list = [[idx,image_paths[target_proj_idx], filename_tif, slope_function,tiff_intermediate_for_sino_gen,already_gain_corrected,sino_frame_skip_order,going_up,proj_target_range[0],gain_path,gain_path_post,dark_path,dark_path_post] for idx,target_proj_idx in enumerate(range(0,proj_per_section))]
                            
                        else:
                            multi_proc_pass_list = [[idx,image_paths[target_proj_idx], filename_tif, slope_function,tiff_intermediate_for_sino_gen,already_gain_corrected,sino_frame_skip_order,going_up,proj_target_range[0]] for idx,target_proj_idx in enumerate(range(proj_target_range[0],proj_target_range[0]+proj_per_section))]
                    
                    
                    
                    
                    else:
                        '''set up temporary intermediate file to write and read from'''
                        filename_npy=os.path.join(proj_dir,'temp.npy')
                        sino_npy_dump = np.memmap(filename_npy, shape=(proj_per_section,sinogram_chunk_size,x_dim),dtype=np.float32,mode='w+')
                        
                        multi_proc_pass_list = [[idx,image_paths[target_proj_idx], sino_npy_dump[idx], slope_function,tiff_intermediate_for_sino_gen,already_gain_corrected,sino_frame_skip_order,going_up,proj_target_range[0]] for idx,target_proj_idx in enumerate(range(proj_target_range[0],proj_target_range[0]+proj_per_section))]
                    # print(multi_proc_pass_list[0])
                    
                    
                    start_sino_gen=datetime.now()

                    pool = Pool(threads_to_use_for_sino_gen)
                    if Vertvelocity_VertDim_ratio==1:
                        if list_mode_read:
                            if combine_processes_with_sinogen:
                                multi_proc_pass_list = [[idx,image_paths[target_proj_idx], filename_tif, slope_function,tiff_intermediate_for_sino_gen,already_gain_corrected,sino_frame_skip_order,going_up,proj_target_range[0],gain_path,gain_path_post,dark_path,dark_path_post,filenames_chunk,filenames_chunk_idx,skew_angle,d,s1,s2,center_of_rotation_px_matx[0]] for idx,target_proj_idx in enumerate(range(0,threads_to_use_for_sino_gen))]
                                
                                pool.map(parallel_sectional_write_plusprocessing_listdist, multi_proc_pass_list)
                            else:
                                pool.map(parallel_sectional_write_listdist, multi_proc_pass_list)
                        else:
                            pool.map(parallel_sectional_write, multi_proc_pass_list)
                    else:
                        pool.map(parallel_helical_write, multi_proc_pass_list)
                    pool.close()
                    pool.join()
                    print('sino generation timing: %s'%(datetime.now()-start_sino_gen))
                    
                    '''skew correction'''
                    if not combine_processes_with_sinogen:
                        if proj_filter:
                            print('filtering projections')
                            start_filter=datetime.now()
                            filenames_chunk=np.array_split(image_paths,threads_to_use_for_sino_gen)
                            filenames_chunk_idx=np.array_split(range(0,len(image_paths)),threads_to_use_for_sino_gen)
                            multi_proc_pass_list = [[idx,filename_tif,skew_angle,vertical_velocity,list(proj_target_range)[0],s2,s1,d,filenames_chunk,filenames_chunk_idx] for idx in range(0,threads_to_use_for_sino_gen)]
                            pool = Pool(threads_to_use_for_sino_gen)
                            pool.map(parallel_memmap_proj_filter, multi_proc_pass_list)
                            pool.close()
                            pool.join()
                            print('filter timing: %s'%(datetime.now()-start_filter))
                        if skew_angle != 0:
                            start_skew=datetime.now()
                            filenames_chunk=np.array_split(image_paths,threads_to_use_for_sino_gen)
                            filenames_chunk_idx=np.array_split(range(0,len(image_paths)),threads_to_use_for_sino_gen)
                            multi_proc_pass_list = [[idx,filename_tif,skew_angle,vertical_velocity,list(proj_target_range)[0],y_dim,y_,center_of_rotation_px_matx[0],filenames_chunk,filenames_chunk_idx] for idx in range(0,threads_to_use_for_sino_gen)]
                            pool = Pool(threads_to_use_for_sino_gen)
                            pool.map(parallel_memmap_center_rotate, multi_proc_pass_list)
                            pool.close()
                            pool.join()
                            print('skew correction timing: %s'%(datetime.now()-start_skew))

                    # '''skew correction'''
                    # if skew_angle:
                        # start_skew=datetime.now()
                        # multi_proc_pass_list = [[idx,filename_tif,skew_angle,vertical_velocity,proj_start,y_dim,y_,center_of_rotation_px_matx[0]] for idx in range(0,proj_per_section)]
                        # pool = Pool(threads_to_use_for_sino_gen)
                        # pool.map(parallel_memmap_center_rotate_round2, multi_proc_pass_list)
                        # pool.close()
                        # pool.join()
                        # print('skew correction timing: %s'%(datetime.now()-start_skew))
                '''
                no stripes to correct in this helical approach, but if we're sectional....
                https://tomopy.readthedocs.io/en/latest/api/tomopy.prep.stripe.html
                '''
                if not vertical_velocity:
                    print('stripe correcting')
                    start_stripe=datetime.now()
                    #These are typical synchrotron values for us for deringing for fw and ti methods.

                    sigma_=4
                    fw_remove=True
                    if fw_remove:
                        
                        projection_matrix=tifffile.memmap(filename_tif, dtype=np.float32)
                        # projection_matrix = tomopy.remove_stripe_fw(projection_matrix,sigma=sigma_)
                        projection_matrix = tomopy.remove_stripe_based_sorting(projection_matrix,size=31)
                        # projection_matrix=tomopy.remove_all_stripe(projection_matrix,snr=3, la_size=61, sm_size=21)
                    print(f'stripe fw timing:{(datetime.now()-start_stripe)}')
                    all_of_the_stripe_removal= not fw_remove
                    if all_of_the_stripe_removal:
                        # projection_matrix=tomopy.remove_all_stripe(projection_matrix,snr=1, la_size=51, sm_size=21)
                        # projection_matrix=tomopy.remove_all_stripe(projection_matrix,snr=3, la_size=61, sm_size=21)
                        #BEST:
                        # projection_matrix=tomopy.remove_all_stripe(projection_matrix,snr=3, la_size=61, sm_size=21)
                        print('stripe removing')
                        parallel_tiff_stripes=True
                        if parallel_tiff_stripes:
                            if (sinogram_chunk_size//2)<cpu_count():
                                threads_to_use_for_sino_gen_=sinogram_chunk_size//2
                            if threads_to_use_for_sino_gen>=sinogram_chunk_size:
                                threads_to_use_for_sino_gen_=sinogram_chunk_size
                            else:
                                # print(f'this machine has {cpu_count()} cores')
                                if sinogram_chunk_size>cpu_count():
                                    threads_to_use_for_sino_gen_=cpu_count()
                                    
                                else:
                                    threads_to_use_for_sino_gen_=sinogram_chunk_size

                            print(f'mem map stripe removal using {threads_to_use_for_sino_gen_} threads!!')
                            filenames_chunk=np.array_split(range(0,sinogram_chunk_size),threads_to_use_for_sino_gen_)
                            filenames_chunk_idx=np.array_split(range(0,sinogram_chunk_size),threads_to_use_for_sino_gen_)
                            multi_proc_pass_list = [[idx,filename_tif,skew_angle,vertical_velocity,list(proj_target_range)[0],s2,s1,d,filenames_chunk,filenames_chunk_idx] for idx in range(0,threads_to_use_for_sino_gen_)]
                            pool = Pool(threads_to_use_for_sino_gen_)
                            pool.map(parallel_memmap_stripes_filter, multi_proc_pass_list)
                            pool.close()
                            pool.join()
                            # print('filter timing: %s'%(datetime.now()-start_filter))
                            # projection_matrix=remove_stripe_based_interpolation(projection_matrix,1,21,.1)
                        
                    # projection_matrix=tomopy.remove_all_stripe(projection_matrix,snr=3, la_size=121, sm_size=41)
                    # #ti takes a long time and is not usually worth it. 
                    # alpha_=4
                    # projection_matrix = tomopy.remove_stripe_ti(projection_matrix,alpha=alpha_)
                    # projection_matrix = tomopy.remove_stripe_sf(projection_matrix,size =5)
                    print(f'stripe timing:{(datetime.now()-start_stripe)}')
                    
                    
                    
                    ''' series of steps to correct projections '''
                    if not fw_remove:
                        if tiff_intermediate_for_sino_gen:
                            projection_matrix=tifffile.memmap(filename_tif, dtype=np.float32)
                        else:
                            del sino_npy_dump
                            projection_matrix=np.memmap(filename_npy,shape=(proj_per_section,sinogram_chunk_size,x_dim),dtype=np.float32,mode='r')
                            # projection_matrix=sino_npy_dump

                if edge_buff:
                    projection_matrix=projection_matrix[:,:,edge_buffers_1:edge_buffers_2]
                    
                    # print(f'edge: {center_of_rotation_px_matx[0]}')
                if theta_clip:
                    projection_matrix=projection_matrix[theta_buffer_pre:projection_matrix.shape[0]-theta_buffer_post,:,:]
                    theta_=theta[theta_buffer_pre:theta.shape[0]-theta_buffer_post]
                #-log correction
                if phase_ret:
                    print('performing phase retrieval')
                    start_phase=datetime.now()
                    projection_matrix=tomopy.retrieve_phase(projection_matrix,pixel_size=0.0005,dist=7,energy=14,alpha=.0006,pad=True)
                    print(f'phase timing:{(datetime.now()-start_phase)}')
                # projection_matrix[projection_matrix<=0]=0.0000000001
                # projection_matrix[projection_matrix>=1]=.1  
                if not combine_processes_with_sinogen:
                    print('performing minus log retrieval')
                    start_mLog=datetime.now()
                    projection_matrix = tomopy.minus_log(projection_matrix)
                    print(f'nLog timing:{(datetime.now()-start_mLog)}')
                
                print(projection_matrix.shape)
                # tifffile.imwrite(os.path.join(proj_dir,'temp_sino.tif'),projection_matrix[:,200:201,:])
                
 
                    
                print('done stripe correcting!!!')
                if chunk_overlap:
                    print(projection_matrix.shape)
                    projection_matrix=projection_matrix[:,chunk_overlap:-chunk_overlap,:]
                    print(projection_matrix.shape)
                        
                
                '''
                center finding 
                '''
                if find_center:
                    if center_interp_function:
                        center_of_rotation_px=center_interp_function(proj_target_range[0])
                    else:
                        center_of_rotation_px=center_of_rotation_px_matx[0]

                    #grab a center slice
                    projection_matrix_centerfind=projection_matrix[:,int(sinogram_chunk_size/2):int(sinogram_chunk_size/2)+1,:]
                    print(f'center find manual: {center_of_rotation_px}')
                    if find_center_pc:
                        if degrees_per_section==180:
                            center_of_rotation_px=tomopy.find_center_pc(projection_matrix_centerfind[0],projection_matrix_centerfind[-1],tol=0.25)
                        else:
                            center_of_rotation_px=tomopy.find_center_pc(projection_matrix_centerfind[0],projection_matrix_centerfind[projection_matrix_centerfind.shape[0]//2],tol=0.25)
                        center_of_rotation_px=int(center_of_rotation_px)
                        print(f'center find pc: {center_of_rotation_px}')
                    if find_center_vo:
                        center_of_rotation_px=tomopy.find_center_vo(projection_matrix)
                        print(f'center find vo: {center_of_rotation_px}')
                    
                    #make dir for center finding output
                    # center_find_output_dir=os.path.join(out_dir+'_slice_'+str(proj_target_range[0]+int(sinogram_chunk_size/2)))
                    center_find_output_dir=os.path.join(out_dir+'_slice_'+str(proj_target_range[0]+int(sinogram_chunk_size/2))+'_r'+str(skew_angle))
                    Path(center_find_output_dir).mkdir(parents=True, exist_ok=True)
                    
                    #generate list of centers to iterate over
                    if center_find_range>1:
                        centers=np.arange(center_of_rotation_px-center_find_range/2,center_of_rotation_px+center_find_range/2,center_find_step)
                    else:
                        centers=[center_of_rotation_px]
                        projection_matrix_centerfind=projection_matrix

                    print('finding center')
                    with Pool(processes=int(center_find_range/center_find_step)) as pool:
                        if theta_clip:
                            bundle_list=[projection_matrix_centerfind,theta_,center_find_output_dir,vertical_velocity,count,str(skew_angle*100)[:3]]
                        else:
                            bundle_list=[projection_matrix_centerfind,theta,center_find_output_dir,vertical_velocity,count,str(skew_angle*100)[:3]]
                        pass_list = [(itm_result, bundle_list) for itm_result in range(0,len(centers))]
                        pass_list=zip(pass_list,centers)
                        pool.map(center_finding, pass_list)
                    # pool.close()
                    # pool.join()
                    count+=1
                    print('\nloop timing: %s'%(datetime.now()-loop_timing))
                    if is_hpc:
                        if save_temp_file:
                            print('copying temp chunk from tmp')
                            destination=os.path.join(proj_dir,'reconstructed')
                            source=filename_tif
                            shutil.copy(source, destination)
                        print('removing temp files from node')
                        shutil.rmtree(tmp_dir)
                    exit()
                
                else:
                    tompy_normalize=False
                    if tompy_normalize:
                        projection_matrix=tomopy.normalize_bg(projection_matrix, air=1)
                    
                    if center_interp_function:
                        center_of_rotation_px=center_interp_function(proj_target_range[0])
                    else:
                        center_of_rotation_px=center_of_rotation_px_matx[0]
                    start_recon=datetime.now()
                    print('starting recon')
                    astratoolbox=False
                    memmap_to_memmap=False
                    if astratoolbox:
                    
                        print(projection_matrix.shape)
                        print(np.min(projection_matrix))
                        print(np.max(projection_matrix))
                        # options = {'proj_type': 'cuda', 'method': 'FBP_CUDA'}
                        options = {'proj_type': 'cuda', 'method': 'FBP_CUDA','filter':'lanczos'}
                        # options = {'proj_type': 'cuda', 'method': 'SART_CUDA','num_iter':10*180}
                        # options = {'proj_type': 'cuda', 'method': 'SIRT_CUDA','num_iter':100}
                        recon = tomopy.recon(projection_matrix,
                                     theta,
                                     center=center_of_rotation_px_matx[0],
                                     algorithm=tomopy.astra,
                                     options=options,
                                     ncore=1)
                    elif memmap_to_memmap:
                            1+1
                            #do SR strat here but recon the slices into a memmap recon tiff files ;D
                    else:
                        # 'filter_name' flag used to denote butterworth image contrast filter*******
                        # recon = tomopy.recon(projection_matrix, theta, center=center_of_rotation_px_matx[0], algorithm='gridrec', filter_name='butterworth', sinogram_order=False)
                        butterworthpars = [.2,2]
                        butts=False
                        if butts:

                            recon = tomopy.recon(projection_matrix, theta, center=center_of_rotation_px_matx[0], algorithm='gridrec',filter_name='butterworth', filter_par=butterworthpars)
                        
                        # mask with circle cuz why not
                        else:
                            recon = tomopy.recon(projection_matrix, theta, center=center_of_rotation_px_matx[0], algorithm='gridrec')

                    recon = tomopy.circ_mask(recon, axis=0, ratio=1)
                        
                        # dx.write_tiff_stack(recon,out_dir+'full_angle_projpreRot'+"/rec",dtype="float32",start=proj_target_range[0])
                    
                    out_dir_recon=os.path.join(proj_dir,f'32bit_reconstructed_{sample_name}')
                    if is_hpc:
                        out_dir_recon=os.path.join('/gpfs','Labs','Cheng','phenome','Argonne','Reconstructed','A3_2022_December',project_name,sample_name,f'32bit_reconstructed_{unique_id}')

                    Path(out_dir_recon).mkdir(parents=True, exist_ok=True)
                    print(f'recon timing:{(datetime.now()-start_recon)}')
                    recon = tomopy.circ_mask(recon, axis=0, ratio=0.95)
                    for image_index,image in enumerate(recon):
                        tifffile.imwrite(os.path.join(out_dir_recon,'rec_'+str(proj_target_range[0]+image_index+chunk_overlap).zfill(5)+'.tif'), image[final_recon_crop])
                
                
                print('\nloop timing: %s'%(datetime.now()-loop_timing))
            print('\nfull script timing: %s'%(datetime.now()-start_full_script))
            if not full_recon:
                if is_hpc:
                    if save_temp_file:
                        print('copying temp chunk from tmp')
                        destination=os.path.join(proj_dir,'reconstructed')
                        source=filename_tif
                        shutil.copy(source, destination)
                    print('removing temp files from node')
                    shutil.copy(source, destination)
                print('chunk finished')
                exit()
    if is_hpc:
        print('removing temp files from node')
        shutil.rmtree(tmp_dir)
