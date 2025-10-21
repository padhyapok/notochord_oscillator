#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 23:24:38 2024
#original trial by cross-correlation from Tianxiao Hao
@author: priyom
"""
import numpy as np
from skimage import io,filters,morphology

from skimage.segmentation import find_boundaries,flood
from skimage.morphology import binary_dilation,disk,binary_erosion,square,ball,remove_small_objects
import matplotlib.pyplot as plt
from skimage import measure
import glob
import re
from datetime import datetime
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import splrep, BSpline
from skimage.exposure import equalize_hist
from skimage.registration import phase_cross_correlation
from tifffile import imwrite
radiuses = np.arange(60, 150, 5)
thick = 20
windows = []
for radius in radiuses:
    window = np.zeros([radius * 2 + thick, radius * 2 + thick])
    
    window = cv2.circle(window, center=(int(radius + thick / 2.0), int(radius + thick / 2.0)), radius=radius, color=1,
                        thickness=thick)
    window = window - np.mean(window)
    windows.append(window) 
#########################
#####stitch multiple images
def stitch_tiles(shift_row,shift_col,num_cols_res_t,big_image_c,template_c):
    big_ydim=np.shape(big_image_c)[0]
    big_xdim=np.shape(big_image_c)[1]
    template_ydim=np.shape(template_c)[0]
    template_xdim=np.shape(template_c)[1]
    if(shift_row>=0):
        
        num_row=int(shift_row)
        if(num_row+template_ydim>big_ydim):
            template_enlarge=np.zeros((template_ydim+num_row,template_xdim),dtype=np.uint8)
            template_enlarge[num_row:(template_ydim+num_row),:template_xdim]=template_c
            num_row_large=(template_ydim+num_row)-big_ydim
            final_big_enlarge=np.zeros((big_ydim+num_row_large,big_xdim),dtype=np.uint8)
            final_big_enlarge[0:big_ydim,:]=big_image_c
        else:
            template_enlarge=np.zeros((big_ydim,template_xdim),dtype=np.uint8)
            #print('dim',np.shape(template_enlarge),np.shape(file_to_read_g),len(range(num_row,dim_y_f)),len(range(0,dim_y_f)))
            template_enlarge[num_row:(template_ydim+num_row),:template_xdim]=template_c
            final_big_enlarge=big_image_c.copy()
    else:
        for_template=big_ydim-(template_ydim-int(abs(shift_row)))
        template_enlarge=np.zeros((template_ydim+for_template,template_xdim),dtype=np.uint8)
        template_enlarge[0:template_ydim,:template_xdim]=template_c
        final_big_enlarge=np.zeros((big_ydim+int(abs(shift_row)),big_xdim),dtype=np.uint8)
        final_big_enlarge[int(abs(shift_row)):,:]=big_image_c
    if(shift_col>=0):
        new_big_im=np.concatenate((final_big_enlarge[:,0:-num_cols_res-int(abs(shift_col))],template_enlarge),axis=1)
    else:
        new_big_im=np.concatenate((final_big_enlarge[:,:],template_enlarge[:,int(abs(shift_col)):]),axis=1)
    return new_big_im

mic_per_pix_xy=0.3
mic_per_pix_z=1.75
step_test=25
num_col_list=np.arange(50,62,2)
rescale_z_fac=mic_per_pix_z/mic_per_pix_xy
base_path_files="/Volumes/phoenix/oscillator_paper_figures_data/fig2/"
sub_folder="raw_data/"
dates="dob_04_21_25"
folder_write='spry2/data/'+dates+'/'
image_full_path=base_path_files+sub_folder+dates+'/'
all_image_files=glob.glob(image_full_path+'f1*211*spry2*tif')
for ff in all_image_files:
    spry_ff=io.imread(ff)
    tile_dim=np.shape(spry_ff)[0]
    entpd_im_nm=ff.replace('spry2','entpd')
    entpd_ff=io.imread(entpd_im_nm)
    max_file_spry_nm='MAX_half_'+ff.replace(image_full_path,'')
    max_file_entpd_nm='MAX_half_'+entpd_im_nm.replace(image_full_path,'')
    for slc in range(tile_dim):
        #entpd
        spry_im = spry_ff[slc,:,:,:]
        spry_im_t=np.transpose(spry_im,(2,1,0))#this is arranged as (x,y,z)
        entpd_im=entpd_ff[slc,:,:,:]
        entpd_im_t=np.transpose(entpd_im,(2,1,0))
        max_all_sig_trans=np.amax(spry_im_t,axis=2)
        max_all_sig=np.transpose(max_all_sig_trans,(1,0))
        x_dim=np.shape(spry_im_t)[0]
        y_dim=np.shape(spry_im_t)[1]
        z_dim=np.shape(spry_im_t)[2]     
        collect_pars=[[],[],[]]
        xvals_to_fit=np.array(range(0,x_dim,step_test))
        for xval in xvals_to_fit:
            #print(xval)
            yz_plane_spry=spry_im_t[xval,:,:]
            yz_plane_spry_re_z=cv2.resize(yz_plane_spry, (int(z_dim*rescale_z_fac),y_dim))
            yz_plane_spry_re_z = equalize_hist(yz_plane_spry_re_z)
            yz_plane_spry_re_z = cv2.GaussianBlur(yz_plane_spry_re_z, (49, 49), 0)
            #rescale by mean
            yz_plane_spry_re_zm=yz_plane_spry_re_z-np.mean(yz_plane_spry_re_z)
            #copy as float 64
            yz_plane_spry_re_zm_f=yz_plane_spry_re_zm.astype('float64').copy()
            #for each plane test fitting of each radius
            corrs = pd.DataFrame(columns=['radius', 'row_v', 'col_v', 'corr','wind'])
            for i, radius in enumerate(radiuses):
                corr = scipy.signal.correlate(yz_plane_spry_re_zm_f, windows[i],mode='valid')
                #print('tes',np.shape(corr))
                row_v, col_v = np.unravel_index(np.argmax(corr), corr.shape)
                cor = corr[row_v, col_v]
                #print(radius,cor)
                corrs.loc[i] = [radius, row_v,col_v, cor,i]    
            corrs = corrs.sort_values(by='corr', ascending=False).reset_index(drop=True)
            row_fin, col_fin, radius, ind_w = corrs.loc[0, ['row_v', 'col_v', 'radius','wind']]
            collect_pars[0].append(col_fin+radius+thick/2 )
            collect_pars[1].append(row_fin+radius+thick/2 )
            collect_pars[2].append(radius)
        
        collect_pars=np.array(collect_pars)
        collect_pars_t=collect_pars.T
        z_cen_all=collect_pars_t[:,0]
        y_cen_all=collect_pars_t[:,1]
        rad_all=collect_pars_t[:,2]#
        all_x_pos=np.array(range(0,x_dim))
        A = np.column_stack([xvals_to_fit, np.ones_like(xvals_to_fit)])
        new_yvals= np.linalg.lstsq(A, y_cen_all, rcond=None)[0].squeeze()
        new_zvals = np.linalg.lstsq(A, z_cen_all, rcond=None)[0].squeeze()
        new_radvals = np.linalg.lstsq(A, rad_all, rcond=None)[0].squeeze()
        #draw all x pos
        big_spry=np.zeros((np.shape(spry_im_t)[0],np.shape(spry_im_t)[1],int(z_dim*rescale_z_fac)),dtype=np.uint8)
        big_entpd=np.zeros((np.shape(spry_im_t)[0],np.shape(spry_im_t)[1],int(z_dim*rescale_z_fac)),dtype=np.uint8)
        for xp in range(len(all_x_pos)):
            xv=all_x_pos[xp]
            yv=new_yvals[0]*xv+new_yvals[1]
            zv=new_zvals[0]*xv+new_zvals[1]
            rv=new_radvals[0]*xv+new_radvals[1]
            ##use the circle info to get spry info
            im_spry = spry_im_t[xp,:,:]
            im_spry_rescale=cv2.resize(im_spry, (int(z_dim*rescale_z_fac),y_dim))  
            im_spry_copy_fir=np.zeros((np.shape(im_spry_rescale)[0],np.shape(im_spry_rescale)[1]),dtype=np.uint8)
            im_spry_copy = cv2.circle(im_spry_copy_fir, center=(int(zv) , int(yv) ),
                            radius=int(rv), color=1, thickness=thick)
            im_spry_copy_uint8=im_spry_copy.astype('uint8').copy()
            #test half..
            im_spry_copy_uint8_half=np.copy(im_spry_copy_uint8)
            im_spry_copy_uint8_half[:,int(zv):]=0  
            spry_mask=np.multiply(im_spry_rescale,im_spry_copy_uint8_half)
            ##use the circle info to get entpd info
            im_entpd = entpd_im_t[xp,:,:]
            im_entpd_rescale=cv2.resize(im_entpd, (int(z_dim*rescale_z_fac),y_dim))  
            im_entpd_copy_fir=np.zeros((np.shape(im_entpd_rescale)[0],np.shape(im_entpd_rescale)[1]),dtype=np.uint8)
            im_entpd_copy = cv2.circle(im_entpd_copy_fir, center=(int(zv) , int(yv) ),
                            radius=int(rv), color=1, thickness=thick)
            im_entpd_copy_uint8=im_entpd_copy.astype('uint8').copy()
            #test half..
            im_entpd_copy_uint8_half=np.copy(im_entpd_copy_uint8)
            im_entpd_copy_uint8_half[:,int(zv):]=0  
            entpd_mask=np.multiply(im_entpd_rescale,im_entpd_copy_uint8_half)
            
            big_spry[xp,:,:]=spry_mask
            big_entpd[xp,:,:]=entpd_mask
            #plt.imshow(spry_mask,vmin=0,vmax=20)
            #plt.show()
            
        #entpd
        final_spry_max=np.amax(big_spry,axis=2)
        final_spry_max_trans=np.transpose(final_spry_max,(1,0))
        final_entpd_max=np.amax(big_entpd,axis=2)
        final_entpd_max_trans=np.transpose(final_entpd_max,(1,0))
        #stitch images using pixel registration
        if(slc==0):
            prev_im=np.copy(max_all_sig)
            big_spry_im=np.copy(final_spry_max_trans)
            big_entpd_im=np.copy(final_entpd_max_trans)
        else:
            min_diff=100
            for vv in num_col_list:
                im1=prev_im[:,-vv:]
                im2=max_all_sig[:,0:vv]
                #try shift over num cols
              
                shift, error, diffphase = phase_cross_correlation(im1,im2)
                if(abs(diffphase)<min_diff):
                    shift_v=shift
                    min_diff=abs(diffphase)
                    num_cols_res=vv
            #print(shift_v,min_diff,np.shape(im1))
            ###shift positive, second image moves to the right or down
            ###shift negative, second image moves up or left
            prev_im=np.copy(max_all_sig)
            big_spry_im=stitch_tiles(shift_v[0],shift_v[1],num_cols_res,big_spry_im,final_spry_max_trans)
            big_entpd_im=stitch_tiles(shift_v[0],shift_v[1],num_cols_res,big_entpd_im,final_entpd_max_trans)
   
    cv2.imwrite(base_path_files+folder_write+max_file_entpd_nm,big_entpd_im)
    cv2.imwrite(base_path_files+folder_write+max_file_spry_nm,big_spry_im)
    #plt.imshow(big_spry_im,vmin=0,vmax=80)
    #plt.show()