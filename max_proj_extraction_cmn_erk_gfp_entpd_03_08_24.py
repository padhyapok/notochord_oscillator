#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 17:09:00 2024
#modified on 02/27/24, edited on 03_05_24, simple code to extract till the mid plane of the notochord tube
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
import cv2
##################################################################
######################parameters##################################
high_thresh=15
low_thresh=8
dil_kernel=3
##################################################################
#linear equation for thresholding z
def intensity_func(zcoor,zmin,zmax,high_thresh,low_thresh):
    slp=(high_thresh-low_thresh)/(zmin-zmax)
    intp=(low_thresh*zmin-high_thresh*zmax)/float(zmin-zmax)    
    return slp*zcoor+intp

base_path_files="/Volumes/phoenix/oscillator_paper_figures_data/fig1/"
sub_folder="raw_data/"
base_folder_dates=["dob_01_02_25_im_01_09_25/"]
folder_write="oscillator_segment/data/dob_01_02_25/"

for dates in base_folder_dates:#loop over multiple time series
    image_full_path=base_path_files+sub_folder+dates
    image_write_path=base_path_files+folder_write
    all_image_files=glob.glob(image_full_path+'*f4*cmn*tif')
   
    for ff in all_image_files:
        #max file name
        cmn_erk_im = io.imread(ff)
        ff2=ff.replace("cmn","entpd")
        entpd_im=io.imread(ff2)
        cmn_erk_im_t=np.transpose(cmn_erk_im,(2,1,0))#this is arranged as (x,y,z)
        entpd_im_t=np.transpose(entpd_im,(2,1,0))
        max_file_nm='MAX_half_'+ff.replace(image_full_path,'')
        max_file_entpd_nm='MAX_half_'+ff2.replace(image_full_path,'')
        #loop through each plane and get out the half tube
        x_dim=np.shape(cmn_erk_im_t)[0]
        #loop over each x position
        big_cmn=np.zeros((np.shape(cmn_erk_im_t)[0],np.shape(cmn_erk_im_t)[1],np.shape(cmn_erk_im_t)[2]),dtype=np.uint8)
        #same dimensions
        big_entpd=np.zeros((np.shape(cmn_erk_im_t)[0],np.shape(cmn_erk_im_t)[1],np.shape(cmn_erk_im_t)[2]),dtype=np.uint8)
        
        for xval in range(0,x_dim):
           
            yz_plane_cmn=cmn_erk_im_t[xval,:,:]
            yz_plane_entpd=entpd_im_t[xval,:,:]
            yz_plane_cmn_thresh=np.zeros((np.shape(yz_plane_cmn)[0],np.shape(yz_plane_cmn)[1]),dtype=np.uint8)
            yz_plane_temp=np.zeros((np.shape(yz_plane_cmn)[0],np.shape(yz_plane_cmn)[1]),dtype=np.uint8)#this is reverse
            #threshold cmn plane
            dim_z=np.shape(yz_plane_cmn)[1]
            zmin_val=0
            zmax_val=dim_z
            for zval in range(dim_z):
                thresh_val=intensity_func(zval,zmin_val,zmax_val,high_thresh,low_thresh)
                yz_plane_cmn_thresh[:,zval]=yz_plane_cmn[:,zval]>thresh_val
           
           
            yz_plane_points=np.argwhere(yz_plane_cmn_thresh)
            #y dimension
            y_plane_points=yz_plane_points[:,0]
            y_plane_points_re=np.reshape(y_plane_points,(len(y_plane_points),1))
            #z dimension
            z_plane_points=yz_plane_points[:,1]
            z_plane_points_re=np.reshape(z_plane_points,(len(z_plane_points),1))
            
            mean_y=int(np.mean(y_plane_points_re))
            mean_z=int(np.mean(z_plane_points_re))
            #all y points till mean_z
            yz_plane_temp[:,0:int(mean_z)]=1
            #plt.imshow(yz_plane_cmn_thresh)
            #plt.scatter(mean_z,mean_y,s=20,c='r')
            #plt.show()
            
            cmn_p=np.multiply(yz_plane_temp,yz_plane_cmn)
            entpd_p=np.multiply(yz_plane_temp,yz_plane_entpd)
            #plt.imshow(cmn_p)
            #plt.show()
            #plt.imshow(yz_plane_temp)
            #plt.show()
            big_cmn[xval,:,:]=cmn_p
            big_entpd[xval,:,:]=entpd_p
            #plt.imshow(yz_plane_cmn_thresh_m)
            #plt.scatter(z_ellipse_coord_int,y_ellipse_coord_int)
            #plt.show()
        #cmn files
        final_cmn_max=np.amax(big_cmn,axis=2)
        final_cmn_max_trans=np.transpose(final_cmn_max,(1,0))
        plt.imshow(np.transpose(final_cmn_max,(1,0)),cmap='gray')
        plt.show()
        #entpd files
        final_entpd_max=np.amax(big_entpd,axis=2)
        final_entpd_max_trans=np.transpose(final_entpd_max,(1,0))
        
        cv2.imwrite(base_path_files+folder_write+max_file_nm,final_cmn_max_trans)
        cv2.imwrite(base_path_files+folder_write+max_file_entpd_nm,final_entpd_max_trans)
        
 
