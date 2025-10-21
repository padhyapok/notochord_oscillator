#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Sep 21 21:19:44 2023

@author: priyom
"""
import glob
import os
import pandas as pd
import numpy as np
import matplotlib.colors as cl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy import stats
from datetime import datetime
import matplotlib.image as mpimg
import seaborn as sns
from matplotlib.pyplot import cm
low_sep=100
high_sep=150
base_path_files='/Volumes/phoenix/oscillator_paper_figures_data/';
fig_sv="fig_1b/all_figs/fig_2a_entpd_7dpf_time_ser/"
sub_folder='fig1/entpd/data/'
base_folder_dates=['dob_04_14_24/']
fish_num_list=['1']
counter_ff=0
plt.figure(1)
######colormap properties
min_v,max_v=0.2,1.0
n=20
color_min=0
color_max=12
ori_map=plt.cm.Blues
colors = ori_map(np.linspace(min_v, max_v, n))
cmap_new = cl.LinearSegmentedColormap.from_list("mycmap", colors)
max_crop_pos=3500
############
over_fish_cluster_space=[]
for dates in base_folder_dates:
    #folder to get data 
    image_full_path=base_path_files+sub_folder+dates
    entpd_time_series=[]
    time_pts=[]
    fish_num=str(fish_num_list[counter_ff])
    all_entpd_files=glob.glob(image_full_path+'MAX_half_f'+fish_num+'*entpd*seg.npy')
    num_clusters=[]
    list_num=[]
    for ff2 in all_entpd_files:
        fish_file=ff2.replace(image_full_path,"")
        #calculate dob and time
        dob_date=fish_file[fish_file.find('dob_')+len('dob_'):fish_file.rfind('_im')]
        dob_date_split=dob_date.split('_')
        bday=datetime(int('20'+dob_date_split[2]),int(dob_date_split[0]),int(dob_date_split[1].lstrip('0')),0,0)
        im_date=fish_file[fish_file.find('im_')+len('im_'):fish_file.rfind('_tm')]
        im_date_split=im_date.split('_')
        im_dt_repl=im_date_split[1].replace('0','')
        tm_image=fish_file[fish_file.find('tm_')+len('tm_'):fish_file.rfind('_entpd')]
        tm_min=int(tm_image[-4:-2])
        tm_hr=int(tm_image[0:-4])
        if(tm_hr==12):
            tm_hr=0
        if('pm' in tm_image):
            tm_hr+=12
        im_date=datetime(int('20'+im_date_split[2]),int(im_date_split[0]),int(im_date_split[1].lstrip('0')),tm_hr,tm_min)
        time_diff=im_date-bday
        time_diff_sec=time_diff.total_seconds()
        time_elapsed_sec_t=int(time_diff_sec)
        print(bday,im_date,time_elapsed_sec_t)
        list_num.append(time_elapsed_sec_t)
    #sort entpd file name according to list num
    entpd_file_names_sor=[x for _, x in sorted(zip(list_num, all_entpd_files), key=lambda pair: pair[0])]
    time_pts_sorted=np.sort(list_num)
    time_pts_hrs_entpd=(time_pts_sorted-time_pts_sorted[0])/(60.*60.)#zero with respect to erk time point 0
    time_counter=0
    for ff in entpd_file_names_sor:   
        fish_file=ff.replace(image_full_path,"")
        
        entpd_data_file=np.load(ff,allow_pickle=True).item()
        #subsitute image file as well
        image_file=ff.replace('_seg.npy','.tif')
        image_mat=mpimg.imread(image_file)#this is uint8
        entpd_masks=entpd_data_file['masks']
        masks_to_display=np.zeros((np.shape(entpd_masks)[0],np.shape(entpd_masks)[1]))
        entpd_label_vals=np.unique(entpd_masks).tolist()
        if(0 in entpd_label_vals):entpd_label_vals.remove(0)#0 is background
        #find cell mask, and calculate average fluorescent exoression
        all_fluor_vals=[]
        all_col_vals=[]
        all_row_vals=[]
        for lab_v in entpd_label_vals:
            [r,c]=np.where(entpd_masks==lab_v)
            rmean=np.mean(r)
            cmean=np.mean(c)
            #specific mask cell and fluorescence
            
            fluor_val=np.mean(image_mat[r,c])
            all_fluor_vals.append(fluor_val)
            all_col_vals.append(cmean)
            all_row_vals.append(rmean)
            #replace each pixel by fluorescent value
            masks_to_display[r,c]=fluor_val
        
        
        fig,ax=plt.subplots()
        #ax.imshow(masks_to_display[:,xpos:xpos+wd],cmap=cmap_new,vmin=color_min,vmax=color_max)
        ax.imshow(masks_to_display[:,:max_crop_pos],cmap=cmap_new,vmin=color_min,vmax=color_max)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_frame_on(False)
       
        time_counter+=1
        
    ##plot the colorbar once at end
    ##################################
    fig2,ax2=plt.subplots()
    sm=cm.ScalarMappable(cmap=cmap_new, norm=plt.Normalize(vmin=color_min,vmax=color_max))
    cbar=fig.colorbar(sm,cax=ax2,orientation='horizontal')
    cbar.ax.set_aspect(0.8)
    cbar.ax.tick_params(size=0)
    cbar.outline.set_visible(False)
    tl=np.linspace(color_min,color_max,3)
    cbar.set_ticks(tl)
    cbar.ax.set_xticklabels(tl)
    cbar.ax.tick_params(labelsize=20,labelfontfamily='Arial')
    
    plt.show()
    #################################
    counter_ff+=1

