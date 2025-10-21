#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Nov 24, Mod Jun 25
@author: priyom

"""

import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy import stats
from scipy.signal import find_peaks, savgol_filter
import pandas as pd
from datetime import datetime
from pathlib import Path
import matplotlib.image as mpimg
from scipy.interpolate import splrep, BSpline, UnivariateSpline
from scipy.optimize import curve_fit
from numpy import trapz
from matplotlib.pyplot import cm
from scipy.signal import find_peaks
from scipy.signal import hilbert
#calculate data time
def dt(fish_file):
    dob_date=fish_file[fish_file.find('dob_')+len('dob_'):fish_file.rfind('_im')]
    dob_date_split=dob_date.split('_')
    bday=datetime(int('20'+dob_date_split[2]),int(dob_date_split[0]),int(dob_date_split[1].lstrip('0')),0,0)
    im_date=fish_file[fish_file.find('im_')+len('im_'):fish_file.rfind('_tm')]
    im_date_split=im_date.split('_')
   
    tm_image=fish_file[fish_file.find('tm_')+len('tm_'):fish_file.rfind('_cmn')]
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
    return time_elapsed_sec_t

def common_cells(val_r,val_c,list2_r,list2_c,list2_erk):
    yes_same=0
    save_counter=np.NAN
    erk_value=np.NAN
    col_d=[]
   
    for counter2 in range(len(list2_r)):
        r2=list2_r[counter2]
        c2=list2_c[counter2]
        dist=np.sqrt((val_r-r2)**2+(val_c-c2)**2)
        col_d.append(dist)
        if(dist<15):
            yes_same=1
           
            save_counter=counter2
            break
    
    if(yes_same):
        erk_value=list2_erk[save_counter]
    return yes_same,save_counter,erk_value
###############given a file calculate erk ratios as a function of space
def erk_ratios(erk_file_nm):
    with open(erk_file_nm) as f:
        ratio_arr=np.array([line.strip().split('\t') for line in f])
        all_erk_vals=ratio_arr[:,1].astype(np.float64) 
        all_row_vals=ratio_arr[:,4].astype(np.float64)
        all_col_vals=ratio_arr[:,5].astype(np.float64)
        all_erk_vals_sorted_fn=[x for _, x in sorted(zip(all_col_vals, all_erk_vals), key=lambda pair: pair[0])]
        all_row_vals_sorted_fn=[x for _, x in sorted(zip(all_col_vals, all_row_vals), key=lambda pair: pair[0])]
        all_erk_vals_sorted_fn=np.array(all_erk_vals_sorted_fn)
        all_col_vals_sorted_fn=np.sort(all_col_vals)
        all_row_vals_sorted_fn=np.array(all_row_vals_sorted_fn)
    return all_erk_vals_sorted_fn,all_row_vals_sorted_fn,all_col_vals_sorted_fn
####################################################################
################given a file measure entpd fluorescence for each detected cell
def entpd_fluor(file_nm_entpd):
    entpd_data_file=np.load(file_nm_entpd,allow_pickle=True).item()
    image_file=file_nm_entpd.replace('_seg.npy','.tif')
    image_mat=mpimg.imread(image_file)#this is uint8
    entpd_masks=entpd_data_file['masks']
    masks_to_display=np.zeros((np.shape(entpd_masks)[0],np.shape(entpd_masks)[1]))
    entpd_label_vals=np.unique(entpd_masks).tolist()
    if(0 in entpd_label_vals):entpd_label_vals.remove(0)#0 is background
    #find cell mask, and calculate average fluorescent exoression
    all_entpd_fluor_vals=[]
    all_entpd_col_vals=[]
    all_entpd_row_vals=[]
    for lab_v in entpd_label_vals:
        [r,c]=np.where(entpd_masks==lab_v)
        rmean=np.mean(r)
        cmean=np.mean(c)
        #specific mask cell and fluorescence
        fluor_val=np.mean(image_mat[r,c])
        all_entpd_fluor_vals.append(fluor_val)
        all_entpd_col_vals.append(cmean)
        all_entpd_row_vals.append(rmean)
        #replace each pixel by fluorescent value
        masks_to_display[r,c]=fluor_val
    all_entpd_col_vals_fn=np.asarray(all_entpd_col_vals,dtype=np.float64)
    all_entpd_row_vals_fn=np.asarray(all_entpd_row_vals,dtype=np.float64)
    all_entpd_fluor_vals_sorted_fn=[x for _, x in sorted(zip(all_entpd_col_vals, all_entpd_fluor_vals), key=lambda pair: pair[0])]
    all_entpd_row_vals_sorted_fn=[x for _, x in sorted(zip(all_entpd_col_vals, all_entpd_row_vals), key=lambda pair: pair[0])]
    all_entpd_col_vals_sorted_fn=np.sort(all_entpd_col_vals)
    sz_col=np.shape(image_mat)[1]
    return all_entpd_row_vals_sorted_fn,all_entpd_col_vals_sorted_fn,all_entpd_fluor_vals_sorted_fn,sz_col
#####################################################################
#####update dict seg
def update_dict_seg(sum_per_cluster_c,mean_cluster_c,mean_entpd_cluster_c,len_entpd_cluster,\
                    high_sep_cc,cur_dict_seg,cur_dict_seg_tm,cur_dict_seg_entpd,\
                    cur_dict_seg_entpd_counter,act_time_counter_cc):
    all_keys=cur_dict_seg.keys()
    counter_key=0
    for ky in all_keys:
        dist_cluster=np.abs(mean_cluster_c-ky)
        if(dist_cluster<high_sep_cc):#same segment
            key=ky
            counter_key=1
            break
    if(counter_key==1):
        cur_dict_seg[key]=np.append(cur_dict_seg[key],sum_per_cluster_c)
        cur_dict_seg_tm[key]=np.append(cur_dict_seg_tm[key],act_time_counter_cc)
        cur_dict_seg_entpd[key]=np.append(cur_dict_seg_entpd[key],mean_entpd_cluster_c)
    else:
        cur_dict_seg[mean_cluster_c]=np.array([sum_per_cluster_c])
        cur_dict_seg_tm[mean_cluster_c]=np.array([act_time_counter_cc])
        cur_dict_seg_entpd[mean_cluster_c]=np.array([mean_entpd_cluster_c])
        cur_dict_seg_entpd_counter[mean_cluster_c]=np.array([len_entpd_cluster])
    return cur_dict_seg,cur_dict_seg_tm,cur_dict_seg_entpd,cur_dict_seg_entpd_counter
###################################################################
##update unseg dict
def update_dict_unseg(sum_per_cluster_c,mean_cluster_c,high_sep2_cc,cur_dict_unseg,cur_dict_unseg_tm,time_counter_cc):
    all_keys=cur_dict_unseg.keys()
    counter_key=0
    for ky in all_keys:
        dist_cluster=np.abs(mean_cluster_c-ky)
        if(dist_cluster<high_sep2_cc):#same segment
            key=ky
            counter_key=1
            break
    if(counter_key==1):
        cur_dict_unseg[key]=np.append(cur_dict_unseg[key],sum_per_cluster_c)
        
    else:
        cur_dict_unseg[mean_cluster_c]=np.array([sum_per_cluster_c])
        cur_dict_unseg_tm[mean_cluster_c]=time_counter_cc    
    return cur_dict_unseg,cur_dict_unseg_tm
##############################
def entpd_sep_seg(all_entpd_row_vals_sorted_c,all_entpd_col_vals_sorted_c,all_entpd_fluor_vals_sorted_c,\
                  all_erk_row_vals_sorted_c,all_erk_col_vals_sorted_c,all_erk_vals_sorted_c,\
                  high_sep_c,dict_seg_c,dict_seg_tm_c,dict_seg_entpd_c,dict_seg_entpd_counter_c,act_time_counter_c,cutoff_per_num_cells_c):
    #this function foes two things
    #calculates at the level of segments
    #finds erk values for each segment
    sorted_col_vals_diff=np.diff(all_entpd_col_vals_sorted_c)
    seg_lev=np.squeeze(np.where(sorted_col_vals_diff>high_sep_c))#150 in pixel is limit
    prev_sg=0
    which_erk_found_c=[]
    for sg in seg_lev:
        
        ind_cluster_row=all_entpd_row_vals_sorted_c[prev_sg:sg+1]
        ind_cluster_col=all_entpd_col_vals_sorted_c[prev_sg:sg+1]
        entpd_cluster=all_entpd_fluor_vals_sorted_c[prev_sg:sg+1]
        len_entpd_cluster=len(entpd_cluster)
        list_per_cluster=[]
        for rv in range(len(ind_cluster_row)):
            row_val=ind_cluster_row[rv]
            col_val=ind_cluster_col[rv]
            found_in_erk,sv_c,erk_val=common_cells(row_val,col_val, all_erk_row_vals_sorted_c, all_erk_col_vals_sorted_c, all_erk_vals_sorted_c)
            if(found_in_erk):
                list_per_cluster.append(erk_val)  
                which_erk_found_c.append(sv_c)
        sum_per_cluster=np.mean(list_per_cluster)
        mean_cluster=int(np.round(np.mean(ind_cluster_col),0))
        mean_entpd_cluster=np.sum(entpd_cluster)
       
        
        #update the dictionary, only if more than cutoff cells found
        if(len(list_per_cluster)>cutoff_per_num_cells_c):
            dict_seg_c,dict_seg_tm_c,dict_seg_entpd_c,dict_seg_entpd_counter_c=update_dict_seg(sum_per_cluster,mean_cluster,\
                                                                  mean_entpd_cluster,len_entpd_cluster,\
                                                                      high_sep_c,dict_seg_c, dict_seg_tm_c, dict_seg_entpd_c,\
                                                                          dict_seg_entpd_counter_c,act_time_counter_c)
        prev_sg=sg+1   
    ##add the last segment information
    ind_cluster_row=all_entpd_row_vals_sorted_c[prev_sg:]
    ind_cluster_col=all_entpd_col_vals_sorted_c[prev_sg:]
    entpd_cluster=all_entpd_fluor_vals_sorted_c[prev_sg:]
    len_entpd_cluster=len(entpd_cluster)
    list_per_cluster=[]
    for rv in range(len(ind_cluster_row)):
        row_val=ind_cluster_row[rv]
        col_val=ind_cluster_col[rv]
        found_in_erk,sv_c,erk_val=common_cells(row_val,col_val, all_erk_row_vals_sorted_c, all_erk_col_vals_sorted_c, all_erk_vals_sorted_c)
        if(found_in_erk):
            list_per_cluster.append(erk_val)  
            which_erk_found_c.append(sv_c)
    sum_per_cluster=np.mean(list_per_cluster)
    mean_cluster=int(np.round(np.mean(ind_cluster_col),0))
    mean_entpd_cluster=np.sum(entpd_cluster)
   
    #update the dictionary
    if(len(list_per_cluster)>cutoff_per_num_cells_c):
        dict_seg_c,dict_seg_tm_c,dict_seg_entpd_c,dict_seg_entpd_counter_c=update_dict_seg(sum_per_cluster,mean_cluster,\
                                                              mean_entpd_cluster,len_entpd_cluster,\
                                                                  high_sep_c,dict_seg_c, dict_seg_tm_c, dict_seg_entpd_c,\
                                                                  dict_seg_entpd_counter_c,act_time_counter_c)
    return dict_seg_c,dict_seg_tm_c,dict_seg_entpd_c,dict_seg_entpd_counter_c,which_erk_found_c
##################################################################
def entpd_sep_unseg(all_erk_vals_sorted_del_c,all_erk_col_vals_sorted_del_c,all_erk_row_vals_sorted_del_c,\
                    cur_seg_keys_c,high_sep2_c,dict_seg_left_c,dict_seg_left_tm_c,time_counter_c):
    #first seg
    fir_min=0
    first_loc_seg=cur_seg_keys_c[0]
    erk_indices_unseg=np.where((all_erk_col_vals_sorted_del_c>fir_min)&(all_erk_col_vals_sorted_del_c<first_loc_seg))
    sum_per_unseg_cluster=np.mean(all_erk_vals_sorted_del_c[erk_indices_unseg])
    mean_unseg_cluster=int(np.round(np.mean(all_erk_col_vals_sorted_del_c[erk_indices_unseg]),0))
    dict_seg_left_c,dict_seg_left_tm_c=\
        update_dict_unseg(sum_per_unseg_cluster,mean_unseg_cluster,high_sep2_c,dict_seg_left_c,dict_seg_left_tm_c,time_counter_c)
    for next_seg in range(1,len(cur_seg_keys_c)):
        min_seg_loc=cur_seg_keys_c[next_seg-1]
        max_seg_loc=cur_seg_keys_c[next_seg]
        #calculate indices and update here
        erk_indices_unseg=np.where((all_erk_col_vals_sorted_del_c>min_seg_loc)&(all_erk_col_vals_sorted_del_c<max_seg_loc))
        sum_per_unseg_cluster=np.mean(all_erk_vals_sorted_del_c[erk_indices_unseg])
        mean_unseg_cluster=int(np.round(np.mean(all_erk_col_vals_sorted_del_c[erk_indices_unseg]),0))
        dict_seg_left_c,dict_seg_left_tm_c=\
            update_dict_unseg(sum_per_unseg_cluster,mean_unseg_cluster,high_sep2_c,dict_seg_left_c,dict_seg_left_tm_c,time_counter_c)
    return dict_seg_left_c,dict_seg_left_tm_c    
####################################################################

base_path_files='/Volumes/phoenix/oscillator_paper_figures_data/fig_1b/';
sub_folder_space='period_seg_unseg/data/'
sub_folder_osci='period_seg_unseg/data/'
base_folder_dates=['dob_01_02_25','dob_10_28_24']
output_nms=['_model_transition_10_12_23/','_model_erk_gfp_03_08_24/']
unique_fish_num=['7','3']#
num_segs=[]
num_pts=5
high_sep=150
high_sep2=100
pix2mic=0.3
seg_thresh=50
bin_spacing=150
cutoff_per_num_cells=0
cutoff_per_num_cells_entpd=0
hex_colors=["#EE6677"]
fish_counter=0

fig, (ax1,ax2)=plt.subplots(2,1,sharex=True,height_ratios=[0.5,2])
fig.subplots_adjust(hspace=0.05)

for dates in base_folder_dates:
    #folder to get data 
    folder_write_nm_osci='output_erk_rat_'+dates+output_nms[fish_counter]
    full_path_tif=base_path_files+sub_folder_osci+dates+'/'
    full_path_data=base_path_files+sub_folder_osci+folder_write_nm_osci
    all_tif_files=glob.glob(full_path_tif+'MAX_half_f'+str(unique_fish_num[fish_counter])+'*cmn*.tif')

    
    #########################################################
    extract_session_nums=[]
    for tf in all_tif_files:
        ff_nm=tf.replace(full_path_tif,"")
        extract_session_nums.append(dt(ff_nm))
    all_session_nums_sorted=np.sort(extract_session_nums)
    all_fish_files_sorted=[x for _, x in sorted(zip(extract_session_nums, all_tif_files), key=lambda pair: pair[0])]
    #########################################################
    
    time_counter=0
    fig_counter=0
   
    dict_seg={}
    dict_seg_tm={}
    dict_seg_entpd={}
    dict_seg_entpd_counter={}
    dict_seg_left={}
    dict_seg_left_tm={}
    ########################################################
   
    all_fish_files_sorted_exist=[]
    all_session_num_exist_counter=[]
    for new_ff_counter in range(len(all_fish_files_sorted)):
        act_file_nm_str=full_path_data+'fish_'+str(unique_fish_num[fish_counter])+'_erk_ratios_sess_'+str(all_session_nums_sorted[new_ff_counter])+'.txt'
        fish_ff_nm=all_fish_files_sorted[new_ff_counter]
        phrase_to_rep=fish_ff_nm[fish_ff_nm.find('_cmn')+len('_cmn'):fish_ff_nm.rfind('.tif')]
        rep_str='_cmn'+phrase_to_rep+'.tif'
        entpd_str='_entpd'+phrase_to_rep+"_seg.npy"
        ff_file_entpd=all_fish_files_sorted[new_ff_counter].replace(rep_str,entpd_str)
        erk_file_exists=Path(act_file_nm_str).is_file()
        entpd_file_exists=Path(ff_file_entpd).is_file()
        if(erk_file_exists and entpd_file_exists):
            all_fish_files_sorted_exist.append(all_fish_files_sorted[new_ff_counter])
            all_session_num_exist_counter.append(all_session_nums_sorted[new_ff_counter])
            all_session_hrs_exist_counter=(all_session_num_exist_counter-all_session_num_exist_counter[0])/(60.*60.)
    #######################################################
    erk_unseg_ar=[]
    for new_exist_counter in range(len(all_fish_files_sorted_exist)):
        act_file_nm_str=full_path_data+'fish_'+str(unique_fish_num[fish_counter])+'_erk_ratios_sess_'+str(all_session_num_exist_counter[new_exist_counter])+'.txt'
        fish_ff_exist_nm=all_fish_files_sorted_exist[new_exist_counter]
        phrase_to_rep=fish_ff_exist_nm[fish_ff_exist_nm.find('_cmn')+len('_cmn'):fish_ff_exist_nm.rfind('.tif')]
        rep_str='_cmn'+phrase_to_rep+'.tif'
        entpd_str='_entpd'+phrase_to_rep+"_seg.npy"
        ff_file_entpd=fish_ff_exist_nm.replace(rep_str,entpd_str)
        act_time_counter=all_session_hrs_exist_counter[new_exist_counter]
        ##################################################
        ##extract all erk ratios here
        all_erk_vals_sorted,all_erk_row_vals_sorted,all_erk_col_vals_sorted=erk_ratios(act_file_nm_str)
        all_erk_vals_sorted_con=np.floor(all_erk_vals_sorted)
        num_on=np.where(all_erk_vals_sorted_con>0)
        #print(len(num_on[0]),len(all_erk_vals_sorted_con))
        #################################################
        ##extrack all entpd data here
        all_entpd_row_vals_sorted,all_entpd_col_vals_sorted,all_entpd_fluor_vals_sorted,sz_col=entpd_fluor(ff_file_entpd)
        ##################################################
        ##update entpd, erk intensities
        dict_seg,dict_seg_tm,dict_seg_entpd,dict_seg_entpd_counter,which_erk_found=\
            entpd_sep_seg(all_entpd_row_vals_sorted,all_entpd_col_vals_sorted,all_entpd_fluor_vals_sorted,\
                              all_erk_row_vals_sorted,all_erk_col_vals_sorted,all_erk_vals_sorted_con,\
                              high_sep,dict_seg,dict_seg_tm,dict_seg_entpd,dict_seg_entpd_counter,act_time_counter,cutoff_per_num_cells)    
        ###################################################    
        ##update entpd, erk in non segmented areas
     
        all_erk_vals_sorted_del=np.delete(all_erk_vals_sorted_con,which_erk_found)
        mn_val=np.mean(all_erk_vals_sorted_del,axis=0)
        erk_unseg_ar.append(mn_val)
        all_erk_col_vals_sorted_del=np.delete(all_erk_col_vals_sorted,which_erk_found)
        all_erk_row_vals_sorted_del=np.delete(all_erk_row_vals_sorted,which_erk_found)
        ##################################################
        ##################################################

        time_counter+=1

   
    #calculate overall segment period
    erk_unseg_ar=np.array(erk_unseg_ar)
    tr_overall,_=find_peaks(-erk_unseg_ar)
    per_unseg=np.diff(all_session_hrs_exist_counter[tr_overall])
    #sort all the unseg keys
    sorted_dict_seg=dict(sorted(dict_seg.items()))
    sorted_dict_seg_tm=dict(sorted(dict_seg_tm.items()))
    sorted_dict_seg_entpd=dict(sorted(dict_seg_entpd.items()))
    sorted_dict_seg_entpd_counter=dict(sorted(dict_seg_entpd_counter.items()))
    sorted_k=sorted_dict_seg.keys()
    all_min_seg_collect=[]
    all_max_seg_collect=[]
    color_k=0
    cm_map2=cm.get_cmap('Greens')
    color2 = iter(cm_map2(np.linspace(0, 1, len(sorted_k))))
    chosen_segs=[]
    chosen_pos=[]
    tm_seg=[]
    for s_ky in sorted_k:
        erk_tm_course=sorted_dict_seg[s_ky]
        tm_hrs=sorted_dict_seg_tm[s_ky]
        entpd_val=sorted_dict_seg_entpd[s_ky]
        if(len(tm_hrs)==num_pts):
            tr,_=find_peaks(-erk_tm_course)
           
            if(len(tr)==2):
           
                time_per_seg=np.diff(tm_hrs[tr])
                rat_is=time_per_seg/per_unseg
               
            else:
                
                rat_is=5#this is a large number, which won't be reached from the data. Its meant to indicate indeterminany
            ax1.scatter(entpd_val[-1],rat_is,c="#BB5566",s=50)
            ax2.scatter(entpd_val[-1],rat_is,c="#BB5566",s=50)
            
      
        color_k+=1
 
    fish_counter+=1
    ##############
ax1.set_ylim(4.9,5.2)
ax2.set_ylim(0.25,1.75)
ax2.set_yticks([0.5,1.0,1.5])
ax2.set_xticks(np.arange(50,350,100))
ax1.set_yticks([])

ax1.spines.bottom.set_visible(False)
ax2.spines.top.set_visible(False)
ax1.xaxis.tick_top()

ax1.tick_params(top=False,labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()

d = .5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

ax2.tick_params(axis='both', which='major', labelsize=24)
ax2.tick_params(axis='both', which='minor', labelsize=24)