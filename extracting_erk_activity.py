"""
Created 03_06_24
Modified 05_15_25
@author: priyom
Code for extracting Erk activity
"""
from cellpose import models
from cellpose.io import imread
import matplotlib.pyplot as plt
import glob
import numpy as np
import tifffile
import re
import os.path
import os
from PIL import Image
from skimage import filters, measure, morphology, segmentation
from datetime import datetime
import os.path 
###############all the functions
model_nuc  = models.CellposeModel(pretrained_model="/Volumes/phoenix/oscillator_paper_figures_data/fig1/cellpose_training_data/oscillator_gfp_nuclei/models/oscillator_gfp_nuclei_03_01_25")
base_path_files='/Volumes/phoenix/oscillator_paper_figures_data/fig1/';
sub_folder='oscillator_segment/data/'
base_folder_dates=['dob_02_23_24']
channe=[0,0]
rtotal=ctotal=100
for dates in base_folder_dates:
    image_full_path=base_path_files+sub_folder+dates+'/'
    folder_write_nm='output_erk_rat_'+dates+'_model_erk_gfp_03_08_24/'
    image_write_path=base_path_files+sub_folder+folder_write_nm
   
    all_image_files=glob.glob(image_full_path+'MAX_half_f5_*cmn*tif')
    for ff in all_image_files:
        #print(ff)
        fish_file=ff.replace(image_full_path,"")
        #calculate dob and time
        dob_date=fish_file[fish_file.find('dob_')+len('dob_'):fish_file.rfind('_im')]
        dob_date_split=dob_date.split('_')
        bday=datetime(int('20'+dob_date_split[2]),int(dob_date_split[0]),int(dob_date_split[1].lstrip('0')),0,0)
        im_date=fish_file[fish_file.find('im_')+len('im_'):fish_file.rfind('_tm')]
        im_date_split=im_date.split('_')
        im_dt_repl=im_date_split[1].replace('0','')
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
        print(bday,im_date,time_elapsed_sec_t)
        #find fish number
        fish_num=fish_file[fish_file.find('half_f')+len('half_f'):fish_file.rfind('_dob')] 
        file_to_write_rats=image_write_path+'fish_'+fish_num+'_erk_ratios_sess_'+str(time_elapsed_sec_t)+'.txt'
        file_to_write_map=image_write_path+'fish_'+fish_num+'_map_sess_'+str(time_elapsed_sec_t)+'.txt'
        #read file here
        read_file=imread(ff)
        read_file_alt=Image.open(ff)
        
        erk_file_manual_cor=ff.replace('.tif','_seg.npy')
        #print(erk_file_manual_cor)
        if(os.path.isfile(erk_file_manual_cor)):
            
                fid1=open(file_to_write_rats,'w')
                data_file=np.load(erk_file_manual_cor,allow_pickle=True).item()
                masks=data_file['masks']
                masks_copy=np.zeros((np.shape(masks)[0],np.shape(masks)[1]),dtype=np.float64)
                label_vals=np.unique(masks).tolist()
                #print('len of total labels',len(label_vals))
                #now for each file compute the labels and check individual cells
                #interesting thing to note for cellpose- first index is the 0 pixel background,
                for lab_v in range(1,len(label_vals)):
                    [r,c]=np.where(masks==label_vals[lab_v])
                    #additionally write x and y positions
                    rmean_pos=np.mean(r)
                    cmean_pos=np.mean(c)
                    rmin,rmax,cmin,cmax=np.min(r),np.max(r),np.min(c),np.max(c)
                    #split remaing space to add to the row 
                    rrange=rmax-rmin
                    split_r=rtotal-rrange
                    if(split_r%2==0):
                        val_r1,val_r2=int(split_r/2.0),int(split_r/2.0)
                    else:
                        val_r1,val_r2=int((split_r-1)/2.0),int((split_r-1)/2.0)+1
                    #split remaning space to add to the col
                    crange=cmax-cmin
                    split_c=ctotal-crange
                    if(split_c%2==0):
                        val_c1,val_c2=int(split_c/2.0),int(split_c/2.0)
                    else:
                        val_c1,val_c2=int((split_c-1)/2.0),int((split_c-1)/2.0)+1
                    #final values for extraction
                    #rows
                    val_r1_lim=max(0,rmin-val_r1)
                    val_r2_lim=min(rmax+val_r2,np.shape(read_file)[0])
                    #cols
                    val_c1_lim=max(0,cmin-val_c1)
                    val_c2_lim=min(cmax+val_c2,np.shape(read_file)[1])
    
                    temp_im=np.zeros((np.shape(read_file)[0],np.shape(read_file)[1]),dtype=np.uint8)
                    temp_im[r,c]=1
                    cell_of_int=np.multiply(temp_im,read_file_alt)
                    extract_cell_of_int=cell_of_int[val_r1_lim:val_r2_lim,val_c1_lim:val_c2_lim]
                    
                    masks_nuc, flows_nuc, styles_nuc = model_nuc.eval(extract_cell_of_int, diameter=None, channels=channe) #letting them set the diamter on a per image basis, have to decide if correct
                    
                    #################################################uncomment for image display####################
                    #fig, ax = plt.subplots(1,3,figsize=(15,4))
                    #ax[0].imshow(extract_cell_of_int,cmap='gray')
                    ################################################################################################
                    #rest of processing done on extracted cells
                    [rev,cev]=np.where(extract_cell_of_int)
                    mean_intensity_cell=np.mean(extract_cell_of_int[rev,cev])
                    if(masks_nuc.any()):
                        [rnuc,cnuc]=np.where(masks_nuc)
                        masks_nuc_bool=np.zeros((np.shape(masks_nuc)[0],np.shape(masks_nuc)[1]),dtype='bool')
                        masks_nuc_bool[rnuc,cnuc]=1
                         
                        #erode nuclear mask
                        dilate_nuc=morphology.binary_dilation(masks_nuc_bool,footprint=morphology.disk(3))
                        erode_nuc=morphology.binary_erosion(masks_nuc_bool,footprint=morphology.disk(1))
                        mask_rim=np.logical_and(dilate_nuc, ~masks_nuc_bool)
                        mask_rim_ex=np.multiply(extract_cell_of_int,mask_rim)
                        
                        [rnuc2,cnuc2]=np.where(erode_nuc)
                            
                        [rcyto,ccyto]=np.where(extract_cell_of_int)
                       
                        cyto_arr=np.array(list(zip(rcyto,ccyto)))
                        cyto_arr_tup=set([tuple(x) for x in cyto_arr])
                        nuc_arr=np.array(list(zip(rnuc,cnuc)))
                        nuc_arr_tup=set([tuple(x) for x in nuc_arr])
                       
                        uncommon_rows=np.array([x for x in cyto_arr_tup if x not in nuc_arr_tup])
                        r_cyto_s=uncommon_rows[:,0]
                        c_cyto_s=uncommon_rows[:,1]
                            
                        [r_cyto_s2,c_cyto_s2]=np.where(mask_rim_ex)#
                       
                            
                        
                        mean_nuc=np.mean(extract_cell_of_int[rnuc2,cnuc2])
                        mean_cyto=np.mean(extract_cell_of_int[r_cyto_s2,c_cyto_s2])
                        ratio_is=mean_cyto/float(mean_nuc)
                        masks_copy[r,c]=np.round(ratio_is,3)
                        size_cell=len(r)
                        fid1.write(str(label_vals[lab_v])+"\t"+str(np.round(ratio_is,3))+"\t"+str(np.round(mean_nuc,3))+"\t"+str(np.round(mean_cyto,3))+"\t"+str(np.round(rmean_pos,3))+"\t"+str(np.round(cmean_pos,3))+"\t"+str(size_cell)+"\n")
                        #""""""""""""""""""""""""""""""""""
                            
                        #uncomment to plot individual nuclei, red higher jet reflects pos
                        
                        #ax[1].imshow(extract_cell_of_int,cmap='gray')
                        #ax[1].imshow(erode_nuc, cmap='jet', alpha=0.5)
                            
                            
                        #ax[2].imshow(extract_cell_of_int,cmap='gray')
                        #ax[2].imshow(mask_rim, cmap='jet', alpha=0.5)
                        #ax[2].scatter(c_cyto_s2,r_cyto_s2,c='y',alpha=0.1)
                        #ax[1].set_title('ratios='+str(np.round(ratio_is,3)))
                        #ax[0].set_title('intensity='+str(np.round(mean_intensity_cell,3)))
                            
                        #plt.show()
                        #plt.savefig(base_path_files+'output/'+'nuclei_masks_'+str(fish_num)+'_'+str(files)+'_'+str(lab_v)+'.png')
                        #"""""""""""""""""""""""""""""""""""
                np.savetxt(file_to_write_map,masks_copy,fmt='%1.3f',delimiter='\t',newline='\n')
                fid1.flush()
                fid1.close()
            
   
