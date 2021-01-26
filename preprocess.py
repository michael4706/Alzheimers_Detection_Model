import os
import numpy as np
import pandas as pd
from util import *

#please change your own root_path, segmented_path, data_path, and folder_to_save argument

if __name__ == '__main__':
    
    #data_path = "\\Users\\459165\\Desktop\\Alzheimer\\denoise_proj\\data\\HC-DM-MC_train"
    #save_path = '\\Users\\459165\\Desktop\\Alzheimer\\denoise_proj\\data\\train_full'
    
    #combine_audio(data_path, save_path, is_MCI = True)
    
    #data_path = save_path
    #save_path = '\\Users\\459165\\Desktop\\Alzheimer\\denoise_proj\\data\\train_full_denoised'
    #denoise(save_path, data_path)
    
    #data_path =  '\\Users\\459165\\Desktop\\Alzheimer\\denoise_proj\\data\\train_full_denoised'
    #save_path = '\\Users\\459165\\Desktop\\Alzheimer\\denoise_proj\\data\\train_4s_denoised'
    
    #audio_segment(save_path, data_path)
    #data_path = '\\Users\\459165\\Desktop\\Alzheimer\\denoise_proj\\data\\train_4s_denoised'
    #save_path = '\\Users\\459165\\Desktop\\Alzheimer\\denoise_proj\\data\\train_4s_denoised_silenceFilter'
    
    #apply_filter_silence(save_path, data_path)
    
    #data_path = save_path
    #save_path = '\\Users\\459165\\Desktop\\Alzheimer\\denoise_proj\\data\\train_full_denoised_silenceFilter'
    #combine_audio(data_path, save_path, is_MCI = True)
    
    data_path = '\\Users\\459165\\Desktop\\Alzheimer\\denoise_proj\\data\\test_4s_denoised_silenceFilter'
    save_path = '\\Users\\459165\\Desktop\\Alzheimer\\denoise_proj\\MFCC\\test_4s_denoised_silenceFilter'
    
    get_MFCC(save_path, data_path)


    #############################################################################################
    #data_path = "\\Users\\459165\\Desktop\\Alzheimer\\denoise_proj\\data\\HC-DM-MC_test"
    #save_path = '\\Users\\459165\\Desktop\\Alzheimer\\denoise_proj\\data\\test_full'
    
    #combine_audio(data_path, save_path, is_MCI = True)
    
    #data_path = save_path
    #save_path = '\\Users\\459165\\Desktop\\Alzheimer\\denoise_proj\\data\\test_full_denoised'
    #denoise(save_path, data_path)
    
    #data_path =  '\\Users\\459165\\Desktop\\Alzheimer\\denoise_proj\\data\\test_full_denoised'
    #save_path = '\\Users\\459165\\Desktop\\Alzheimer\\denoise_proj\\data\\test_4s_denoised'
    
    #audio_segment(save_path, data_path)
    
    #data_path = '\\Users\\459165\\Desktop\\Alzheimer\\denoise_proj\\data\\test_4s_denoised'
    #save_path = '\\Users\\459165\\Desktop\\Alzheimer\\denoise_proj\\data\\test_4s_denoised_silenceFilter'
    
    #apply_filter_silence(save_path, data_path)
    
    #data_path = save_path
    #save_path = '\\Users\\459165\\Desktop\\Alzheimer\\denoise_proj\\data\\test_full_denoised_silenceFilter'
    #combine_audio(data_path, save_path, is_MCI = True)