import os
import sys
import pandas as pd
import numpy as np
from pydub import AudioSegment
import scipy
from IPython.display import Audio
import soundfile as sf
import librosa
import IPython.display as ipd 
#import librosa.display
import shutil
from pydub.utils import make_chunks
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
import pickle
import random
import noisereduce as nr
import acoustic

def remove_ds(lst):
    if ".DS_Store" in lst:
        lst.remove('.DS_Store')


"""
function to get the average dBFS across the whole data

data_path: the path to the data

"""
def get_avg_dBFS(data_path):
    avg_dBFS = 0
    count_samp = 0
    for root, _, _ in os.walk(data_path):
        #get all the folder_path of the data
        if root.split("\\")[-1].replace("-", "").isdigit():
            #loop through every sound files inside a folder_path
            for sound in os.listdir(root):
                sound_path = os.path.join(root, sound)
                sound = AudioSegment.from_file(sound_path, format = "mp3")
                get_dBFS = sound.dBFS
                avg_dBFS += get_dBFS
                count_samp += 1

    avg_dBFS = avg_dBFS / count_samp
    return avg_dBFS


"""
data_path = path to the data.
    ex: '/Users/hongtingyang/Desktop/切割病人音檔/alzheimer/processed_data'
save_path = path to save the data
    ex: '/Users/hongtingyang/Desktop/切割病人音檔/alzheimer/saving'
avg_dbFS_dict = the calculated avg_dbFS of the entire dataset
out: normalized sound files. Followed by the structure illustrated below:
save_path
    Control
            folders
                    data
    Dementia
            folders
                    data
"""
def normalize_data_single(data_path, save_path, avg_dBFS, is_MCI):

    ##Check if save_path exists. If yes, delete the path and re-create it again
    if os.path.exists(save_path):
            shutil.rmtree(save_path)

    if not os.path.exists(save_path):
        os.mkdir(save_path)


    if is_MCI:
        groups = ["Control", "Dementia", "MCI"]
    else:
        groups = ["Control", "Dementia"]

    #get the paths for every sub_folder inside BOTH groups
    data_lst = [i[0] for i in os.walk(data_path) if i[0].split("\\")[-1][-1].isdigit()]
    #process group independently
    for group in groups:
        group_dir = os.path.join(save_path, group)
        os.mkdir(group_dir)
        
        #accesss only a particular group's data
        group_data = [i for i in data_lst if group in i]
        
        #loop through each folder_path
        for folder_path in group_data:
            dp_dir = data_path.split("\\")[-1]
            save_dir = save_path.split("\\")[-1]
            sub_dir = folder_path.replace(dp_dir, save_dir)
            if os.path.exists(sub_dir):
                continue
            else:
                os.mkdir(sub_dir)   
            sounds = os.listdir(folder_path)
            sounds = sorted(sounds, key = lambda x: int(x.split(".")[0].replace("-", "")))
            
            #loop through each sound file inside a particular folder_path
            for sound in sounds:
                sound_path = os.path.join(folder_path, sound)
                s_sound, sr = librosa.load(sound_path)
                s_sound = s_sound / avg_dBFS
                save_name = os.path.join(sub_dir, folder_path.split("\\")[-1] + "_" + sound)
                sf.write(save_name, s_sound, sr)                
"""
data_path: path to the folder that has the normalized data
    ex: 'Users/hongtingyang/Desktop/切割病人音檔/alzheimer/normalized_data'
save_path = path that you wish to store the data in

out: normalized sound files. Followed by the structure illustrated below:
save_path
    Control
            folders
                    data
    Dementia
            folders
                    data
"""
def silence_fade(data_path, save_path, fade_time = 15, silence_time = 10):
    
    #create folder, if already exist, delete it and create again
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
        
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        
    groups = ["Control", "Dementia"]
    
    #get the path to the group for getting data and make group-sub folders to store.
    for group in groups:
        group_save_path = os.path.join(save_path, group)
        os.mkdir(group_save_path)
        
        group_data_path = os.path.join(data_path, group)

        folders = os.listdir(group_data_path)
        remove_ds(folders)
        folders = sorted(folders, key = lambda x: int(x.replace("-", "")))

        #get the folder from the group_data_path and make sub folders to store
        for folder in folders:
            folder_path = os.path.join(group_data_path, folder)
            sounds = os.listdir(folder_path)
            remove_ds(sounds)
            subdir = os.path.join(group_save_path, folder)
            os.mkdir(subdir)
            
            print("Adding fades/silences in the {} group audio folder of {}".format(group, folder))

            #for every sound inside the folder from group_data_path and add fade-in/fade-out and silence to the sound
            #then store it inside the sub-folders
            for sound in sounds:
                sound_path = os.path.join(folder_path, sound)
                x = AudioSegment.from_file(sound_path, format = "wav")
                x_fade = x.fade_in(fade_time).fade_out(fade_time)
                silence_segment =  AudioSegment.silent(duration=silence_time)
                x_fade_silence = silence_segment + x_fade + silence_segment
                x_fade_silence.export(os.path.join(subdir, sound), format = "wav")

'''
data_path: path to the folder that you want to combine audio
    ex: 'Users/hongtingyang/Desktop/切割病人音檔/alzheimer/data/normalized_data'
save_path = path that you wish to store the data in

out: normalized sound files. Followed by the structure illustrated below:
save_path
    Control
            folders
                    data
    Dementia
            folders
                    data
'''
def combine_audio(data_path, save_path, is_MCI):
    
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
        
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    if is_MCI:
        groups = ["Control", "Dementia", "MCI"]
    else:
        groups = ["Control", "Dementia"]
    
    for group in groups:
        group_data_path = os.path.join(data_path, group)
        group_save_path = os.path.join(save_path, group)
        os.mkdir(group_save_path)
        
        folders = os.listdir(group_data_path)
        #folders = [i for i in folders if i.startswith("0")]
        folders = [i for i in folders if i[-1].isdigit()]
        folders = sorted(folders, key = lambda x: int(x.replace("-", "")))
        
        for folder in folders:
            folder_path = os.path.join(group_data_path, folder)
            sounds = os.listdir(folder_path)
            sounds = [i for i in sounds if i.endswith(".wav")]
            sounds = sorted(sounds, key = lambda x: int(x.split(".")[0].replace("-","").replace("_","")))
            subdir = os.path.join(group_save_path, folder)
            os.mkdir(subdir)
            print("subdir:", subdir)
            whole_sound = []
            for sound in sounds:
                sound_path = os.path.join(folder_path, sound)
                x = AudioSegment.from_wav(sound_path)
                whole_sound.append(x)
            complete_sound = sum(whole_sound)
            #print("save_path:", os.path.join(subdir, folder + ".wav"))
            complete_sound.export(os.path.join(subdir, folder + ".wav"), format = 'wav')  
            
        print("finished")
            
'''
save_dir: The path of the new folder to store the data
data_path: the path to the data
segment_length: the length of the segmented audio in miliseconds
default_silence: the short silence segment added to the beginning and the end of the audio file
out: segmented sound files. Followed by the structure illustrated below:
folder_name
    Control
            folders
                    data
    Dementia
            folders
                    data
'''
def audio_segment(save_dir, data_path, segment_length = 4000, default_silence = 0):
    
    folder_name = save_dir.split("\\")[-1]
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for root, dirs, files in os.walk(data_path):
        for name in dirs:
            file_path = os.path.join(root, name)
            dir_name = data_path.split("\\")[-1]
            new_file_path = file_path.replace(dir_name, folder_name)
            if not os.path.exists(new_file_path):
                os.mkdir(new_file_path)
            if (new_file_path.split('\\')[-1] == "Dementia") or (new_file_path.split('\\')[-1] == "Control") or (new_file_path.split('\\')[-1] == "MCI"):
                continue
            else:
                folder_prefix = file_path.split("\\")[-1]
                for sub_root, _, sub_files in os.walk(file_path):
                    for file in sub_files:
                        if file == ".DS_Store":
                            continue
                        audio_path = os.path.join(sub_root, file)
                        audio = AudioSegment.from_file(audio_path , "wav") 
                        chunks = make_chunks(audio, segment_length)
                        for i, chunk in enumerate(chunks):
                            chunk_name = '{}_{}'.format(folder_prefix, i) + ".wav"
                            silence_segment =  AudioSegment.silent(duration=default_silence)
                            chunk = silence_segment + chunk + silence_segment
                            if chunk.duration_seconds * 1000 < segment_length:
                                silence_time = segment_length - chunk.duration_seconds * 1000
                                silence_segment =  AudioSegment.silent(duration=silence_time)
                                chunk = chunk + silence_segment
                            print("exporting {}".format(chunk_name))
                            chunk.export(os.path.join(new_file_path, chunk_name), format = "wav")

                            
"""
save_dir: The path of the new folder to store the data
data_path: the path to the data
out: the clean sounds in the following format:

folder_name
    Control
            folders
                    data
    Dementia
            folders
                    data
                    
This function filter out the background noise of the sound file 
"""
def denoise(save_dir, data_path):
    
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    folder_lst = []
    for folder, subfolders, files in os.walk(data_path):
        folder_lst.append(folder)

    ct = [i for i in folder_lst if "Control" in i]
    ad = [i for i in folder_lst if "Dementia" in i]
    mci = [i for i in folder_lst if "MCI" in i]

    data_lst = [mci, ct, ad]

    root_name = data_path.split("\\")[-1]
    save_name = save_dir.split("\\")[-1]

    for lst in data_lst:
        group_dir = lst[0].split("\\")[-1]
        working = lst[1:]
        group_dir = os.path.join(save_dir, group_dir)
        os.mkdir(group_dir)

        for folder_path in working:

            sub_folder = os.path.join(group_dir, folder_path.split("\\")[-1])
            os.mkdir(sub_folder)

            files = os.listdir(folder_path)

            for file in files:
                file_path = os.path.join(folder_path, file)
                print("processing: ", file_path)
                data, rate = librosa.load(file_path)
                noisy_part = data
                # perform noise reduction
                reduced_noise = nr.reduce_noise(audio_clip=data, noise_clip=noisy_part,n_std_thresh=0.8,use_tensorflow=True, verbose=False)
                out_f = os.path.join(sub_folder, file)
                print("saving file:", out_f)
                sf.write(out_f, reduced_noise,rate)   

"""
Function to get the MFCC features from the audio and exported it as csv file

save_dir: path to save the exported data
data_path: path to the audio data

out: 
folder_name
    Control_folder
        .csv
    Dementia_folder
        .csv
    MCI_folder
        .csv            
"""
def get_MFCC(save_dir, data_path):

    groups = {"Control":0, "Dementia":1, "MCI":2}

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for group in groups.keys():
        group_path = os.path.join(data_path, group)
        save_group_path =  os.path.join(save_dir, group)

        sub_dirs = os.listdir(group_path)

        results = []
        for sub_dir in sub_dirs:
            dir_path = os.path.join(group_path, sub_dir)
            data = acoustic.get_all(dir_path)
            results.append(pd.DataFrame.from_dict(data, orient="index"))

        os.mkdir(save_group_path)

        df = pd.concat(results)
        df.reset_index(inplace=True)
        df = df.rename(columns = {'index':'Name'})
        df['label'] = groups[group]
        df = df.reset_index(drop=True)

        csv_path = os.path.join(save_group_path, "{}.csv".format(group))
        df.to_csv(csv_path, index = False)
        
'''
sound: a pydub.AudioSegment
silence_threshold: threshold in dBs to detect silence
chunk_size: the length of the audio to be checked at a time (in milliseconds)
'''
def filter_silence(sound, silence_threshold=-50.0, chunk_size=10):
    
    
    trim_ms_leading = 0 # ms
    trim_ms_ending = 0
    
    sound_reverse = sound.reverse()
    assert chunk_size > 0 # to avoid infinite loop
    while sound[trim_ms_leading:trim_ms_leading+chunk_size].dBFS < silence_threshold and trim_ms_leading < len(sound):
        trim_ms_leading += chunk_size
    
    while sound_reverse[trim_ms_ending:trim_ms_ending+chunk_size].dBFS < silence_threshold and trim_ms_ending < len(sound_reverse):
        trim_ms_ending += chunk_size
    
    duration = len(sound)    
    trimmed_sound = sound[trim_ms_leading:duration-trim_ms_ending]
        
    return trimmed_sound

"""
Function that apply the filter_silence function to every audio in the data_path
data_path: the path to the audio data
save_dir: the path to save the filtered audio
out:
    folder_name
        Control
            patient_folder
                .wav files
                ...
            ...
        Dementia
            patient_folder
                .wav files
                ...
            ...
        MCI
            patient_filder
                .wav fles
                ...
            ...
"""
def apply_filter_silence(save_dir, data_path, min_length = 1):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for root, dirs, files in os.walk(data_path):
        folder_name = save_dir.split("\\")[-1]
        for name in dirs:
            file_path = os.path.join(root, name)
            dir_name = data_path.split("\\")[-1]
            new_file_path = file_path.replace(dir_name, folder_name)
            if not os.path.exists(new_file_path):
                os.mkdir(new_file_path)

            to_check = new_file_path.split('\\')[-1]
            if (to_check == "Dementia") or (to_check == "Control") or (to_check == "MCI"):
                continue

            else:
                folder_prefix = file_path.split("\\")[-1]
                for sub_root, _, sub_files in os.walk(file_path):
                    for file in sub_files:
                        if file == ".DS_Store":
                            continue
                        audio_path = os.path.join(sub_root, file)
                        audio = AudioSegment.from_file(audio_path , "wav")
                        audio = filter_silence(audio)
                        if audio.duration_seconds <= min_length:
                            print("The audio:{} is either too short or include silent sound".format(audio_path))
                            continue
                        else:
                            print("exporting file:{}".format(file))
                            audio.export(os.path.join(new_file_path, file), format = "wav")
"""
Function that takes in the segmented audio directory and randomly mix the audios of a particular group with a specific time duration. The sampling will first sample the "folder" or the participant and then goes in that folder to sample an audio file. This ensures uniform distribution across all participant

data_path: The segmented audio directory
save_dir: path to save the mixed audio
num_sample: The number of mixed sample to be generated for a given group
duration: The length of the audio file
"""
def mix_sounds(data_path, save_dir, num_sample, duration):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    grps = ["Control", "Dementia"]

    for grp in grps:
        group_path = os.path.join(data_path, grp)
        folders = os.listdir(group_path)

        save_group_dir = os.path.join(save_dir, grp)
        os.mkdir(save_group_dir)

        for i in range(num_sample):
            mixed_audio = AudioSegment.empty()
            flag = True
            while flag:
                sample_folder = random.sample(folders, 1)[0]
                sounds_path = os.path.join(group_path, sample_folder)
                sounds = os.listdir(sounds_path)

                random.shuffle(sounds)
                sample_sound = random.sample(sounds, 1)[0]
                sample_sound_path = os.path.join(sounds_path, sample_sound)
                audio = AudioSegment.from_file(sample_sound_path , "wav")

                mixed_audio += audio

                if mixed_audio.duration_seconds >= duration:
                    sound_name = os.path.join(save_group_dir, "mixed-{}.wav".format(i+1))
                    mixed_audio.export(sound_name, format="wav")
                    flag = False
                
                            
"""
In: IS09_path
out: a list of lists that contain control and dementia participants' features
"""
def get_features_IS09(IS09_path, mixed = False):
    groups = ["Control", "Dementia"]
    data_lst = []
    for group in groups:
        group_path = os.path.join(IS09_path, group)
        store = []
        
        if mixed:
            for csv in os.listdir(group_path):
                csv_path = os.path.join(group_path, csv)
                feature = pd.read_csv(csv_path, skiprows = lambda x: x in range(0, 390),
                                          header = None).iloc[0, 1:-1].values
                feature = feature.astype(float)
                store.append(feature)
        else:
            for folder in os.listdir(group_path):
                folder_path = os.path.join(group_path, folder)
                if ".DS_Store" in folder_path:
                    continue

                for csv in os.listdir(folder_path):
                    csv_path = os.path.join(folder_path, csv)
                    feature = pd.read_csv(csv_path, skiprows = lambda x: x in range(0, 390)
                                     , header = None).iloc[0, 1:-1].values
                    feature = feature.astype(float)
                    store.append(feature)

        data_lst.append(store)
    
    return data_lst

def get_features_IS10(IS10_path, mixed = False):
    groups = ["Control", "Dementia"]
    data_lst = []
    for group in groups:
        group_path = os.path.join(IS10_path, group)
        store = []
        
        if mixed:
            for csv in os.listdir(group_path):
                csv_path = os.path.join(group_path, csv)
                feature = pd.read_csv(csv_path, skiprows = lambda x: x in range(0, 1589),
                                          header = None).iloc[0, 1:-1].values
                feature = feature.astype(float)
                store.append(feature)
        else:
            for folder in os.listdir(group_path):
                folder_path = os.path.join(group_path, folder)
                if ".DS_Store" in folder_path:
                    continue

                for csv in os.listdir(folder_path):
                    csv_path = os.path.join(folder_path, csv)
                    feature = pd.read_csv(csv_path, skiprows = lambda x: x in range(0, 1589),
                                          header = None).iloc[0, 1:-1].values
                    feature = feature.astype(float)
                    store.append(feature)
       
        data_lst.append(store)
   
    return data_lst

def get_features_IS10_person(IS10_path):
    groups = ["Control", "Dementia"]
    data_lst = []
    for group in groups:
        group_path = os.path.join(IS10_path, group)
        temp = []
       
        for folder in os.listdir(group_path):
            folder_path = os.path.join(group_path, folder)
            if ".DS_Store" in folder_path:
                continue
                
            store = []
            
            for csv in os.listdir(folder_path):
                csv_path = os.path.join(folder_path, csv)
                feature = pd.read_csv(csv_path, skiprows = lambda x: x in range(0, 1589),
                                      header = None).iloc[0, 1:-1].values
                feature = feature.astype(float)
                store.append(feature)
                
            temp.append(store)
        data_lst.append(temp)
   
    return data_lst

def get_features_IS11(IS11_path, mixed = False):
    groups = ["Control", "Dementia"]
    data_lst = []
    for group in groups:
        group_path = os.path.join(IS11_path, group)
        store = []
        
        if mixed:
            for csv in os.listdir(group_path):
                csv_path = os.path.join(group_path, csv)
                feature = pd.read_csv(csv_path, skiprows = lambda x: x in range(0, 4375),
                                          header = None).iloc[0, 1:-1].values
                feature = feature.astype(float)
                store.append(feature)
        else: 
            for folder in os.listdir(group_path):
                folder_path = os.path.join(group_path, folder)
                if ".DS_Store" in folder_path:
                    continue

                for csv in os.listdir(folder_path):
                    csv_path = os.path.join(folder_path, csv)
                    feature = pd.read_csv(csv_path, skiprows = lambda x: x in range(0, 4375),
                                          header = None).iloc[0, 1:-1].values
                    feature = feature.astype(float)
                    store.append(feature)

        data_lst.append(store)
   
    return data_lst

def get_features_IS12(IS12_path, mixed = False):
    groups = ["Control", "Dementia"]
    data_lst = []
    
    
    for group in groups:
        group_path = os.path.join(IS12_path, group)
        store = []
        
        if mixed:
            for csv in os.listdir(group_path):
                csv_path = os.path.join(group_path, csv)
                feature = pd.read_csv(csv_path, skiprows = lambda x: x in range(0, 6132),
                                      header = None).iloc[0, 1:-1].values
                feature = feature.astype(float)
                store.append(feature)
                
        else: 
            for folder in os.listdir(group_path):
                folder_path = os.path.join(group_path, folder)
                if ".DS_Store" in folder_path:
                    continue

                for csv in os.listdir(folder_path):
                    csv_path = os.path.join(folder_path, csv)
                    feature = pd.read_csv(csv_path, skiprows = lambda x: x in range(0, 6132),
                                          header = None).iloc[0, 1:-1].values
                    feature = feature.astype(float)
                    store.append(feature)
       
        data_lst.append(store)
   
    return data_lst

def get_features_IS13(IS13_path, mixed = False):
    groups = ["Control", "Dementia"]
    data_lst = []
    for group in groups:
        group_path = os.path.join(IS13_path, group)
        store = []
        
        if mixed:
            for csv in os.listdir(group_path):
                csv_path = os.path.join(group_path, csv)
                feature = pd.read_csv(csv_path, skiprows = lambda x: x in range(0, 6380),
                                          header = None).iloc[0, 1:-1].values
                feature = feature.astype(float)
                store.append(feature)
        else:
            for folder in os.listdir(group_path):
                folder_path = os.path.join(group_path, folder)
                if ".DS_Store" in folder_path:
                    continue

                for csv in os.listdir(folder_path):
                    csv_path = os.path.join(folder_path, csv)
                    feature = pd.read_csv(csv_path, skiprows = lambda x: x in range(0, 6380),
                                          header = None).iloc[0, 1:-1].values
                    feature = feature.astype(float)
                    store.append(feature)
       
        data_lst.append(store)
   
    return data_lst

def get_features_emoLarge(emoLarge_path, mixed = False):
    groups = ["Control", "Dementia", "MCI"]
    data_lst = []
    for group in groups:
        group_path = os.path.join(emoLarge_path, group)
        store = []
        
        if mixed:
            for csv in os.listdir(group_path):
                csv_path = os.path.join(group_path, csv)
                feature = pd.read_csv(csv_path, skiprows = lambda x: x in range(0, 6560),
                                          header = None).iloc[0, 2:-1].values
                feature = feature.astype(float)
                store.append(feature)
        else:
            for folder in os.listdir(group_path):
                folder_path = os.path.join(group_path, folder)
                if ".DS_Store" in folder_path:
                    continue

                for csv in os.listdir(folder_path):
                    csv_path = os.path.join(folder_path, csv)
                    feature = pd.read_csv(csv_path, skiprows = lambda x: x in range(0, 6560),
                                          header = None).iloc[0, 2:-1].values
                    feature = feature.astype(float)
                    store.append(feature)
       
        data_lst.append(store)
   
    return data_lst

def get_features_emoLarge_person(emoLarge_path, mixed = False):
    groups = ["Control", "Dementia", "MCI"]
    data_lst = []
    all_code = []
    for group in groups:
        group_path = os.path.join(emoLarge_path, group)
        store = []
        code_name = []
        if mixed:
            for csv in os.listdir(group_path):
                csv_path = os.path.join(group_path, csv)
                feature = pd.read_csv(csv_path, skiprows = lambda x: x in range(0, 6560),
                                          header = None).iloc[0, 2:-1].values
                feature = feature.astype(float)
                store.append(feature)
        else:
            for folder in os.listdir(group_path):
                folder_path = os.path.join(group_path, folder)
                code = "{}:{}".format(group, folder)
                code_name.append(code)
                print("Processing: {}".format(code))
                by_person = []
                if ".DS_Store" in folder_path:
                    continue

                for csv in os.listdir(folder_path):
                    csv_path = os.path.join(folder_path, csv)
                    feature = pd.read_csv(csv_path, skiprows = lambda x: x in range(0, 6560),
                                          header = None).iloc[0, 2:-1].values
                    feature = feature.astype(float)
                    by_person.append(feature)
                    
                store.append(by_person)
        
        all_code.append(code_name)    
        data_lst.append(store)
   
    return data_lst, all_code

"""
function to re-engineer features using mean, median, and standard deviation. Apply works for the data from get_features_ISxx_person!!!

ct: list of control group
dm: list of dementia group

out: the re-engineered features.

consider data shape (60000, 36), transform to --> (1, 36, 3). Typically, just flatten it to get the newly engineered features
"""
def get_person_stats(ct, dm):
    
    ct_lst = []
    for i in range(len(ct)):
        a_list = []
        data = np.array(ct[i])
        _, num_feat = data.shape

        for j in range(num_feat):
            mean = np.mean(data[:, j])
            std = np.std(data[:, j])
            median = np.median(data[:, j])
            info = [mean, std, median]
            a_list.append(info)
        a_list = np.array(a_list).flatten()
        ct_lst.append(a_list)


    dm_lst = []
    for i in range(len(dm)):
        a_list = []
        data = np.array(dm[i])
        _, num_feat = data.shape

        for j in range(num_feat):
            mean = np.mean(data[:, j])
            std = np.std(data[:, j])
            median = np.median(data[:, j])
            info = [mean, std, median]
            a_list.append(info)
        a_list = np.array(a_list).flatten()
        dm_lst.append(a_list)
    
    return ct_lst, dm_lst


def feature_select(X,y, save_path):
    sel_ = SelectFromModel(LogisticRegression(C=1, penalty='l1', solver = "liblinear"))
    sel_.fit(X, y)
    feat_bool = sel_.get_support()
    pd.DataFrame(feat_bool).to_csv(save_path, index = False)
    print("feature selecture finished, saved at: {}".format(save_path))
    return feat_bool

"""
model: trained model
save_path: the path to save the model. Make sure the model is saved as ".sav" file type
"""
def save_ML(model, save_path):
        pickle.dump(model, open(save_path, 'wb'))
#loaded_model = pickle.load(open(save_path, 'rb'))


"""
model: trained model
save_path: the path to load the model
"""
def load_ML(save_path):
    loaded_model = pickle.load(open(save_path, 'rb'))
    return loaded_model
    
