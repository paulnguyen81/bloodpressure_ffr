import pickle
import numpy as np
import pandas as pd
import time
import os
from feature_utils import *

def picklereader(fpath):
    with open(fpath,'rb') as f:
        content = pd.read_pickle(f)
    return content

def scale(sx):
    """
    scalining
    """
    if sx==480:
        return (50**2)*(256/480)**2
    elif sx == 512:
        return (56.5**2)*(256/512)**2

if __name__=='__main__':
    
    t0 = time.time()
    df = picklereader(os.getcwd()+'/data/df18S_rs.p')
    df['luA'] = df['lu'].apply(lambda x: np.array(x))/df['sx0'].apply(lambda x: scale(x)) #convert lumen pixel counts to sqmm
    df['plA'] = df['pl'].apply(lambda x: np.array(x))/df['sx0'].apply(lambda x: scale(x)) #convert plaque pixel counts to sqmm
    #df['pb'] = [plaqueburden(df.ix[i].luA,df.ix[i].plA) for i in df.index] #plaqueburden by frame
    #df['f_1'] = roi(df,'pb').apply(lambda x:len_pb(x,0.4)) #feature 1
    #df['f_2'] = roi(df,'pb').apply(lambda x:len_pb(x,0.7)) #feature 2
    
    #constant variables
    gap = 300       #combines lesions if number of frames between is less than
    pbth = 0.4      #plaque burden threshold in lesion

    ## SMOOTHING
                                       #runtimewarning
    df['luAc'] = df['luA'].apply(lambda x: np.convolve(x,np.ones((50,))/50,mode='same'))                    #smooth luA
    df['plAc'] = df['plA'].apply(lambda x: np.convolve(x,np.ones((50,))/50,mode='same'))                    #smooth plA
    df['pb'] = df.apply(lambda x: list(plaqueburden(x.luAc,x.plAc)),axis=1) 
    t1 = time.time()
    total = t1-t0
    print('Smoothing - Execution time: ',total,"secs")
    
    ## FEATURE EXTRACTION
    #feature 1-11
    t0 = time.time()
    df['len_PB40'] = df.apply(lambda x: lesion(x.luAc,x.plAc,x.distal,x.OS,pbth,gap),axis=1)
    df['len_PB70'] = df.apply(lambda x: lesion(x.luAc,x.plAc,x.distal,x.OS,0.7,gap),axis=1)
    df['OS_PB40'] = df.apply(lambda x: fromos_pb(x.luAc,x.plAc,x.distal,x.OS,0.4,gap),axis=1)
    df['OS_PB70'] = df.apply(lambda x: fromos_pb(x.luAc,x.plAc,x.distal,x.OS,0.7,gap),axis=1)              
    df['OS_MLA'] = df.apply(lambda x: fromos_mla(x.luAc,x.distal,x.OS),axis=1)                             
    df['MLA'] = df.apply(lambda x: mla(x.luAc,x.distal,x.OS),axis=1)
    df['EEM_MLA'] = df.apply(lambda x: eem_mla(x.luAc,x.plAc,x.distal,x.OS),axis=1)
    df['PB_MLA'] = df.apply(lambda x: pb_mla(x.luAc,x.plAc,x.distal,x.OS),axis=1)
    df['No_PB40'] = df.apply(lambda x: numberofpb(x.pb,pbth,x.distal,x.OS),axis=1)                           #runtimewarning
    df['No_PB70'] = df.apply(lambda x: numberofpb(x.pb,0.7,x.distal,x.OS),axis=1)                           #runtimewarning
    t1 = time.time()
    total = t1-t0
    print('Feature 1-11 - Execution time: ',total,"secs")
    
    #feature 12-20
    t0 = time.time()
    df['No_lumen40_ROI'] = df.apply(lambda x: no_lumen_roi(x.luAc,x.plAc,x.distal,x.OS,4.0,0),axis=1)
    df['No_lumen25_ROI'] = df.apply(lambda x: no_lumen_roi(x.luAc,x.plAc,x.distal,x.OS,2.5,0),axis=1)
    df['No_lumen30_ROI'] = df.apply(lambda x: no_lumen_roi(x.luAc,x.plAc,x.distal,x.OS,3.0,0),axis=1)
    df['Sum_plaque_ROI'] = df.apply(lambda x: sum_plaque_roi(x.luAc,x.plAc,x.distal,x.OS,0),axis=1)
    df['Sum_EEM_ROI'] = df.apply(lambda x: sum_eem_roi(x.luAc,x.plAc,x.distal,x.OS,0),axis=1)
    df['PB_ROI'] = df.apply(lambda x: pb_roi(x.luAc,x.plAc,x.distal,x.OS,0),axis=1)
    df['mean_lumen_ROI'] = df.apply(lambda x: mean_lumen_roi(x.luAc,x.plAc,x.distal,x.OS,0),axis=1)
    df['mean_plaque_ROI'] = df.apply(lambda x: mean_plaque_roi(x.luAc,x.plAc,x.distal,x.OS,0),axis=1)
    df['mean_EEM_ROI'] = df.apply(lambda x: mean_eem_roi(x.luAc,x.plAc,x.distal,x.OS,0),axis=1)
    t1 = time.time()
    total = t1-t0
    print('Feature 12-20 - Execution time: ',total,"secs")
        
    #feature 21-29
    t0 = time.time()
    df['No_lumen40_PB40'] = df.apply(lambda x: no_lumen_roi(x.luAc,x.plAc,x.distal,x.OS,4.0,pbth),axis=1)
    df['No_lumen25_PB40'] = df.apply(lambda x: no_lumen_roi(x.luAc,x.plAc,x.distal,x.OS,2.5,pbth),axis=1)
    df['No_lumen30_PB40'] = df.apply(lambda x: no_lumen_roi(x.luAc,x.plAc,x.distal,x.OS,3.0,pbth),axis=1)
    df['Sum_plaque_PB40'] = df.apply(lambda x: sum_plaque_roi(x.luAc,x.plAc,x.distal,x.OS,pbth),axis=1)
    df['Sum_EEM_PB40'] = df.apply(lambda x: sum_eem_roi(x.luAc,x.plAc,x.distal,x.OS,pbth),axis=1)
    df['PB_PB40'] = df.apply(lambda x: pb_roi(x.luAc,x.plAc,x.distal,x.OS,pbth),axis=1)                     
    df['mean_lumen_PB40'] = df.apply(lambda x: mean_lumen_roi(x.luAc,x.plAc,x.distal,x.OS,pbth),axis=1)
    df['mean_plaque_PB40'] = df.apply(lambda x: mean_plaque_roi(x.luAc,x.plAc,x.distal,x.OS,pbth),axis=1)
    df['mean_EEM_PB40'] = df.apply(lambda x: mean_eem_roi(x.luAc,x.plAc,x.distal,x.OS,pbth),axis=1)
    t1 = time.time()
    total = t1-t0
    print('Feature 21-29 - Execution time: ',total,"secs")

    #feature 30-38
    t0 = time.time()
    df['No_lumen40_PB70'] = df.apply(lambda x: no_lumen_roi(x.luAc,x.plAc,x.distal,x.OS,4.0,0.7),axis=1)
    df['No_lumen25_PB70'] = df.apply(lambda x: no_lumen_roi(x.luAc,x.plAc,x.distal,x.OS,2.5,0.7),axis=1)
    df['No_lumen30_PB70'] = df.apply(lambda x: no_lumen_roi(x.luAc,x.plAc,x.distal,x.OS,3.0,0.7),axis=1)
    df['Sum_plaque_PB70'] = df.apply(lambda x: sum_plaque_roi(x.luAc,x.plAc,x.distal,x.OS,0.7),axis=1)
    df['Sum_EEM_PB70'] = df.apply(lambda x: sum_eem_roi(x.luAc,x.plAc,x.distal,x.OS,0.7),axis=1)
    df['PB_PB70'] = df.apply(lambda x: pb_roi(x.luAc,x.plAc,x.distal,x.OS,0.7),axis=1)                     
    df['mean_lumen_PB70'] = df.apply(lambda x: mean_lumen_roi(x.luAc,x.plAc,x.distal,x.OS,0.7),axis=1)
    df['mean_plaque_PB70'] = df.apply(lambda x: mean_plaque_roi(x.luAc,x.plAc,x.distal,x.OS,0.7),axis=1)
    df['mean_EEM_PB70'] = df.apply(lambda x: mean_eem_roi(x.luAc,x.plAc,x.distal,x.OS,0.7),axis=1)
    t1 = time.time()
    total = t1-t0
    print('Feature 30-38 - Execution time: ',total,"secs")

    #feature 39-47
    t0 = time.time()
    df['No_lumen40_worst'] = df.apply(lambda x: no_lumen_worst(x.luAc,x.distal,x.OS,luth=4.0),axis=1)
    df['No_lumen25_worst'] = df.apply(lambda x: no_lumen_worst(x.luAc,x.distal,x.OS,luth=2.5),axis=1)
    df['Sum_plaque_worst'] = df.apply(lambda x: sum_plaque_worst(x.luAc,x.plAc,x.distal,x.OS),axis=1)
    df['Sum_EEM_worst'] = df.apply(lambda x: sum_eem_worst(x.luAc,x.plAc,x.distal,x.OS),axis=1)
    df['PB_worst'] = df.apply(lambda x: pb_worst(x.luAc,x.plAc,x.distal,x.OS),axis=1)
    df['mean_lumen_worst'] = df.apply(lambda x: mean_lumen_worst(x.luAc,x.distal,x.OS),axis=1)
    df['mean_plaque_worst'] = df.apply(lambda x: mean_plaque_worst(x.luAc,x.plAc,x.distal,x.OS),axis=1)
    df['mean_EEM_worst'] = df.apply(lambda x: mean_eem_worst(x.luAc,x.plAc,x.distal,x.OS),axis=1)
    t1 = time.time()
    total = t1-t0
    print('Feature 39-47 - Execution time: ',total,"secs")

    #feature 49-57
    t0 = time.time()
    df['No_lumen40_prox'] = df.apply(lambda x: no_lumen_prox(x.luAc,x.plAc,x.distal,x.OS,pbth,4.0,gap),axis=1)   #all zeroes
    df['No_lumen25_prox'] = df.apply(lambda x: no_lumen_prox(x.luAc,x.plAc,x.distal,x.OS,pbth,2.5,gap),axis=1)   #all zeroes
    df['No_lumen30_prox'] = df.apply(lambda x: no_lumen_prox(x.luAc,x.plAc,x.distal,x.OS,pbth,3.0,gap),axis=1)   #all zeroes
    df['Sum_plaque_prox'] = df.apply(lambda x: sum_plaque_prox(x.luAc,x.plAc,x.distal,x.OS,pbth,gap),axis=1)
    df['Sum_EEM_prox'] = df.apply(lambda x: sum_eem_prox(x.luAc,x.plAc,x.distal,x.OS,pbth,gap),axis=1)
    df['PB_prox'] = df.apply(lambda x: pb_prox(x.luAc,x.plAc,x.distal,x.OS,pbth,gap),axis=1)
    df['mean_lumen_prox'] = df.apply(lambda x: mean_lumen_prox(x.luAc,x.plAc,x.distal,x.OS,pbth,gap),axis=1)
    df['mean_plaque_prox'] = df.apply(lambda x: mean_plaque_prox(x.luAc,x.plAc,x.distal,x.OS,pbth,gap),axis=1)
    df['mean_EEM_prox'] = df.apply(lambda x: mean_eem_prox(x.luAc,x.plAc,x.distal,x.OS,pbth,gap),axis=1)
    df['max_EEM_prox'] = df.apply(lambda x: max_eem_prox(x.luAc,x.plAc,x.distal,x.OS,pbth,gap),axis=1)
    t1 = time.time()
    total = t1-t0
    print('Feature 49-57 - Execution time: ',total,"secs")

    #feature 58-67
    t0 = time.time()
    df['No_lumen40_distal'] = df.apply(lambda x: no_lumen_distal(x.luAc,x.plAc,x.distal,x.OS,pbth,4.0,gap),axis=1)
    df['No_lumen25_distal'] = df.apply(lambda x: no_lumen_distal(x.luAc,x.plAc,x.distal,x.OS,pbth,2.5,gap),axis=1)   #all zeroes
    df['No_lumen30_distal'] = df.apply(lambda x: no_lumen_distal(x.luAc,x.plAc,x.distal,x.OS,pbth,3.0,gap),axis=1)   #all zeroes
    df['Sum_plaque_distal'] = df.apply(lambda x: sum_plaque_distal(x.luAc,x.plAc,x.distal,x.OS,pbth,gap),axis=1)
    df['Sum_EEM_distal'] = df.apply(lambda x: sum_eem_distal(x.luAc,x.plAc,x.distal,x.OS,pbth,gap),axis=1)
    df['PB_distal'] = df.apply(lambda x: pb_distal(x.luAc,x.plAc,x.distal,x.OS,0.4,gap),axis=1)
    df['mean_lumen_distal'] = df.apply(lambda x: mean_lumen_distal(x.luAc,x.plAc,x.distal,x.OS,pbth,gap),axis=1)
    df['mean_plaque_distal'] = df.apply(lambda x: mean_plaque_distal(x.luAc,x.plAc,x.distal,x.OS,pbth,gap),axis=1)
    df['mean_EEM_distal'] = df.apply(lambda x: mean_eem_distal(x.luAc,x.plAc,x.distal,x.OS,pbth,gap),axis=1)
    df['max_EEM_distal'] = df.apply(lambda x: max_eem_distal(x.luAc,x.plAc,x.distal,x.OS,pbth,gap),axis=1)
    t1 = time.time()
    total = t1-t0
    print('Feature 58-67 - Execution time: ',total,"secs")
    
    
    #feature 68-77
    t0 = time.time()
    df['No_lumen40_prox5'] = df.apply(lambda x: no_lumen40_prox5(x.luAc,x.plAc,x.distal,x.OS,pbth,4.0,gap),axis=1)   #all zeroes
    df['No_lumen25_prox5'] = df.apply(lambda x: no_lumen40_prox5(x.luAc,x.plAc,x.distal,x.OS,pbth,2.5,gap),axis=1)   #all zeroes
    df['No_lumen30_prox5'] = df.apply(lambda x: no_lumen40_prox5(x.luAc,x.plAc,x.distal,x.OS,pbth,3.0,gap),axis=1)   #all zeroes
    df['Sum_plaque_prox5'] = df.apply(lambda x: sum_plaque_prox5(x.luAc,x.plAc,x.distal,x.OS,pbth,gap),axis=1)
    df['Sum_EEM_prox5'] = df.apply(lambda x: sum_eem_prox5(x.luAc,x.plAc,x.distal,x.OS,pbth,gap),axis=1)
    df['PB_prox5'] = df.apply(lambda x: pb_prox5(x.luAc,x.plAc,x.distal,x.OS,pbth,gap),axis=1)
    df['mean_lumen_prox5'] = df.apply(lambda x: mean_lumen_prox5(x.luAc,x.plAc,x.distal,x.OS,pbth,gap),axis=1)
    df['mean_plaque_prox5'] = df.apply(lambda x: mean_plaque_prox5(x.luAc,x.plAc,x.distal,x.OS,pbth,gap),axis=1)
    df['mean_EEM_prox5'] = df.apply(lambda x: mean_eem_prox5(x.luAc,x.plAc,x.distal,x.OS,pbth,gap),axis=1)
    df['max_EEM_prox5'] = df.apply(lambda x: max_eem_prox5(x.luAc,x.plAc,x.distal,x.OS,pbth,gap),axis=1)
    t1 = time.time()
    total = t1-t0
    print('Feature 68-77 - Execution time: ',total,"secs")
    
    #feature 78-87
    t0 = time.time()
    df['No_lumen40_dist5'] = df.apply(lambda x: no_lumen40_dist5(x.luAc,x.plAc,x.distal,x.OS,pbth,4.0,gap),axis=1)   #all zeroes
    df['No_lumen25_dist5'] = df.apply(lambda x: no_lumen40_dist5(x.luAc,x.plAc,x.distal,x.OS,pbth,2.5,gap),axis=1)   #all zeroes
    df['No_lumen30_dist5'] = df.apply(lambda x: no_lumen40_dist5(x.luAc,x.plAc,x.distal,x.OS,pbth,3.0,gap),axis=1)   #all zeroes
    df['Sum_plaque_dist5'] = df.apply(lambda x: sum_plaque_dist5(x.luAc,x.plAc,x.distal,x.OS,pbth,gap),axis=1)
    df['Sum_EEM_dist5'] = df.apply(lambda x: sum_eem_dist5(x.luAc,x.plAc,x.distal,x.OS,pbth,gap),axis=1)
    df['PB_dist5'] = df.apply(lambda x: pb_dist5(x.luAc,x.plAc,x.distal,x.OS,pbth,gap),axis=1)
    df['mean_lumen_dist5'] = df.apply(lambda x: mean_lumen_dist5(x.luAc,x.plAc,x.distal,x.OS,pbth,gap),axis=1)
    df['mean_plaque_dist5'] = df.apply(lambda x: mean_plaque_dist5(x.luAc,x.plAc,x.distal,x.OS,pbth,gap),axis=1)
    df['mean_EEM_dist5'] = df.apply(lambda x: mean_eem_dist5(x.luAc,x.plAc,x.distal,x.OS,pbth,gap),axis=1)
    df['max_EEM_dist5'] = df.apply(lambda x: max_eem_dist5(x.luAc,x.plAc,x.distal,x.OS,pbth,gap),axis=1)
    t1 = time.time()
    total = t1-t0
    print('Feature 78-87 - Execution time: ',total,"secs")

    #feature 88-101
    t0 = time.time()
    df['mean_lumen_aver'] = df.apply(lambda x: mean_lumen_aver(x.luAc,x.plAc,x.distal,x.OS,pbth,gap),axis=1)
    df['mean_EEM_aver'] = df.apply(lambda x: mean_eem_aver(x.luAc,x.plAc,x.distal,x.OS,pbth,gap),axis=1)
    df['area1_stenosis_aver'] = df.apply(lambda x: area1_stenosis_aver(x.luAc,x.plAc,x.distal,x.OS,pbth,gap),axis=1)
    df['area1_stenosis_prox5'] = df.apply(lambda x: area1_stenosis_prox5(x.luAc,x.plAc,x.distal,x.OS,pbth,gap),axis=1)
    df['area1_stenosis_dist5'] = df.apply(lambda x: area1_stenosis_dist5(x.luAc,x.plAc,x.distal,x.OS,pbth,gap),axis=1)
    df['area2_stenosis_aver'] = df.apply(lambda x: area2_stenosis_aver(x.luAc,x.plAc,x.distal,x.OS,pbth,gap),axis=1)
    df['area2_stenosis_prox5'] = df.apply(lambda x: area2_stenosis_prox5(x.luAc,x.plAc,x.distal,x.OS,pbth,gap),axis=1)
    df['area2_stenosis_dist5'] = df.apply(lambda x: area2_stenosis_dist5(x.luAc,x.plAc,x.distal,x.OS,pbth,gap),axis=1)
    df['area3_stenosis_aver'] = df.apply(lambda x: area3_stenosis_aver(x.luAc,x.plAc,x.distal,x.OS,pbth,gap),axis=1)
    df['area3_stenosis_prox5'] = df.apply(lambda x: area3_stenosis_prox5(x.luAc,x.plAc,x.distal,x.OS,pbth,gap),axis=1)
    df['area3_stenosis_dist5'] = df.apply(lambda x: area3_stenosis_dist5(x.luAc,x.plAc,x.distal,x.OS,pbth,gap),axis=1)
    df['area4_stenosis_aver'] = df.apply(lambda x: area4_stenosis_aver(x.luAc,x.plAc,x.distal,x.OS,pbth,gap),axis=1)
    df['area4_stenosis_prox5'] = df.apply(lambda x: area4_stenosis_prox5(x.luAc,x.plAc,x.distal,x.OS,pbth,gap),axis=1)
    df['area4_stenosis_dist5'] = df.apply(lambda x: area4_stenosis_prox5(x.luAc,x.plAc,x.distal,x.OS,pbth,gap),axis=1)
    t1 = time.time()
    total = t1-t0
    print('Feature 88-101 - Execution time: ',total,"secs")

    # 102-112 Not completed

    #feature 113-122
    t0 = time.time()
    df['RI_MLA_ref'] = df.apply(lambda x: ri_mla_ref(x.luAc,x.plAc,x.distal,x.OS,pbth,gap),axis=1)
    df['RI_MLA_prox5'] = df.apply(lambda x: ri_mla_prox5(x.luAc,x.plAc,x.distal,x.OS,pbth,gap),axis=1)
    df['RI_worst_ref'] = df.apply(lambda x: ri_worst_ref(x.luAc,x.plAc,x.distal,x.OS,pbth,gap),axis=1)
    df['RI_worst_prox5'] = df.apply(lambda x: ri_worst_prox5(x.luAc,x.plAc,x.distal,x.OS,pbth,gap),axis=1)
    df['variance_lumen_worst'] = df.apply(lambda x: variance_lumen_worst(x.luAc,x.distal,x.OS),axis=1)
    df['variance_lumen_PB40'] = df.apply(lambda x: variance_lumen_pb40(x.luAc,x.plAc,x.distal,x.OS,pbth),axis=1)
    df['variance_plaque_worst'] = df.apply(lambda x: variance_plaque_worst(x.luAc,x.plAc,x.distal,x.OS),axis=1)
    df['variance_plaque_PB40'] = df.apply(lambda x: variance_plaque_pb40(x.luAc,x.plAc,x.distal,x.OS,pbth),axis=1)
    df['long_eccentricity_worst'] = df.apply(lambda x: long_eccentricity_worst(x.luAc,x.plAc,x.distal,x.OS),axis=1)
    df['long_eccentricity_PB40'] = df.apply(lambda x: long_eccentricity_pb40(x.luAc,x.plAc,x.distal,x.OS,pbth,gap),axis=1)
    t1 = time.time()
    total = t1-t0
    print('Feature 113-122 - Execution time: ',total,"secs")

    print('Feature extraction complete!!!')