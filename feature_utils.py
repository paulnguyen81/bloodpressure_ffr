"""
Add functions for feature extraction
"""
import pickle
import numpy as np
import pandas as pd
import time
import os

def windowed_mean(y,w):
    """
    moving average of y with
    window size = w
    step size = w/2
    """
    ymm=[]
    for j in range(len(y))[0:len(y):int(w/2)]:
        if len(y[j:j+w])==w:
            ymm.append(y[j:j+w].mean())
    return ymm

def PB(lu, pl):
    """
    lu, pl: lists or array-like
    returns plaque burden
    """
    x = np.array(lu)
    y = np.array(pl)
    return y/(x+y)

def findlesions(i,df,th):
    """
    Returns a list of lesion start frame and end fraim pair
    It ignores the dips (under the threshold) in a lesion if it's less than 300 frames long, and considers to be the same lesion.
    """
    x0 = df.ix[i].distal
    pb = PB(df.ix[i].luA, df.ix[i].plA)
    Y = pb >th
    dY = Y[:-1].astype(int)-Y[1:].astype(int)
    li = np.where(dY==-1)[0]+1
    lf = np.where(dY==1)[0]+1
    #print(df.ix[i].PID, df.ix[i].ffr)
    if li[0]<lf[0]:
        if len(li)==len(lf):
            pass
        elif len(li)>len(lf):
            lf= np.array(list(lf)+[len(Y)])
    else:
        if len(li)==len(lf):
            li= np.array([0]+list(li))  
            lf= np.array(list(lf)+[len(Y)])
        elif len(li)<len(lf):
            li= np.array([0]+list(li))    
    br = np.where((li[1:]-lf[:-1])>300)[0]
    nl = len(br)+1 #number of lesions
    lesions=[]
    if nl<2:
        lesions = [[0,len(Y)]]
    else:
        for j in range(nl):
            if j<1: #first lesion
                lesions.append([li[0],lf[br[j]]])
            elif j==nl-1: #last lesion
                lesions.append([li[br[j-1]+1],lf[-1]])
            else: #middle
                lesions.append([li[br[j-1]+1],lf[br[j]]])
    #print(nl)
    #print(lesions,'\n')
    return lesions
    
if __name__=='__main__':
    with open(os.getcwd()+'/data/df50.p','rb') as f:
        df50 = pickle.load(f)

#### example use
    lu = df50.luA
    pl = df50.plA
    pb = PB(lu,pl)
    for i in df50.index[:10]:
    	print(findlesions(i,df50,0.4))
        






################################################################### 
###################################################################

#### FEATURES 1-2
def plaqueburden(lu,pl):
    """
    lu, pl: lists
    returns plaque burden
    """
    x = np.array(lu)
    y = np.array(pl)
    return y/(x + y)
    
def lesion(lu,pl,distal,os,pbth,gap):
    """
    pb: plaqueburden list
    th: threshold (.4, .7)
    gap: number of frames to ignore with pb below threshold in lesion
    returns nested list with lesion start and end
    """
    
    index = np.where(plaqueburden(lu,pl) > pbth)[0]
    index2 = index[(index >= distal) & (index <= os)]   #only lesion within roi
    if len(index2 != 0):
        ### combine lesions if number of frames between is less than gap
        runs = []
        start = index2[0]
        count = 0 
        end = []
        for i in index2:                      
            if i == (start + count): 
                end = i
            elif i - end - 1 < gap:
                end = i
            else:
                runs.append([start,end])   #append lesion to runs and restart counter
                start = i
                end = i
                count = 0
        
            #print(i, count, start)
            count +=1
        runs.append([start,end])
        
        #check lesion length greater than 300
        check = []
        for r in runs:
            start = r[0]
            end = r[1]
            if (end - start) > 300:
                check.append(True)
            else:
                check.append(False)
        check = np.array(check)
        runs = np.array(runs)
        runs = list(runs[check])
        if len(runs) == 0:
            return -1
        else:
            return runs
    else:
        return -1
    
    



#### FEATURES 3-5
def mla(luA,distal,os):
    """
    luA: list of lumen area mm2
    distal: roi distal
    os: roi ostium 
    returns minimum luA value
    """
    roi = luA[distal:os] 
    return min(roi)

def fromos_pb(luA,plA,distal,os,pbth,gap): 
    """
    luA/plA: list of lumen/plaque area mm2
    pbth: threshold (0.4, 0.7)
    gap: number of frames to ignore with pb below threshold in lesion
    distal: roi distal
    os: roi ostium
    returns distance from OS to farthest right lesion proximal site
    """
    lesionlist = lesion(luA,plA,distal,os,pbth,gap)
    if(lesionlist == -1):
        return 0
    else:
        lesionproximal = lesionlist[-1][-1] #ignores multiple lesions in roi and only uses farthest right lesion
        return (os - lesionproximal) #if negative the lesion is on outside of roi ostium

def fromos_mla(luA,distal,os): 
    """
    luA: list of lumen area mm2
    distal: roi distal
    os: roi ostium
    returns number of frames between MLA and OS
    """
    indexmla = np.argmin(luA[distal:os]) + distal
    return os - indexmla



################################################################### 
###################################################################

#### FEATURES 6-11
def mla(luA,distal,os):
    """
    luA: list of lumen area mm2
    distal: roi distal
    os: roi ostium 
    returns minimum mla value in roi
    """
    roi = luA[distal:os] 
    return min(roi)

def eem_mla(luA,plA,distal,os):
    """
    luA,plA: list of lumen area and plaque area mm2
    distal: roi distal
    os: roi ostium 
    returns eem mm2 at mla within roi
    """
    indexmla = np.where(luA == mla(luA,distal,os))
    x = np.array(luA)
    y = np.array(plA)
    eem = x + y
    return eem[indexmla][0]

def pb_mla(luA,plA,distal,os):
    """
    luA,plA: list of lumen area and plaque area mm2
    distal: roi distal
    os: roi ostium 
    returns plaqueburden mm2 at mla in roi
    """
    indexmla = np.where(luA == mla(luA,distal,os))
    pb = plaqueburden(luA,plA)
    return pb[indexmla][0]

def max_pb(luA,plA,distal,os):
    """
    luA,plA: list of lumen area and plaque area mm2
    distal: roi distal
    os: roi ostium 
    returns max plaqueburden mm2 in roi
    """
    pb = plaqueburden(luA,plA)
    roi = pb[distal:os]
    return max(roi)

def numberofpb(pb,pbth,distal,os):
    """
    pb: plaqueburden list
    th: threshold (0.4, 0.7)
    distal: roi distal
    os: roi ostium 
    returns number of frames with pb > threshold
    """
    roi = np.array(pb[distal:os])
    index = np.where(roi > pbth)[0]
    return len(index)



################################################################### 
###################################################################

#### FEATURES 12-38
### Combined features 12-38 by including a plaque burden threshold parameter. 
### this asumes the features are within the ROI. It wasn't defined in the variables_def.xlsx
### similar to features 12-20 which clearly defined within ROI region


def Intersection(lst1, lst2):
    """
    lst1,lst2: list
    returns the intersection (common frames) between two lists
    """
    return set(lst1).intersection(lst2)


def no_lumen_roi(luA,plA,distal,os,th,pbth):
    """
    luA,plA: list of lumen/plaque area mm2
    distal: roi distal
    os: roi ostium
    th: threshold (4,3,2.5)
    pbth: plaqueburden threshold (0.4-default,0.7,0)
    returns number of frames greater than thresh in roi
    """
    roi = np.array(luA[distal:os])
    numlua = np.where(roi < th)  #converts numlua(tuple) to list
    numlua = numlua[0].tolist()
    
    if pbth != 1:
        x = np.array(luA[distal:os])
        y = np.array(plA[distal:os])
        pb = y/(x + y)
        pb2 = np.where(pb > pbth)
        pb2 = pb2[0].tolist()  #converts pb2(tuple) to list
        numlua = list(Intersection(numlua,pb2))
        
    return len(numlua)

def sum_plaque_roi(luA,plA,distal,os,pbth):
    """
    plA: list of plaque area mm2
    distal: roi distal
    os: roi ostium
    pbth: plaqueburden threshold (0.4-default,0.7,0)
    returns the total plaque area mm2 in roi
    """
    if pbth == 1:
        roi = np.array(plA[distal:os])
    else:
        x = np.array(luA[distal:os])
        y = np.array(plA[distal:os])
        pb = y/(x + y)
        pb2 = np.where(pb > pbth)
        roi = plA[pb2]
        
    return sum(roi)
    
def sum_eem_roi(luA,plA,distal,os,pbth):
    """
    luA, plA: list of lumen/plaque area mm2
    distal: roi distal
    os: roi ostium
    pbth: plaqueburden threshold (0.4-default,0.7,0)
    returns the total eem area mm2 in roi
    """
    x = np.array(luA[distal:os])
    y = np.array(plA[distal:os])
    if pbth == 1:
        eem = sum(x + y)
    else:
        pb = y/(x + y)
        pb2 = np.where(pb > pbth)
        eem = sum(x[pb2] + y[pb2])
        
    return eem
    
def pb_roi(luA,plA,distal,os,pbth):
    """
    luA, plA: list of lumen/plaque area mm2
    distal: roi distal
    os: roi ostium
    pbth: plaqueburden threshold (0.4-default,0.7,0)
    returns the overall plaqueburden in roi
    """
    plsum = sum_plaque_roi(luA,plA,distal,os,pbth)
    eemsum = sum_eem_roi(luA,plA,distal,os,pbth)
    if eemsum == 0:
        return 0
    else:
        pb = plsum/eemsum
        return pb


def mean_lumen_roi(luA,plA,distal,os,pbth):
    """
    lu: list of lumen area
    distal: roi distal
    os: roi ostium
    pbth: plaqueburden threshold (0.4-default,0.7,0)
    returns the mean lumen in roi
    """
    x = np.array(luA[distal:os])
    y = np.array(plA[distal:os])
    pb = y/(x + y)
    pb2 = np.where(pb > pbth)
    if len(pb2[0])==0:     #if no values above thresh return 0
        return 0
    else:
        roi = luA[distal:os]
        roi = roi[pb2]
        return np.mean(roi)
        
    


def mean_plaque_roi(luA,plA,distal,os,pbth):
    """
    lu: list of plaque area
    distal: roi distal
    os: roi ostium
    pbth: plaqueburden threshold (0.4-default,0.7,0)
    returns the mean plaque in roi
    """
    x = np.array(luA[distal:os])
    y = np.array(plA[distal:os])
    pb = y/(x + y)
    pb2 = np.where(pb > pbth)
    if len(pb2[0])==0:     #if no values above thresh return 0
        return 0
    else:
        roi = plA[distal:os]
        roi = roi[pb2]
        return np.mean(roi) 
    
    

def mean_eem_roi(luA,plA,distal,os,pbth):
    """
    lu,pl: list
    distal: roi distal
    os: roi ostium
    pbth: plaqueburden threshold (0.4-default,0.7,0)
    returns the mean eem in roi
    """
    x = np.array(luA[distal:os])
    y = np.array(plA[distal:os])
    pb = y/(x + y)
    pb2 = np.where(pb > pbth)
    if len(pb2[0])==0:     #if no values above thresh return 0
        return 0
    else:
        x = np.array(luA[distal:os])
        y = np.array(plA[distal:os])
        roi = x + y
        roi = roi[pb2]
        return np.mean(roi)
    
        



################################################################### 
###################################################################

#### FEATURES 39-47
#### effective plaque burden within the worst-5mm region 150 frames on either side of MLA (5-mm)
#### mla() function for min lumen area see FEATURES 3-5

def worst5mm (luA,distal,os):
    indexmla = np.argmin(luA[distal:os]) + distal
    if indexmla - 150 < distal:
        distal5mm = distal
    else:
        distal5mm = indexmla - 150
    if indexmla + 150 > os:
        proximal5mm = os
    else:
        proximal5mm = indexmla + 150
    return distal5mm, proximal5mm

def no_lumen_worst(luA,distal,os,luth=1):
    """
    luA: list of lumen area mm2
    distal: roi distal
    os: roi ostium 
    luth: threshold (luA,1-default,4.0,3.0,2.5,)
    returns number of frames with luman area < threshold in worst 5mm 
    """
    index5mm = worst5mm(luA,distal,os)
    lumenarea = luA[index5mm[0]:index5mm[1]]
    lumenarea2 = np.where(lumenarea < luth)[0]
    return len(lumenarea2)

def sum_plaque_worst(luA,plA,distal,os):
    """
    plA: list of plaque area mm2
    distal: roi distal
    os: roi ostium 
    returns sum of plaque area in worst 5mm 
    """
    index5mm = worst5mm(luA,distal,os)
    plaquearea = plA[index5mm[0]:index5mm[1]]
    return sum(plaquearea)

def sum_eem_worst(luA,plA,distal,os):
    """
    luA: list of lumen area mm2
    plA: list of plaque area mm2
    distal: roi distal
    os: roi ostium 
    returns sum of eem area in worst 5mm 
    """
    index5mm = worst5mm(luA,distal,os)
    lumenarea = luA[index5mm[0]:index5mm[1]]
    plaquearea = plA[index5mm[0]:index5mm[1]]
    eemarea = lumenarea + plaquearea
    return sum(eemarea)

def pb_worst(luA,plA,distal,os):
    """
    luA: list of lumen area mm2
    plA: list of plaque area mm2
    distal: roi distal
    os: roi ostium 
    returns plaque burden in worst 5mm 
    """
    index5mm = worst5mm(luA,distal,os)
    lumenarea = luA[index5mm[0]:index5mm[1]]
    plaquearea = plA[index5mm[0]:index5mm[1]]
    eemarea = lumenarea + plaquearea
    return sum(plaquearea)/sum(eemarea)

def mean_lumen_worst(luA,distal,os):
    """
    luA: list of lumen area mm2
    distal: roi distal
    os: roi ostium 
    returns plaque burden in worst 5mm 
    """
    index5mm = worst5mm(luA,distal,os)
    lumenarea = luA[index5mm[0]:index5mm[1]]
    return np.mean(lumenarea)

def mean_plaque_worst(luA,plA,distal,os):
    """
    luA: list of lumen area mm2
    plA: list of plaque area mm2
    distal: roi distal
    os: roi ostium 
    returns plaque burden in worst 5mm 
    """
    index5mm = worst5mm(luA,distal,os)
    plaquearea = plA[index5mm[0]:index5mm[1]]
    return np.mean(plaquearea)

def mean_eem_worst(luA,plA,distal,os):
    """
    luA: list of lumen area mm2
    plA: list of plaque area mm2
    distal: roi distal
    os: roi ostium 
    returns plaque burden in worst 5mm 
    """
    index5mm = worst5mm(luA,distal,os)
    lumenarea = luA[index5mm[0]:index5mm[1]]
    plaquearea = plA[index5mm[0]:index5mm[1]]
    eemarea = lumenarea + plaquearea
    return np.mean(eemarea)



################################################################### 
###################################################################

#### FEATURES 48-57
#### effective plaque burden in the proximal reference frames bwteen the lesion proximal edge to beginning of the ROI

def no_lumen_prox(luA,plA,distal,os,pbth,th,gap):
    """
    luA,plA: list of lumen/plaque area mm2
    os: roi ostium 
    pbth: plaqueburden threshold for lesion (0.4,0.7)
    th: threshold for lumen area (4.0,3.0,2.5,)
    gap: number of frames to ignore with pb below threshold in lesion
    returns number of frames with luman area < threshold in proximal reference
    """
    if (lesion(luA,plA,distal,os,pbth,gap) == -1 or 
        lesion(luA,plA,distal,os,pbth,gap)[-1][-1] == os):
        return 0   #no lesion return 0
    else:
        indexprox = lesion(luA,plA,distal,os,pbth,gap)[-1][-1]
        lumenarea = np.where(luA[indexprox:os]<th)[0]
        return len(lumenarea)
    

def sum_plaque_prox(luA,plA,distal,os,pbth,gap):
    """
    luA,plA: list of lumen/plaque area mm2
    os: roi ostium 
    pbth: plaqueburden threshold for lesion (0.4,0.7)
    gap: number of frames to ignore with pb below threshold in lesion
    returns sum of plaquearea in proximal reference
    """
    if (lesion(luA,plA,distal,os,pbth,gap) == -1 or 
        lesion(luA,plA,distal,os,pbth,gap)[-1][-1] == os):
        return 0   #no lesion return 0   
    else:
        indexprox = lesion(luA,plA,distal,os,pbth,gap)[-1][-1]
        plaquearea = plA[indexprox:os]
        return sum(plaquearea)


def sum_eem_prox(luA,plA,distal,os,pbth,gap):
    """
    luA,plA: list of lumen/plaque area mm2
    os: roi ostium 
    pbth: plaqueburden threshold for lesion (0.4,0.7)
    gap: number of frames to ignore with pb below threshold in lesion
    returns sum of eem area in proximal reference
    """
    if (lesion(luA,plA,distal,os,pbth,gap) == -1 or 
        lesion(luA,plA,distal,os,pbth,gap)[-1][-1] == os):
        return 0   #no lesion return 0
    else:
        indexprox = lesion(luA,plA,distal,os,pbth,gap)[-1][-1]
        lumenarea = luA[indexprox:os]
        plaquearea = plA[indexprox:os]
        eem = lumenarea + plaquearea
        return sum(eem)

def pb_prox(luA,plA,distal,os,pbth,gap):
    """
    luA,plA: list of lumen/plaque area mm2
    os: roi ostium 
    pbth: plaqueburden threshold for lesion (0.4,0.7)
    gap: number of frames to ignore with pb below threshold in lesion
    returns plaqueburden in proximal reference
    """
    if (lesion(luA,plA,distal,os,pbth,gap) == -1 or 
        lesion(luA,plA,distal,os,pbth,gap)[-1][-1] == os):
        return 0   #no lesion return 0
    else:
        indexprox = lesion(luA,plA,distal,os,pbth,gap)[-1][-1]
        lumenarea = luA[indexprox:os]
        plaquearea = plA[indexprox:os]
        eem = lumenarea + plaquearea
        return sum(plaquearea)/sum(eem)
         
def mean_lumen_prox(luA,plA,distal,os,pbth,gap):
    """
    luA,plA: list of lumen/plaque area mm2
    os: roi ostium 
    pbth: plaqueburden threshold for lesion (0.4,0.7)
    gap: number of frames to ignore with pb below threshold in lesion
    returns mean lumen area in proximal reference
    """
    if (lesion(luA,plA,distal,os,pbth,gap) == -1 or 
        lesion(luA,plA,distal,os,pbth,gap)[-1][-1] == os):
        return 0   #no lesion return 0
    else:
        indexprox = lesion(luA,plA,distal,os,pbth,gap)[-1][-1]
        lumenarea = luA[indexprox:os]
        return np.mean(lumenarea)
        
def mean_plaque_prox(luA,plA,distal,os,pbth,gap):
    """
    luA,plA: list of lumen/plaque area mm2
    os: roi ostium 
    pbth: plaqueburden threshold for lesion (0.4,0.7)
    gap: number of frames to ignore with pb below threshold in lesion
    returns mean plaque area in proximal reference
    """
    if (lesion(luA,plA,distal,os,pbth,gap) == -1 or 
        lesion(luA,plA,distal,os,pbth,gap)[-1][-1] == os):
        return 0   #no lesion return 0
    else:
        indexprox = lesion(luA,plA,distal,os,pbth,gap)[-1][-1]
        plaquearea = plA[indexprox:os]
        return np.mean(plaquearea)
    
def mean_eem_prox(luA,plA,distal,os,pbth,gap):
    """
    luA,plA: list of lumen/plaque area mm2
    os: roi ostium 
    pbth: plaqueburden threshold for lesion (0.4,0.7)
    gap: number of frames to ignore with pb below threshold in lesion
    returns mean eem area in proximal reference
    """
    if (lesion(luA,plA,distal,os,pbth,gap) == -1 or 
        lesion(luA,plA,distal,os,pbth,gap)[-1][-1] == os):
        return 0   #no lesion return 0
    else:
        indexprox = lesion(luA,plA,distal,os,pbth,gap)[-1][-1]
        lumenarea = luA[indexprox:os]
        plaquearea = plA[indexprox:os]
        eem = lumenarea + plaquearea
        return np.mean(eem)

def max_eem_prox(luA,plA,distal,os,pbth,gap):
    """
    luA,plA: list of lumen/plaque area mm2
    os: roi ostium 
    pbth: plaqueburden threshold for lesion (0.4,0.7)
    gap: number of frames to ignore with pb below threshold in lesion
    returns mean eem area in proximal reference
    """
    if (lesion(luA,plA,distal,os,pbth,gap) == -1 or 
        lesion(luA,plA,distal,os,pbth,gap)[-1][-1] == os):
        return 0   #no lesion return 0
    else:
        indexprox = lesion(luA,plA,distal,os,pbth,gap)[-1][-1]
        lumenarea = luA[indexprox:os]
        plaquearea = plA[indexprox:os]
        eem = lumenarea + plaquearea
        return max(eem) 



################################################################### 
###################################################################

#### FEATURES 58-67
#### effective plaque burden in the distal reference frames bwteen the lesion proximal edge to beginning of the ROI

def no_lumen_distal(luA,plA,distal,os,pbth,th,gap):
    """
    luA,plA: list of lumen/plaque area mm2
    distal: roi distal 
    pbth: plaqueburden threshold for lesion (0.4,0.7)
    th: threshold for lumen area (4.0,3.0,2.5,)
    gap: number of frames to ignore with pb below threshold in lesion
    returns number of frames with luman area < threshold in distal reference
    """
    if (lesion(luA,plA,distal,os,pbth,gap) == -1 or 
        lesion(luA,plA,distal,os,pbth,gap)[0][0] == distal):
        return 0   #no lesion return 0
    else:
        indexdistal = lesion(luA,plA,distal,os,pbth,gap)[0][0]
        lumenarea = np.where(luA[distal:indexdistal]<th)[0]
        return len(lumenarea)

def sum_plaque_distal(luA,plA,distal,os,pbth,gap):
    """
    luA,plA: list of lumen/plaque area mm2
    distal: roi distal 
    pbth: plaqueburden threshold for lesion (0.4,0.7)
    gap: number of frames to ignore with pb below threshold in lesion
    returns sum of plaquearea in distal reference
    """
    if (lesion(luA,plA,distal,os,pbth,gap) == -1 or 
        lesion(luA,plA,distal,os,pbth,gap)[0][0] == distal):
        return 0   #no lesion return 0
    else:
        indexdistal = lesion(luA,plA,distal,os,pbth,gap)[0][0]
        plaquearea = plA[distal:indexdistal]
        return sum(plaquearea)   

def sum_eem_distal(luA,plA,distal,os,pbth,gap):
    """
    luA,plA: list of lumen/plaque area mm2
    distal: roi distal 
    pbth: plaqueburden threshold for lesion (0.4,0.7)
    gap: number of frames to ignore with pb below threshold in lesion
    returns sum of eem area in distal reference
    """
    if (lesion(luA,plA,distal,os,pbth,gap) == -1 or 
        lesion(luA,plA,distal,os,pbth,gap)[0][0] == distal):
        return 0   #no lesion return 0
    else:
        indexdistal = lesion(luA,plA,distal,os,pbth,gap)[0][0]
        lumenarea = luA[distal:indexdistal]
        plaquearea = plA[distal:indexdistal]
        eem = lumenarea + plaquearea
        return sum(eem)

def pb_distal(luA,plA,distal,os,pbth,gap):
    """
    luA,plA: list of lumen/plaque area mm2
    distal: roi distal 
    pbth: plaqueburden threshold for lesion (0.4,0.7)
    gap: number of frames to ignore with pb below threshold in lesion
    returns plaqueburden in distal reference
    """
    if (lesion(luA,plA,distal,os,pbth,gap) == -1 or 
        lesion(luA,plA,distal,os,pbth,gap)[0][0] == distal):
        return 0   #no lesion return 0
    else:
        indexdistal = lesion(luA,plA,distal,os,pbth,gap)[0][0]
        lumenarea = luA[distal:indexdistal]
        plaquearea = plA[distal:indexdistal]
        eem = lumenarea + plaquearea
        return sum(plaquearea)/sum(eem)
            
def mean_lumen_distal(luA,plA,distal,os,pbth,gap):
    """
    luA,plA: list of lumen/plaque area mm2
    distal: roi distal 
    pbth: plaqueburden threshold for lesion (0.4,0.7)
    gap: number of frames to ignore with pb below threshold in lesion
    returns mean lumen area in distal reference
    """
    if (lesion(luA,plA,distal,os,pbth,gap) == -1 or 
        lesion(luA,plA,distal,os,pbth,gap)[0][0] == distal):
        return 0   #no lesion return 0
    else:
        indexdistal = lesion(luA,plA,distal,os,pbth,gap)[0][0]
        lumenarea = luA[distal:indexdistal]
        return np.mean(lumenarea)
       
def mean_plaque_distal(luA,plA,distal,os,pbth,gap):
    """
    luA,plA: list of lumen/plaque area mm2
    distal: roi distal  
    pbth: plaqueburden threshold for lesion (0.4,0.7)
    gap: number of frames to ignore with pb below threshold in lesion
    returns mean plaque area in distal reference
    """
    if (lesion(luA,plA,distal,os,pbth,gap) == -1 or 
        lesion(luA,plA,distal,os,pbth,gap)[0][0] == distal):
        return 0   #no lesion return 0
    else:
        indexdistal = lesion(luA,plA,distal,os,pbth,gap)[0][0]
        plaquearea = plA[distal:indexdistal]
        return np.mean(plaquearea)      
    
def mean_eem_distal(luA,plA,distal,os,pbth,gap):
    """
    luA,plA: list of lumen/plaque area mm2
    distal: roi distal 
    pbth: plaqueburden threshold for lesion (0.4,0.7)
    gap: number of frames to ignore with pb below threshold in lesion
    returns mean eem area in distal reference
    """
    if (lesion(luA,plA,distal,os,pbth,gap) == -1 or 
        lesion(luA,plA,distal,os,pbth,gap)[0][0] == distal):
        return 0   #no lesion return 0
    else:
        indexdistal = lesion(luA,plA,distal,os,pbth,gap)[0][0]
        lumenarea = luA[distal:indexdistal]
        plaquearea = plA[distal:indexdistal]
        eem = lumenarea + plaquearea
        return np.mean(eem)
        
def max_eem_distal(luA,plA,distal,os,pbth,gap):
    """
    luA,plA: list of lumen/plaque area mm2
    distal: roi distal 
    pbth: plaqueburden threshold for lesion (0.4,0.7)
    gap: number of frames to ignore with pb below threshold in lesion
    returns mean eem area in distal reference
    """
    if (lesion(luA,plA,distal,os,pbth,gap) == -1 or 
        lesion(luA,plA,distal,os,pbth,gap)[0][0] == distal):
        return 0   #no lesion return 0
    else:
        indexdistal = lesion(luA,plA,distal,os,pbth,gap)[0][0]
        lumenarea = luA[distal:indexdistal]
        plaquearea = plA[distal:indexdistal]
        eem = lumenarea + plaquearea
        return max(eem)
        



################################################################### 
###################################################################

#### FEATURES 68-77
### effective plaque burden in the proximal reference 5mm
### the functions test if a proximal reference 5mm exists
### proximal reference 5mm exists if there are more the 300 frames and the center (min luA) 
### is not within 150 frames of the lesion proximal edge or os

def no_lumen40_prox5(luA,plA,distal,os,pbth,th,gap):
    """
    luA,plA: list of lumen/plaque area mm2
    distal: roi distal
    os: roi ostium 
    pbth: plaqueburden threshold for lesion (0.4, 0.7)
    th: threshold for lumen area (4.0, 3.0, 2.5)
    gap: number of frames to ignore with pb below threshold in lesion
    returns number of frames with luman area < threshold in proximal reference 5mm
    """
    if (lesion(luA,plA,distal,os,pbth,gap) == -1 or lesion(luA,plA,distal,os,pbth,gap)[-1][-1] == os):
        return 0   #no lesion return 0
    else:
        prox5start = lesion(luA,plA,distal,os,pbth,gap)[-1][-1]
        if os - 300 > prox5start:
            prox5end = prox5start + 300
        else:
            prox5end = os
        lumenarea = np.where(luA[prox5start:prox5end]<th)[0]
        return len(lumenarea)


def sum_plaque_prox5(luA,plA,distal,os,pbth,gap):
    """
    luA,plA: list of lumen/plaque area mm2
    distal: roi distal
    os: roi ostium 
    pbth: plaqueburden threshold for lesion (0.4, 0.7)
    gap: number of frames to ignore with pb below threshold in lesion
    returns sum of plaque area in proximal reference 5mm
    """
    if (lesion(luA,plA,distal,os,pbth,gap) == -1 or lesion(luA,plA,distal,os,pbth,gap)[-1][-1] == os):
        return 0   #no lesion return 0
    else:
        prox5start = lesion(luA,plA,distal,os,pbth,gap)[-1][-1]
        if os - 300 > prox5start:
            prox5end = prox5start + 300
        else:
            prox5end = os
        plaquearea = plA[prox5start:prox5end]
        return sum(plaquearea)

    
def sum_eem_prox5(luA,plA,distal,os,pbth,gap):
    """
    luA,plA: list of lumen/plaque area mm2
    distal: roi distal
    os: roi ostium 
    pbth: plaqueburden threshold for lesion (0.4, 0.7)
    gap: number of frames to ignore with pb below threshold in lesion
    returns sum of eem in proximal reference 5mm
    """
    if (lesion(luA,plA,distal,os,pbth,gap) == -1 or lesion(luA,plA,distal,os,pbth,gap)[-1][-1] == os):
        return 0   #no lesion return 0
    else:
        prox5start = lesion(luA,plA,distal,os,pbth,gap)[-1][-1]
        if os - 300 > prox5start:
            prox5end = prox5start + 300
        else:
            prox5end = os
        x = np.array(luA[prox5start:prox5end])
        y = np.array(plA[prox5start:prox5end])
        return sum(x + y)

    
def pb_prox5(luA,plA,distal,os,pbth,gap):
    """
    luA,plA: list of lumen/plaque area mm2
    distal: roi distal
    os: roi ostium 
    pbth: plaqueburden threshold for lesion (0.4, 0.7)
    gap: number of frames to ignore with pb below threshold in lesion
    returns plaque burden in proximal reference 5mm
    """
    if (lesion(luA,plA,distal,os,pbth,gap) == -1 or lesion(luA,plA,distal,os,pbth,gap)[-1][-1] == os):
        return 0   #no lesion return 0
    else:
        prox5start = lesion(luA,plA,distal,os,pbth,gap)[-1][-1]
        if os - 300 > prox5start:
            prox5end = prox5start + 300
        else:
            prox5end = os
        x = np.array(luA[prox5start:prox5end])
        y = np.array(plA[prox5start:prox5end])
        return sum(y)/sum(x + y)

    
def mean_lumen_prox5(luA,plA,distal,os,pbth,gap):
    """
    luA,plA: list of lumen/plaque area mm2
    distal: roi distal
    os: roi ostium 
    pbth: plaqueburden threshold for lesion (0.4, 0.7)
    gap: number of frames to ignore with pb below threshold in lesion
    returns mean luman area in proximal reference 5mm
    """
    if (lesion(luA,plA,distal,os,pbth,gap) == -1 or lesion(luA,plA,distal,os,pbth,gap)[-1][-1] == os):
        return 0   #no lesion return 0
    else:
        prox5start = lesion(luA,plA,distal,os,pbth,gap)[-1][-1]
        if os - 300 > prox5start:
            prox5end = prox5start + 300
        else:
            prox5end = os
        lumenarea = luA[prox5start:prox5end]
        return np.mean(lumenarea)
    
def mean_plaque_prox5(luA,plA,distal,os,pbth,gap):
    """
    luA,plA: list of lumen/plaque area mm2
    distal: roi distal
    os: roi ostium 
    pbth: plaqueburden threshold for lesion (0.4, 0.7)
    gap: number of frames to ignore with pb below threshold in lesion
    returns mean plaque area in proximal reference 5mm
    """
    if (lesion(luA,plA,distal,os,pbth,gap) == -1 or lesion(luA,plA,distal,os,pbth,gap)[-1][-1] == os):
        return 0   #no lesion return 0
    else:
        prox5start = lesion(luA,plA,distal,os,pbth,gap)[-1][-1]
        if os - 300 > prox5start:
            prox5end = prox5start + 300
        else:
            prox5end = os
        plaquearea = plA[prox5start:prox5end]
        return np.mean(plaquearea)
    
def mean_eem_prox5(luA,plA,distal,os,pbth,gap):
    """
    luA,plA: list of lumen/plaque area mm2
    distal: roi distal
    os: roi ostium 
    pbth: plaqueburden threshold for lesion (0.4, 0.7)
    gap: number of frames to ignore with pb below threshold in lesion
    returns mean eem in proximal reference 5mm
    """
    if (lesion(luA,plA,distal,os,pbth,gap) == -1 or lesion(luA,plA,distal,os,pbth,gap)[-1][-1] == os):
        return 0   #no lesion return 0
    else:
        prox5start = lesion(luA,plA,distal,os,pbth,gap)[-1][-1]
        if os - 300 > prox5start:
            prox5end = prox5start + 300
        else:
            prox5end = os
        x = np.array(luA[prox5start:prox5end])
        y = np.array(plA[prox5start:prox5end])
        return np.mean(x + y)
    
def max_eem_prox5(luA,plA,distal,os,pbth,gap):
    """
    luA,plA: list of lumen/plaque area mm2
    distal: roi distal
    os: roi ostium 
    pbth: plaqueburden threshold for lesion (0.4, 0.7)
    gap: number of frames to ignore with pb below threshold in lesion
    returns maximum eem in proximal reference 5mm
    """
    if (lesion(luA,plA,distal,os,pbth,gap) == -1 or lesion(luA,plA,distal,os,pbth,gap)[-1][-1] == os):
        return 0   #no lesion return 0
    else:
        prox5start = lesion(luA,plA,distal,os,pbth,gap)[-1][-1]
        if os - 300 > prox5start:
            prox5end = prox5start + 300
        else:
            prox5end = os
        x = np.array(luA[prox5start:prox5end])
        y = np.array(plA[prox5start:prox5end])
        return max(x + y)



################################################################### 
###################################################################

#### FEATURES 78-87
### effective plaque burden in the distal reference 5mm
### the functions test if a distal reference 5mm exists
### distal reference 5mm exists if there are more the 300 frames and the center (min luA) 
### is not within 150 frames of the lesion distal edge or distal


############################

def no_lumen40_dist5(luA,plA,distal,os,pbth,th,gap):
    """
    luA,plA: list of lumen/plaque area mm2
    distal: roi distal
    os: roi ostium 
    pbth: plaqueburden threshold for lesion (0.4, 0.7)
    th: threshold for lumen area (4.0, 3.0, 2.5)
    gap: number of frames to ignore with pb below threshold in lesion
    returns number of frames with luman area < threshold in distal reference 5mm
    """
    if (lesion(luA,plA,distal,os,pbth,gap) == -1 or 
        lesion(luA,plA,distal,os,pbth,gap)[0][0] == distal):
        return 0   #no lesion return 0
    else:
        distal5end = lesion(luA,plA,distal,os,pbth,gap)[0][0]
        if distal + 300 < distal5end:
            distal5start = distal5end - 300
        else:
            distal5start = distal
        lumenarea = np.where(luA[distal5start:distal5end]<th)[0]
        return len(lumenarea)

def sum_plaque_dist5(luA,plA,distal,os,pbth,gap):
    """
    luA,plA: list of lumen/plaque area mm2
    distal: roi distal
    os: roi ostium 
    pbth: plaqueburden threshold for lesion (0.4, 0.7)
    gap: number of frames to ignore with pb below threshold in lesion
    returns sum of plaque area in distal reference 5mm
    """
    if (lesion(luA,plA,distal,os,pbth,gap) == -1 or 
        lesion(luA,plA,distal,os,pbth,gap)[0][0] == distal):
        return 0   #no lesion return 0
    else:
        distal5end = lesion(luA,plA,distal,os,pbth,gap)[0][0]
        if distal + 300 < distal5end:
            distal5start = distal5end - 300
        else:
            distal5start = distal
        plaquearea = plA[distal5start:distal5end]
        return sum(plaquearea)

def sum_eem_dist5(luA,plA,distal,os,pbth,gap):
    """
    luA,plA: list of lumen/plaque area mm2
    distal: roi distal
    os: roi ostium 
    pbth: plaqueburden threshold for lesion (0.4, 0.7)
    gap: number of frames to ignore with pb below threshold in lesion
    returns sum of eem in distal reference 5mm
    """
    if (lesion(luA,plA,distal,os,pbth,gap) == -1 or 
        lesion(luA,plA,distal,os,pbth,gap)[0][0] == distal):
        return 0   #no lesion return 0
    else:
        distal5end = lesion(luA,plA,distal,os,pbth,gap)[0][0]
        if distal + 300 < distal5end:
            distal5start = distal5end - 300
        else:
            distal5start = distal
        x = np.array(luA[distal5start:distal5end])
        y = np.array(plA[distal5start:distal5end])
        return sum(x + y)
    
def pb_dist5(luA,plA,distal,os,pbth,gap):
    """
    luA,plA: list of lumen/plaque area mm2
    distal: roi distal
    os: roi ostium 
    pbth: plaqueburden threshold for lesion (0.4, 0.7)
    gap: number of frames to ignore with pb below threshold in lesion
    returns plaque burden in distal reference 5mm
    """
    if (lesion(luA,plA,distal,os,pbth,gap) == -1 or 
        lesion(luA,plA,distal,os,pbth,gap)[0][0] == distal):
        return 0   #no lesion return 0
    else:
        distal5end = lesion(luA,plA,distal,os,pbth,gap)[0][0]
        if distal + 300 < distal5end:
            distal5start = distal5end - 300
        else:
            distal5start = distal
        x = np.array(luA[distal5start:distal5end])
        y = np.array(plA[distal5start:distal5end])
        return sum(y)/sum(x + y)   
    
def mean_lumen_dist5(luA,plA,distal,os,pbth,gap):
    """
    luA,plA: list of lumen/plaque area mm2
    distal: roi distal
    os: roi ostium 
    pbth: plaqueburden threshold for lesion (0.4, 0.7)
    th: threshold for lumen area (4.0, 3.0, 2.5)
    gap: number of frames to ignore with pb below threshold in lesion
    returns mean luman area in distal reference 5mm
    """
    if (lesion(luA,plA,distal,os,pbth,gap) == -1 or 
        lesion(luA,plA,distal,os,pbth,gap)[0][0] == distal):
        return 0   #no lesion return 0
    else:
        distal5end = lesion(luA,plA,distal,os,pbth,gap)[0][0]
        if distal + 300 < distal5end:
            distal5start = distal5end - 300
        else:
            distal5start = distal
        lumenarea = luA[distal5start:distal5end]
        return np.mean(lumenarea)  
    
def mean_plaque_dist5(luA,plA,distal,os,pbth,gap):
    """
    luA,plA: list of lumen/plaque area mm2
    distal: roi distal
    os: roi ostium 
    pbth: plaqueburden threshold for lesion (0.4, 0.7)
    gap: number of frames to ignore with pb below threshold in lesion
    returns mean plaque area in distal reference 5mm
    """
    if (lesion(luA,plA,distal,os,pbth,gap) == -1 or 
        lesion(luA,plA,distal,os,pbth,gap)[0][0] == distal):
        return 0   #no lesion return 0
    else:
        distal5end = lesion(luA,plA,distal,os,pbth,gap)[0][0]
        if distal + 300 < distal5end:
            distal5start = distal5end - 300
        else:
            distal5start = distal
        plaquearea = plA[distal5start:distal5end]
        return np.mean(plaquearea)
    
def mean_eem_dist5(luA,plA,distal,os,pbth,gap):
    """
    luA,plA: list of lumen/plaque area mm2
    distal: roi distal
    os: roi ostium 
    pbth: plaqueburden threshold for lesion (0.4, 0.7)
    gap: number of frames to ignore with pb below threshold in lesion
    returns mean eem in distal reference 5mm
    """
    if (lesion(luA,plA,distal,os,pbth,gap) == -1 or 
        lesion(luA,plA,distal,os,pbth,gap)[0][0] == distal):
        return 0   #no lesion return 0
    else:
        distal5end = lesion(luA,plA,distal,os,pbth,gap)[0][0]
        if distal + 300 < distal5end:
            distal5start = distal5end - 300
        else:
            distal5start = distal
        x = np.array(luA[distal5start:distal5end])
        y = np.array(plA[distal5start:distal5end])
        return np.mean(x + y)
    
def max_eem_dist5(luA,plA,distal,os,pbth,gap):
    """
    luA,plA: list of lumen/plaque area mm2
    distal: roi distal
    os: roi ostium 
    pbth: plaqueburden threshold for lesion (0.4, 0.7)
    gap: number of frames to ignore with pb below threshold in lesion
    returns max eem in distal reference 5mm
    """
    if (lesion(luA,plA,distal,os,pbth,gap) == -1 or 
        lesion(luA,plA,distal,os,pbth,gap)[0][0] == distal):
        return 0   #no lesion return 0
    else:
        distal5end = lesion(luA,plA,distal,os,pbth,gap)[0][0]
        if distal + 300 < distal5end:
            distal5start = distal5end - 300
        else:
            distal5start = distal
        x = np.array(luA[distal5start:distal5end])
        y = np.array(plA[distal5start:distal5end])
        return max(x + y)



################################################################### 
###################################################################

#### FEATURES 88-101


############################

def mean_lumen_aver(luA,plA,distal,os,pbth,gap):
    x = mean_lumen_prox5(luA,plA,distal,os,pbth,gap)
    y = mean_lumen_dist5(luA,plA,distal,os,pbth,gap)
    return (x + y)/ 2

def mean_eem_aver(luA,plA,distal,os,pbth,gap):
    x = mean_eem_prox5(luA,plA,distal,os,pbth,gap)
    y = mean_eem_dist5(luA,plA,distal,os,pbth,gap)
    if x == 0:
        return 0
    else:
        return (x + y)/ 2

def area1_stenosis_aver(luA,plA,distal,os,pbth,gap):
    x = mean_lumen_aver(luA,plA,distal,os,pbth,gap)
    y = mla(luA,distal,os)
    if x == 0:
        return 0
    else:
        return (x - y)/ x

def area1_stenosis_prox5(luA,plA,distal,os,pbth,gap):
    x = mean_lumen_prox5(luA,plA,distal,os,pbth,gap)
    y = mla(luA,distal,os)
    if x == 0:
        return 0
    else:
        return (x - y)/ x

def area1_stenosis_dist5(luA,plA,distal,os,pbth,gap):
    x = mean_lumen_dist5(luA,plA,distal,os,pbth,gap)
    y = mla(luA,distal,os)
    if x == 0:
        return 0
    else:
        return (x - y)/ x

def area2_stenosis_aver(luA,plA,distal,os,pbth,gap):
    x = mean_eem_aver(luA,plA,distal,os,pbth,gap)
    y = mla(luA,distal,os)
    if x == 0:
        return 0
    else:
        return (x - y)/ x

def area2_stenosis_prox5(luA,plA,distal,os,pbth,gap):
    x = mean_eem_prox5(luA,plA,distal,os,pbth,gap)
    y = mla(luA,distal,os)
    if x == 0:
        return 0
    else:
        return (x - y)/ x

def area2_stenosis_dist5(luA,plA,distal,os,pbth,gap):
    x = mean_eem_dist5(luA,plA,distal,os,pbth,gap)
    y = mla(luA,distal,os)
    if x == 0:
        return 0
    else:
        return (x - y)/ x

def area3_stenosis_aver(luA,plA,distal,os,pbth,gap):
    x = mean_lumen_aver(luA,plA,distal,os,pbth,gap)
    y = mean_lumen_worst(luA,distal,os)
    if x == 0:
        return 0
    else:
        return (x - y)/ x

def area3_stenosis_prox5(luA,plA,distal,os,pbth,gap):
    x = mean_lumen_prox5(luA,plA,distal,os,pbth,gap)
    y = mean_lumen_worst(luA,distal,os)
    if x == 0:
        return 0
    else:
        return (x - y)/ x

def area3_stenosis_dist5(luA,plA,distal,os,pbth,gap):
    x = mean_lumen_dist5(luA,plA,distal,os,pbth,gap)
    y = mean_lumen_worst(luA,distal,os)
    if x == 0:
        return 0
    else:
        return (x - y)/ x

def area4_stenosis_aver(luA,plA,distal,os,pbth,gap):
    x = mean_eem_aver(luA,plA,distal,os,pbth,gap)
    y = mean_lumen_worst(luA,distal,os)
    if x == 0:
        return 0
    else:
        return (x - y)/ x

def area4_stenosis_prox5(luA,plA,distal,os,pbth,gap):
    x = mean_eem_prox5(luA,plA,distal,os,pbth,gap)
    y = mean_lumen_worst(luA,distal,os)
    if x == 0:
        return 0
    else:
        return (x - y)/ x

def area4_stenosis_dist5(luA,plA,distal,os,pbth,gap):
    x = mean_eem_dist5(luA,plA,distal,os,pbth,gap)
    y = mean_lumen_worst(luA,distal,os)
    if x == 0:
        return 0
    else:
        return (x - y)/ x
    
################################################################### 
###################################################################

#### FEATURES 102-112


### NOTE: missing 121 and 122

############################  
    


################################################################### 
###################################################################

#### FEATURES 113-122


### NOTE: missing 121 and 122

############################
    
def ri_mla_ref(luA,plA,distal,os,pbth,gap):
    x = eem_mla(luA,plA,distal,os)
    y = mean_eem_aver(luA,plA,distal,os,pbth,gap)
    if y == 0:
        return 0
    else:
        return x/y

def ri_mla_prox5(luA,plA,distal,os,pbth,gap):
    x = eem_mla(luA,plA,distal,os)
    y = mean_eem_prox5(luA,plA,distal,os,pbth,gap)
    if y == 0:
        return 0
    else:
        return x/y

def ri_worst_ref(luA,plA,distal,os,pbth,gap):
    x = mean_eem_worst(luA,plA,distal,os)
    y = mean_eem_aver(luA,plA,distal,os,pbth,gap)
    if y == 0:
        return 0
    else:
        return x/y

def ri_worst_prox5(luA,plA,distal,os,pbth,gap):
    x = mean_eem_worst(luA,plA,distal,os)
    y = mean_eem_prox5(luA,plA,distal,os,pbth,gap)
    if y == 0:
        return 0
    else:
        return x/y

def variance_lumen_worst(luA,distal,os):
    """
    luA: list of lumen area mm2
    distal: roi distal
    os: roi ostium 
    returns variance of lumen area in worst 5mm 
    """
    index5mm = worst5mm(luA,distal,os)
    lumenarea = luA[index5mm[0]:index5mm[1]]
    return np.var(lumenarea)

def variance_lumen_pb40(luA,plA,distal,os,pbth):
    """
    luA,plA: list of lumen/plaque area mm2
    distal: roi distal
    os: roi ostium
    th: threshold (4,3,2.5)
    pbth: plaqueburden threshold (1-default,0.7,0.4)
    returns number of frames greater than thresh in roi
    """
    x = np.array(luA)
    y = np.array(plA)
    pb = y/(x + y)
    pb2 = np.where(pb > pbth)[0]
    roi = pb2[distal:os]
    return np.var(luA[roi])
    
def variance_plaque_worst(luA,plA,distal,os):
    """
    luA,plA: list of lumen/plaque area mm2
    distal: roi distal
    os: roi ostium 
    returns plaque area in worst 5mm 
    """
    index5mm = worst5mm(luA,distal,os)
    plaquearea = plA[index5mm[0]:index5mm[1]]
    return np.var(plaquearea)

def variance_plaque_pb40(luA,plA,distal,os,pbth):
    """
    luA,plA: list of lumen/plaque area mm2
    distal: roi distal
    os: roi ostium
    th: threshold (4,3,2.5)
    pbth: plaqueburden threshold (1-default,0.7,0.4)
    returns number of frames greater than thresh in roi
    """
    x = np.array(luA)
    y = np.array(plA)
    pb = y/(x + y)
    pb2 = np.where(pb > pbth)[0]
    roi = pb2[distal:os]
    return np.var(plA[roi])

def long_eccentricity_worst(luA,plA,distal,os):
    """
    luA,plA: list of lumen/plaque area mm2
    distal: roi distal
    os: roi ostium 
    returns plaque area in worst 5mm 
    """
    index5mm = worst5mm(luA,distal,os)
    plaquearea = plA[index5mm[0]:index5mm[1]]
    return np.var(plaquearea)


################################################################### 
###################################################################

#### VISUALIZATIONS

############################

def lesionvis(i, th, show=True):
    """
    i: index
    th: plaqueburden threshold
    """
    pb = plaqueburden(df50.loc[i].luA, df50.loc[i].plA)
    lesionindex = lesion(df50.loc[i].luA, df50.loc[i].plA,df50.loc[i].distal,df50.loc[i].OS,th,300)
    print (lesionindex)
    plt.figure(figsize=(12,5));
    plt.plot(pb)
    ymax = max(pb)
    plt.ylim(0,ymax)
    ki = df50['distal'].loc[i]
    kf = df50['OS'].loc[i]
    plt.axvline(kf, color='lime', label='OS')
    plt.axvline(ki, color='magenta',label='distal')
    plt.axhline(th, color='red',label=th)
    
    for l in range(0,len(lesionindex)):
        start = lesionindex[l][0]
        end = lesionindex[l][1]
        plt.axvspan(start, end, facecolor='r', alpha=0.2)
        
    plt.legend(loc="right",bbox_to_anchor=(1.3, 0.5));
    if show:
        plt.show()
    else:
        return plt 

    
################################################################### 
###################################################################

#### SMOOTHING

############################

def smoothconvolve(x,window,mode='same'):
    return np.convolve(x,np.ones((window,))/window,mode=mode)

def convolvevis(i, th, gap, show=True):
    """
    i: index
    th: plaqueburden threshold
    """
    # raw data
    pb = plaqueburden(df50.loc[i].luA, df50.loc[i].plA)
    # smoothed data
    pb2 = plaqueburden(df50.loc[i].luAc, df50.loc[i].plAc)
    lesionindex2 = lesion(df50.loc[i].luAc, df50.loc[i].plAc,df50.loc[i].distal,df50.loc[i].OS,th,gap)
    print(lesionindex2)
    plt.figure(figsize=(12,5));
    plt.plot(pb,alpha=0.3,label='original')
    plt.plot(pb2, color='red',alpha=0.7,label='smooth 50')
    ymax = max(pb)
    plt.ylim(0,ymax)
    ki = df50['distal'].loc[i]
    kf = df50['OS'].loc[i]
    plt.axvline(kf, color='cyan', label='OS')
    plt.axvline(ki, color='darkcyan',label='distal')
    plt.axhline(th, linestyle='--', color='red',label=th,alpha=0.5)
    
    if(lesionindex2 != -1):
        for l in range(0,len(lesionindex2)):
            start = lesionindex2[l][0]
            end = lesionindex2[l][1]
            plt.axvspan(start, end, facecolor='r', alpha=0.2)
    plt.legend(loc="right",bbox_to_anchor=(1.3, 0.5));
    if show:
        plt.show()
    else:
        return plt
    
# Example Use
# for i in df50.index:
#   print(i)
#   convolvevis(i,0.7,300)


def smoothsavitzky(x,window,polyorder,mode='mirror'):
    return savgol_filter(x, window,polyorder, mode=mode)


def savitzkyvis(i, th, gap, show=True):
    """
    i: index
    th: plaqueburden threshold
    """
    # raw data
    pb = plaqueburden(df50.loc[i].luA, df50.loc[i].plA)
    # smoothed data
    pb2 = plaqueburden(df50.loc[i].luAs, df50.loc[i].plAs)
    lesionindex2 = lesion(df50.loc[i].luAs, df50.loc[i].plAs,df50.loc[i].distal,df50.loc[i].OS,th,gap)
    print(lesionindex2)
    plt.figure(figsize=(12,5));
    plt.plot(pb,alpha=0.3,label='original')
    plt.plot(pb2, color='red',alpha=0.7)
    ymax = max(pb)
    plt.ylim(0,ymax)
    ki = df50['distal'].loc[i]
    kf = df50['OS'].loc[i]
    plt.axvline(kf, color='cyan', label='OS')
    plt.axvline(ki, color='darkcyan',label='distal')
    plt.axhline(th, color='red',label=th)
    
    if(lesionindex2 != -1):
        for l in range(0,len(lesionindex2)):
            start = lesionindex2[l][0]
            end = lesionindex2[l][1]
            plt.axvspan(start, end, facecolor='r', alpha=0.2)
        
    plt.legend(loc="right",bbox_to_anchor=(1.3, 0.5));
    plt.show()

    
# Example Use
# for i in df50.index:
#   print(i)
#   savitzkyvis(i,0.7,300)