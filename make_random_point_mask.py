
from PIL import Image
import numpy as np
import random
# from cv2 import cv2
import cv2
import glob,os
from tqdm import tqdm



def make_random_point_mask_for_train_for_label_mini(mean,std,\
                min_width,max_width,max_aspect_ratio,buffer_size,img_size=256):
    ok=0
    while not ok: 
        length,width=np.random.normal(mean, std, size=2)
        length,width=int(round(length)),int(round(width))
        length,width=max(length,width),min(length,width)
        aspect_ratio=length/(width+1e-08)
        if length <= img_size : 
            if width>=min_width and width <= max_width and width <= img_size :
                if aspect_ratio <= max_aspect_ratio : 
                    ok=1
    lines=np.array([length+buffer_size,width+buffer_size])
    lines[lines>img_size]=img_size
    length,width=lines
    
    return length,width




def get_contours_points_2(binary):
    ret, binary = cv2.threshold(binary,127,255,cv2.THRESH_BINARY)
    # contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, 2)
    # contour_points=contours[0]
    edges = cv2.Canny(binary,50,200)
    contour_points=np.argwhere(edges ==255 )
    contour_points=np.expand_dims(contour_points,axis=1)
    contour_points2=[]
    for contour_point in contour_points:
        if (contour_point==255).any() or (contour_point==0).any():
            continue
        else:
            contour_points2.append(contour_point)
    contour_points=contour_points2
    return contour_points



def make_random_point_mask_for_train_for_label_Non_directional(img_binary):
    ret, img_binary = cv2.threshold(img_binary,127,255,cv2.THRESH_BINARY)
    #
    contour_points=get_contours_points_2(img_binary)
    #
    #
    class_n=np.random.normal(0, 2, 1)
    class_n=int(class_n)
    class_n=min(max(class_n, 1),7)
    #
    contour_points_len=len(contour_points)
    if contour_points_len<50:
        class_n=1
    elif contour_points_len<100:
        class_n=min(class_n,2)
    elif contour_points_len<150:
        class_n=min(class_n,3)
    elif contour_points_len<200:
        class_n=min(class_n,4)
    elif contour_points_len<250:
        class_n=min(class_n,5)
    #
    center_location_range={}
    center_location_range_buffer=0.5
    center_location_range_for_one=contour_points_len/class_n*(1+center_location_range_buffer)/2
    for class_i in range(class_n):
        center_location=contour_points_len/class_n*(class_i+0.5)
        a=int(center_location-center_location_range_for_one)
        a=max(0,int(center_location-center_location_range_for_one))
        b=int(center_location+center_location_range_for_one)
        b=min(contour_points_len-1,int(center_location+center_location_range_for_one))
        center_location_range.update({class_i+1:[a,b]})
    #
    mask_all= np.zeros((256, 256), np.uint8)
    for class_i in range(class_n):
        ok=0
        while not ok:
            #
            mean,std=65,50
            min_width,max_width=15,150
            max_aspect_ratio=8
            buffer_size=5
            length,width=make_random_point_mask_for_train_for_label_mini(mean,std,\
                min_width,max_width,max_aspect_ratio,buffer_size,img_size=256)
            #
            #
            center_location_range_i=center_location_range[class_i+1]
            random_position=random.randint(center_location_range_i[0],center_location_range_i[1])
            center_row,center_clo=contour_points[random_position][0]
            #
            if random.random()<0.5:
                row_line,clo_line=length,width
            else :
                row_line,clo_line=width,length
            row1=center_row-int(round(row_line/2))
            row2=row1+row_line-1
            clo1=center_clo-int(round(clo_line/2))
            clo2=clo1+clo_line-1
            if row1<0:
                row2=row2+(0-row1)
                row1=0
            if clo1<0:
                clo2=clo2+(0-clo1)
                clo1=0
            if row2>255:
                row1=row1-(row2-255)
                row2=255
            if clo2>255:
                clo1=clo1-(clo2-255)
                clo2=255
            #
            mask = np.zeros((256, 256), np.uint8)
            mask[row1:row2+1,clo1:clo2+1] = 255
            limit=mask.copy()
            limit[img_binary==0]=0
            if limit.max()==255:
                ok=1
        mask_all[mask==255]=255
    mask=mask_all
    return mask



def make_random_point_mask_directional(img_binary):
    ret, img_binary = cv2.threshold(img_binary,127,255,cv2.THRESH_BINARY)
    #
    contour_points=get_contours_points_2(img_binary)
    #
    #
    class_n=np.random.normal(0, 2, 1)
    class_n=int(class_n)
    class_n=min(max(class_n, 1),7)
    #
    contour_points_len=len(contour_points)
    if contour_points_len<50:
        class_n=1
    elif contour_points_len<100:
        class_n=min(class_n,2)
    elif contour_points_len<150:
        class_n=min(class_n,3)
    elif contour_points_len<200:
        class_n=min(class_n,4)
    elif contour_points_len<250:
        class_n=min(class_n,5)
    #
    center_location_range={}
    center_location_range_buffer=0.5
    center_location_range_for_one=contour_points_len/class_n*(1+center_location_range_buffer)/2
    for class_i in range(class_n):
        center_location=contour_points_len/class_n*(class_i+0.5)
        a=int(center_location-center_location_range_for_one)
        a=max(0,int(center_location-center_location_range_for_one))
        b=int(center_location+center_location_range_for_one)
        b=min(contour_points_len-1,int(center_location+center_location_range_for_one))
        center_location_range.update({class_i+1:[a,b]})
    #
    if random.random()<0.9:
        only_one_mining_change_class=random.random()
    else: 
        only_one_mining_change_class=0   #
    #
    mask_all= np.zeros((256, 256), np.uint8)
    #
    mask_all_2= np.zeros((256, 256), np.uint8)
    #
    mask_all_rec= np.zeros((256, 256), np.uint8)
    # ######################################################
    for class_i in range(class_n):
        ok=0
        while not ok:
            #
            mean,std=65,50
            min_width,max_width=15,150
            max_aspect_ratio=8
            buffer_size=5
            length,width=make_random_point_mask_for_train_for_label_mini(mean,std,\
                min_width,max_width,max_aspect_ratio,buffer_size,img_size=256)
            #
            #
            center_location_range_i=center_location_range[class_i+1]
            random_position=random.randint(center_location_range_i[0],center_location_range_i[1])
            center_row,center_clo=contour_points[random_position][0]
            #
            if random.random()<0.5:
                row_line,clo_line=length,width
            else :
                row_line,clo_line=width,length
            row1=center_row-int(round(row_line/2,0))
            row2=row1+row_line-1
            clo1=center_clo-int(round(clo_line/2,0))
            clo2=clo1+clo_line-1
            if row1<0:
                row2=row2+(0-row1)
                row1=0
            if clo1<0:
                clo2=clo2+(0-clo1)
                clo1=0
            if row2>255:
                row1=row1-(row2-255)
                row2=255
            if clo2>255:
                clo1=clo1-(clo2-255)
                clo2=255
            #
            mask = np.zeros((256, 256), np.uint8)
            mask[row1:row2+1,clo1:clo2+1] = 255
            mask_2=mask.copy()
            mask_rec=mask.copy()
            limit=mask.copy()
            limit[img_binary==0]=0
            # ###########################################
            if limit.max()==255:
                #
                limit2=limit[row1:row2+1,clo1:clo2+1]
                limit3=limit2.copy()
                limit2=255-limit3
                ret, limit2 = cv2.threshold(limit2,127,255,cv2.THRESH_BINARY)
                contour_points_limit2, hierarchy = cv2.findContours(limit2, cv2.RETR_LIST, 2)
                if len(contour_points_limit2)==1:
                    ok=1
                    #
                    if only_one_mining_change_class:
                        only_one_mining_change_class_value=only_one_mining_change_class
                    else :
                        only_one_mining_change_class_value=random.random()
                    if only_one_mining_change_class_value<0.5: #
                        pass
                    else:                                      #
                        limit=mask.copy()
                        limit[img_binary==255]=0
                    mask=limit
                    if only_one_mining_change_class_value<0.5:
                        # mask_2=np.zeros((256, 256), np.uint8)
                        mask_2=mask.copy()
                    else:
                        mask_2[mask==255]=0
        mask_all[mask==255]=255
        mask_all_2[mask_2==255]=255
        mask_all_rec[mask_rec==255]=255
    mask=mask_all
    mask_2=mask_all_2
    mask_rec=mask_all_rec
    #
    if only_one_mining_change_class:
        only_one_mining_change_class=1
    # ######################################################
    # print(line_s_min)
    # ######################################################
    return mask, mask_2, mask_rec




def main():
    # data\data\01_001_002002_18_label.jpg
    label_paths = glob.glob(r"./data\data\*_label.jpg")

    out_label_mask_n_max=20
    id_n=0
    pbar=tqdm(label_paths)
    for label_path in pbar:
        id_n+=1
        label_path=label_path.replace('\\','/')
        # ######################
        # if id_n<281:
        #     continue
        # ######################
        # ######################
        # if not ('01_010_004003_18_label' in label_path):
        #     continue
        # ######################
        out_label_up_path=label_path.replace('.jpg','').replace('/data/data/','/data/data_mask/')
        if not os.path.exists(out_label_up_path):
            os.makedirs(out_label_up_path)
        out_label_up_path_2=label_path.replace('.jpg','').replace('/data/data/','/data/data_mask_2/')
        if not os.path.exists(out_label_up_path_2):
            os.makedirs(out_label_up_path_2)
        out_label_up_path_rec=label_path.replace('.jpg','').replace('/data/data/','/data/data_mask_rec/')
        if not os.path.exists(out_label_up_path_rec):
            os.makedirs(out_label_up_path_rec)
        for id_ii in range(out_label_mask_n_max):
            img_binary=cv2.imdecode(np.fromfile(label_path,dtype=np.uint8),-1)
            mask, mask_2, mask_rec=make_random_point_mask_directional(img_binary)
            #
            out_label_path=out_label_up_path+'/'+str(id_ii)+'.jpg'
            cv2.imencode('.jpg', mask)[1].tofile(out_label_path.replace('.tif','.jpg'))
            out_label_path_2=out_label_up_path_2+'/'+str(id_ii)+'.jpg'
            cv2.imencode('.jpg', mask_2)[1].tofile(out_label_path_2.replace('.tif','.jpg'))
            out_label_path_rec=out_label_up_path_rec+'/'+str(id_ii)+'.jpg'
            cv2.imencode('.jpg', mask_rec)[1].tofile(out_label_path_rec.replace('.tif','.jpg'))




if __name__=='__main__':
     
    seed=100
    np.random.seed(seed)
    random.seed(seed)
    main()