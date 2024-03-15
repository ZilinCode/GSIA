import os
import time
import datetime
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import network
import test_dataset
import utils

##############################################
import glob
from tqdm import tqdm
####################
##########################################
from my_models.my_deeplab.deeplab import DeeplabV3
######################
#############################################
import train_dataset
#####################

def WGAN_tester(opt):
    
    ##############################################
    # def load_model_generator(net, epoch, opt):
    def load_model_generator(generator, opt):
        # # model_name = 'deepfillv2_WGAN_G_epoch%d_batchsize%d.pth' % (epoch, 4)
        # # model_name = os.path.join('pretrained_model', model_name)
        # # pretrained_dict = torch.load(model_name)
        # #
        # pretrained_dict = torch.load(opt.load_model_G_path)
        # generator.load_state_dict(pretrained_dict)
        #
        pretrained_dict = torch.load(opt.load_model_G_path)
        try:
            generator.load_state_dict(pretrained_dict)
        except:
            model_dict = generator.state_dict()
            try:
                pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict}
                generator.load_state_dict(pretrained_dict)
                print('Pretrained network has excessive layers; Only loading layers that are used')
            except:
                print('Pretrained network has fewer layers; The following are not initialized:')
                not_initialized = set()
                for k, v in pretrained_dict.items():
                    if v.size() == model_dict[k].size():
                        model_dict[k] = v

                for k, v in model_dict.items():
                    if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                        not_initialized.add(k.split('.')[0])
                print(sorted(not_initialized))
                generator.load_state_dict(model_dict)
        
    ######################

    # ----------------------------------------
    #      Initialize training parameters
    # ----------------------------------------

    # configurations
    ##################################################
    if not os.path.exists(opt.results_path):
        os.makedirs(opt.results_path)
    log_all_txt_path = opt.results_path+'/all_log.txt'
    with open(log_all_txt_path, "a") as log_file:
        for arg in vars(opt):
            print(format(arg, '<20'), format(str(getattr(opt, arg)), '<') ,file=log_file)   
        log_file.write('\n')
    ###################

    # Build networks
    generator = utils.create_generator(opt).eval()
    print('-------------------------Loading Pretrained Model-------------------------')
    ########################################
    # load_model_generator(generator, opt.epoch, opt)
    load_model_generator(generator, opt)
    #####################
    print('-------------------------Pretrained Model Loaded-------------------------')
    
    deeplab = DeeplabV3()

    # To device
    generator = generator.cuda()
    
    # ----------------------------------------
    #       Initialize training dataset
    # ----------------------------------------

    # # Define the dataset
    # trainset = test_dataset.InpaintDataset(opt)
    # print('The overall number of images equals to %d' % len(trainset))

    # # Define the dataloader
    # dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = False, num_workers = opt.num_workers, pin_memory = True)
    
    # ----------------------------------------
    #            Testing
    # ----------------------------------------
    # Testing loop
    # for batch_idx, (img, mask) in enumerate(dataloader):
    img_paths=glob.glob(opt.baseroot+'/*.jpg')
    
    for img_path in tqdm(img_paths):
        img = cv2.imread(img_path) #
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #
        # img_copy = img.copy()
        img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).contiguous() #
        
        label_original_path=img_path.replace('JPEGImages_0','SegmentationClass').replace('.jpg','.png')
        label_original = cv2.imread(label_original_path)[:,:,0] #
        # label_original_copy = label_original.copy()
        label_original = torch.from_numpy(label_original.astype(np.float32) ).unsqueeze(0).unsqueeze(0).contiguous() #
        
        
        
        imgname = os.path.basename(img_path)
        mask_up_path = os.path.join(opt.baseroot_mask, imgname.replace('.jpg','_label'))
        label_up_path = os.path.join(opt.baseroot_label, imgname.replace('.jpg','_label'))
        mask_paths=glob.glob(mask_up_path+'/*.png') #
        for mask_path in mask_paths :
            mask = cv2.imread(mask_path)[:, :, 0] #
            
            ################################事实证明，mask_new_new2+buffer效果好
            mask_copy=mask.copy()*255
            #
            kernel = np.ones((9, 9), np.uint8)
            #
            mask_2 = cv2.dilate(mask_copy, kernel)
            mask=mask_2/255.0
            ###################
            
            # mask = torch.from_numpy(mask.astype(np.float32) / 255.0).unsqueeze(0).contiguous()
            mask = torch.from_numpy(mask.astype(np.float32) ).unsqueeze(0).unsqueeze(0).contiguous() #
            maskname = os.path.basename(mask_path)
            label_path = os.path.join(label_up_path, maskname.replace('.png','.jpg'))
            label = cv2.imread(label_path)[:, :, 0] 
            ret, label = cv2.threshold(label, 127, 255, cv2.THRESH_BINARY)
            label = torch.from_numpy(label.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).contiguous() 
            
            mask, label, img, label_original = mask.cuda(), label.cuda(), img.cuda(), label_original.cuda()
            
            #########################################
            if opt.if_add_free_form:
                # set the same free form masks for each batch
                mask_add_free_form = mask.clone().detach()
                free_form = torch.from_numpy(train_dataset.InpaintDataset.random_ff_mask(
                                                    shape=(mask.shape[2],mask.shape[3])).astype(np.float32)).cuda()
                mask_add_free_form[0][free_form==1]=1
            ###################
        
            # img = img.cuda()
            # mask = mask.cuda()

            # Generator output
            with torch.no_grad():
                #######################################
                # first_out, second_out = generator(img, mask)
                if opt.if_add_free_form:
                    first_out, second_out = generator(img, mask_add_free_form, label)
                else :
                    first_out, second_out = generator(img, mask, label)
                ##################

            # # forward propagation
            # first_out_wholeimg = img * (1 - mask) + first_out * mask        # in range [0, 1]
            if opt.if_add_free_form:
                second_out_wholeimg = img * (1 - mask_add_free_form) + second_out * mask_add_free_form      # in range [0, 1]
            else :
                second_out_wholeimg = img * (1 - mask) + second_out * mask

            #segmentation_loss
            label_predict = deeplab.detect_image(second_out_wholeimg)
            
            img_list = [img, 
                        img * (1-mask), 
                        first_out, 
                        second_out, 
                        second_out_wholeimg,
                        label_original,
                        label,
                        label * (1-mask),
                        label * (1-mask) + label_predict * mask]
            name_list = ['image', 
                        'image_corrupted', 
                        'image_out_1', 
                        'image_out_2', 
                        'image_composite',
                        'label_original',
                        'label', 
                        'label_corrupted',
                        'label_predict']
            if opt.if_add_free_form:
                img_list.append(img * (1-mask_add_free_form))
                name_list.append('image_corrupted_2')
            # utils.save_sample_png(sample_folder = opt.results_path, sample_name = '%d' % (batch_idx + 1), img_list = img_list, name_list = name_list, pixel_max_cnt = 255)
            sample_folder = os.path.join(opt.results_path+'/label_guided_ICGAN_results', imgname.replace('.jpg',''))+'/'+maskname.replace('.png','')
            utils.my_save_sample_png(sample_folder = sample_folder, img_list = img_list, name_list = name_list, pixel_max_cnt = 255)
            # print('----------------------batch_idx%d' % (batch_idx + 1) + ' has been finished----------------------')
            ###########################