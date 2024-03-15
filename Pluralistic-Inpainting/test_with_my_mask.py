from options import test_options
from dataloader import data_loader
from model import create_model
from util import visualizer
from itertools import islice

import torch
# from dataloader.my_dataset.dataset import InpaintDataset
from tqdm import tqdm
import glob,os
import numpy as np
from PIL import Image
import cv2
import random
##################################################


if __name__=='__main__':
    # get testing options
    opt = test_options.TestOptions().parse()
    # # create a model
    model = create_model(opt)
    model.eval()
    # create a visualizer
    visualizer = visualizer.Visualizer(opt)

    #################################################
    img_paths=glob.glob(opt.img_file_test+'/*.jpg')
    if not opt.how_many==None:
        img_paths=islice(img_paths, opt.how_many)
    
    for imgpath in tqdm(img_paths):
        # image
        imgname = os.path.basename(imgpath)                                 # name of one image
        img = Image.open(imgpath).convert('L')                              # read one image (RGB)
        img = np.array(img)                                                 # read one image
        _, img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        img = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).contiguous()
        # mask
        mask_up_path = os.path.join(opt.maskroot_test, imgname.replace('.jpg',''))
        maskpaths=glob.glob(mask_up_path+'/*.jpg')
        for maskpath in maskpaths:
            mask = Image.open(maskpath).convert('L')
            mask = np.array(mask)                                                 # read one image
            _, mask = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
            mask = torch.from_numpy(mask.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).contiguous()
            # To device
            img = img.cuda()                                        # out: [B, 1, 256, 256]
            mask = mask.cuda()                                                  # out: [B, 1, 256, 256]
            mask_name=imgname.replace('.jpg','') +'/'+ os.path.basename(maskpath).replace('.jpg','')

            # # for i, data in enumerate(islice(dataset, opt.how_many)):
            # pbar=islice(tqdm(dataset), opt.how_many)
            # for batch_idx, (img, mask, img_name) in enumerate(pbar):
            mask = 1 - mask  #
            data={'img': img, 'mask': mask, 'img_path': mask_name }
            ####################
            model.set_input(data)
            # # model.test()
            # model.my_test()
            model.my_test2()