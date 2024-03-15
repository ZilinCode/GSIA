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
import train_dataset
import utils


#####################################
import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from make_random_point_mask import make_random_point_mask_for_train_for_label_Non_directional
def my_make_mask(img):
    mask=make_random_point_mask_for_train_for_label_Non_directional(img)
    return mask
#####################################
##########################################
from my_models.my_deeplab.deeplab import DeeplabV3
######################

def WGAN_trainer(opt):
    # ----------------------------------------
    #      Initialize training parameters
    # ----------------------------------------

    # cudnn benchmark accelerates the network
    cudnn.benchmark = opt.cudnn_benchmark

    ##################################
    # configurations
    # save_folder = opt.save_path
    # sample_folder = opt.sample_path
    # if not os.path.exists(save_folder):
    #     os.makedirs(save_folder)
    # if not os.path.exists(sample_folder):
    #     os.makedirs(sample_folder)
    if not os.path.exists(opt.log_root):
        os.makedirs(opt.log_root)
    save_folder = opt.log_root+'/'+os.path.basename(opt.save_path) 
    sample_folder = opt.log_root+'/'+os.path.basename(opt.sample_path)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not os.path.exists(sample_folder):
        os.makedirs(sample_folder)
    ###################

    # Build networks
    generator = utils.create_generator(opt)
    discriminator = utils.create_discriminator(opt)
    perceptualnet = utils.create_perceptualnet()

    # Loss functions
    L1Loss = nn.L1Loss()
    MSELoss = nn.MSELoss()
    ########################################
    deeplab = DeeplabV3()
    #########################

    # Optimizers
    optimizer_g = torch.optim.Adam(generator.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr = opt.lr_d, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)

    # Learning rate decrease
    def adjust_learning_rate(lr_in, optimizer, epoch, opt):
        """Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs"""
        lr = lr_in * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    # Save the two-stage generator model
    def save_model_generator(net, epoch, opt):
        """Save the model at "checkpoint_interval" and its multiple"""
        model_name = 'deepfillv2_WGAN_G_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)
        model_name = os.path.join(save_folder, model_name)
        if opt.multi_gpu == True:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.module.state_dict(), model_name)
                print('The trained model is successfully saved at epoch %d' % (epoch))
        else:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.state_dict(), model_name)
                print('The trained model is successfully saved at epoch %d' % (epoch))
                
    # Save the dicriminator model
    def save_model_discriminator(net, epoch, opt):
        """Save the model at "checkpoint_interval" and its multiple"""
        model_name = 'deepfillv2_WGAN_D_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)
        model_name = os.path.join(save_folder, model_name)
        if opt.multi_gpu == True:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.module.state_dict(), model_name)
                print('The trained model is successfully saved at epoch %d' % (epoch))
        else:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.state_dict(), model_name)
                print('The trained model is successfully saved at epoch %d' % (epoch))
    
    # load the model
    ########################################
    # def load_model(net, epoch, opt, type='G'):
    def load_model(net, model_path):
        # """Save the model at "checkpoint_interval" and its multiple"""
        # if type == 'G':
        #     model_name = 'deepfillv2_WGAN_G_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)
        # else:
        #     model_name = 'deepfillv2_WGAN_D_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)
        # model_name = os.path.join(save_folder, model_name)
        # pretrained_dict = torch.load(model_name)
        pretrained_dict = torch.load(model_path)
        net.load_state_dict(pretrained_dict)

    if opt.resume:
        # load_model(generator, opt.resume_epoch, opt, type='G')
        # load_model(discriminator, opt.resume_epoch, opt, type='D')
        model_path_G=opt.load_model_root+'/'+opt.load_model_G
        load_model(generator, model_path_G)
        model_path_D=opt.load_model_root+'/'+opt.load_model_D
        load_model(discriminator, model_path_D)
        print('--------------------Pretrained Models are Loaded--------------------')
        
        # Learning rate decrease 
        adjust_learning_rate(opt.lr_g, optimizer_g, (opt.resume_epoch + 1), opt)
        adjust_learning_rate(opt.lr_d, optimizer_d, (opt.resume_epoch + 1), opt)
    ########################################
        
    # To device
    if opt.multi_gpu == True:
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)
        perceptualnet = nn.DataParallel(perceptualnet)
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        perceptualnet = perceptualnet.cuda()
    else:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        perceptualnet = perceptualnet.cuda()
    
    # ----------------------------------------
    #       Initialize training dataset
    # ----------------------------------------

    # Define the dataset
    trainset = train_dataset.InpaintDataset(opt)
    print('The overall number of images equals to %d' % len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = False, num_workers = opt.num_workers, pin_memory = True, drop_last=True)
    
    # ----------------------------------------
    #            Training
    # ----------------------------------------

    # Initialize start time
    prev_time = time.time()
    
    # Tensor type
    Tensor = torch.cuda.FloatTensor
    
    # Training loop
    for epoch in range(opt.resume_epoch, opt.epochs):
        ######################################
        # for batch_idx, (img, height, width) in enumerate(dataloader):
        for batch_idx, (img, height, width, label) in enumerate(dataloader):
            ###################

            img = img.cuda()
            ############################################
            # # set the same free form masks for each batch
            # mask = torch.empty(img.shape[0], 1, img.shape[2], img.shape[3]).cuda()
            # for i in range(opt.batch_size):
            #     mask[i] = torch.from_numpy(train_dataset.InpaintDataset.random_ff_mask(
            #                                     shape=(height[0], width[0])).astype(np.float32)).cuda()
            if opt.mask_type=='free_form':
                # set the same free form masks for each batch
                mask = torch.empty(img.shape[0], 1, img.shape[2], img.shape[3]).cuda()
                for i in range(opt.batch_size):
                    mask[i] = torch.from_numpy(train_dataset.InpaintDataset.random_ff_mask(
                                                    shape=(height[0], width[0])).astype(np.float32)).cuda()
            elif opt.mask_type=='my_random_mask':
                mask = torch.empty(img.shape[0], 1, img.shape[2], img.shape[3]).cuda()
                for i in range(opt.batch_size):
                    mask_i = train_dataset.InpaintDataset.random_ff_mask(
                                                    shape=(height[0], width[0])).astype(np.float32)
                    
                    mask_i_2 = make_random_point_mask_for_train_for_label_Non_directional(label[i,0].numpy() * 255)
                    
                    mask_i_2 = mask_i_2/255.0
                    mask_i[0][mask_i_2==1]=1
                    mask[i] = torch.from_numpy(mask_i.astype(np.float32)).cuda()
            ##########################
            
            #
            label = label.float().cuda()
            ############################
            
            # LSGAN vectors
            valid = Tensor(np.ones((img.shape[0], 1, height[0]//32, width[0]//32)))
            fake = Tensor(np.zeros((img.shape[0], 1, height[0]//32, width[0]//32)))
            zero = Tensor(np.zeros((img.shape[0], 1, height[0]//32, width[0]//32)))

            ### Train Discriminator
            optimizer_d.zero_grad()

            ############################################
            # Generator output
            # first_out, second_out = generator(img, mask)
            first_out, second_out = generator(img, mask ,label)
            #####################

            # forward propagation
            first_out_wholeimg = img * (1 - mask) + first_out * mask        # in range [0, 1]
            second_out_wholeimg = img * (1 - mask) + second_out * mask      # in range [0, 1]

            ############################################
            # Fake samples
            # fake_scalar = discriminator(second_out_wholeimg.detach(), mask) 
            fake_scalar = discriminator(second_out_wholeimg.detach(), mask, label) 
            # True samples
            # true_scalar = discriminator(img, mask)
            true_scalar = discriminator(img, mask, label)
            #######################
            
            # Loss and optimize
            loss_fake = -torch.mean(torch.min(zero, -valid-fake_scalar))
            loss_true = -torch.mean(torch.min(zero, -valid+true_scalar))
            # Overall Loss and optimize
            loss_D = 0.5 * (loss_fake + loss_true)
            loss_D.backward()
            optimizer_d.step()

            ### Train Generator
            optimizer_g.zero_grad()

            # L1 Loss
            first_L1Loss = (first_out - img).abs().mean()
            second_L1Loss = (second_out - img).abs().mean()
            
            ############################################
            # GAN Loss
            # fake_scalar = discriminator(second_out_wholeimg, mask)
            fake_scalar = discriminator(second_out_wholeimg, mask, label)
            #################
            #######################################
            # GAN_Loss = -torch.mean(fake_scalar) #
            GAN_Loss = -torch.mean(torch.min(zero, fake_scalar)) #
            #################

            # Get the deep semantic feature maps, and compute Perceptual Loss
            if opt.lambda_perceptual!=0:
                img_featuremaps = perceptualnet(img)                          # feature maps
                second_out_featuremaps = perceptualnet(second_out)
                second_PerceptualLoss = L1Loss(second_out_featuremaps, img_featuremaps)
                   
            ########################################
            #segmentation_loss
            label_predict = deeplab.detect_image(second_out_wholeimg)
            segmentation_loss = L1Loss(label_predict * mask, label * mask)
            
            # Compute losses
            # loss = opt.lambda_l1 * first_L1Loss + opt.lambda_l1 * second_L1Loss + \
            #        opt.lambda_perceptual * second_PerceptualLoss + opt.lambda_gan * GAN_Loss
            loss = opt.lambda_l1 * (first_L1Loss+second_L1Loss) + opt.lambda_gan * GAN_Loss
            if opt.lambda_perceptual!=0:
                loss = loss + opt.lambda_perceptual * second_PerceptualLoss
            lambda_segmentation = max(opt.lambda_segmentation * (min(epoch, 100) /100.), 0.05) 
            loss =  (1-lambda_segmentation) * loss + \
                    lambda_segmentation * segmentation_loss
            #######################################
        
            loss.backward()
            optimizer_g.step()

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + batch_idx
            batches_left = opt.epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds = batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [first Mask L1 Loss: %.5f] [second Mask L1 Loss: %.5f]" %
                ((epoch + 1), opt.epochs, batch_idx, len(dataloader), first_L1Loss.item(), second_L1Loss.item()))
            print("\r[D Loss: %.5f] [G Loss: %.5f] [Seg Loss: %.5f] time_left: %s" %
                (loss_D.item(), GAN_Loss.item(), segmentation_loss.item(), time_left))
            
            masked_img = img * (1 - mask) + mask
            mask = torch.cat((mask, mask, mask), 1)
            if (batch_idx + 1) % 40 == 0:
                img_list = [img, mask, masked_img, first_out, second_out]
                name_list = ['gt', 'mask', 'masked_img', 'first_out', 'second_out']
                utils.save_sample_png(sample_folder = sample_folder, sample_name = 'epoch%d' % (epoch + 1), img_list = img_list, name_list = name_list, pixel_max_cnt = 255)

        # Learning rate decrease
        adjust_learning_rate(opt.lr_g, optimizer_g, (epoch + 1), opt)
        adjust_learning_rate(opt.lr_d, optimizer_d, (epoch + 1), opt)

        # Save the model
        save_model_generator(generator, (epoch + 1), opt)
        save_model_discriminator(discriminator, (epoch + 1), opt)

        ### Sample data every epoch
        if (epoch + 1) % 1 == 0:
            img_list = [img, mask, masked_img, first_out, second_out]
            name_list = ['gt', 'mask', 'masked_img', 'first_out', 'second_out']
            utils.save_sample_png(sample_folder = sample_folder, sample_name = 'epoch%d' % (epoch + 1), img_list = img_list, name_list = name_list, pixel_max_cnt = 255)
