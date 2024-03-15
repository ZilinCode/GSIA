import argparse
import os

if __name__ == "__main__":

    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # General parameters
    #########################################
    # parser.add_argument('--results_path', type = str, default = './results', help = 'testing samples path that is a folder')
    from datetime import datetime
    current_time = datetime.now().strftime("%Y%m%d-%H%M")
    parser.add_argument('--results_path', type = str, default = './DeepFillv2_Pytorch-master/results/%s'%current_time, help = 'testing samples path that is a folder')
    #####################
    parser.add_argument('--gan_type', type = str, default = 'WGAN', help = 'the type of GAN for training')
    parser.add_argument('--gpu_ids', type = str, default = "0", help = 'gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--cudnn_benchmark', type = bool, default = True, help = 'True for unchanged input data type')
    # Training parameters
    parser.add_argument('--epoch', type = int, default = 1, help = 'number of epochs of training')
    parser.add_argument('--batch_size', type = int, default = 1, help = 'size of the batches')
    parser.add_argument('--num_workers', type = int, default = 8, help = 'number of cpu threads to use during batch generation')
    # Network parameters
    parser.add_argument('--in_channels', type = int, default = 4+1, help = 'input RGB image + 1 channel mask')
    parser.add_argument('--out_channels', type = int, default = 3, help = 'output RGB image')
    parser.add_argument('--latent_channels', type = int, default = 48, help = 'latent channels')
    parser.add_argument('--pad_type', type = str, default = 'reflect', help = 'the padding type')
    parser.add_argument('--activation', type = str, default = 'lrelu', help = 'the activation type')
    parser.add_argument('--norm', type = str, default = 'none', help = 'normalization type')
    parser.add_argument('--init_type', type = str, default = 'kaiming', help = 'the initialization type')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'the initialization gain')
    # Dataset parameters
    parser.add_argument('--baseroot', type = str, default = './data_test/JPEGImages_0')
    parser.add_argument('--baseroot_mask', type = str, default = './data_test/mask_new_new2')
    ###################################
    parser.add_argument('--baseroot_label', type = str, default = './data_test/my_results_new2')
    #####################
    ###################################
    load_model_G_path='./DeepFillv2_Pytorch-master/log/pre_train/models'+\
        '/deepfillv2_WGAN_G_epoch120_batchsize4.pth'
    parser.add_argument('--load_model_G_path', type = str, default = load_model_G_path)
    #####################
    
    #####################################
    parser.add_argument('--if_add_free_form', default = True, help ='add free_form')
    ####################
    
    opt = parser.parse_args()
    
    
    # ----------------------------------------
    #       Choose CUDA visible devices
    # ----------------------------------------
    
    # Enter main function
    import tester
    if opt.gan_type == 'WGAN':
        tester.WGAN_tester(opt)
    
