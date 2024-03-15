import argparse
import os

if __name__ == "__main__":

    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # General parameters
    parser.add_argument('--save_path', type = str, default = './models', help = 'saving path that is a folder')
    parser.add_argument('--sample_path', type = str, default = './samples', help = 'training samples path that is a folder')
    parser.add_argument('--gan_type', type = str, default = 'WGAN', help = 'the type of GAN for training')
    parser.add_argument('--multi_gpu', type = bool, default = False, help = 'nn.Parallel needs or not')
    parser.add_argument('--gpu_ids', type = str, default = "0,1", help = 'gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--cudnn_benchmark', type = bool, default = True, help = 'True for unchanged input data type')
    parser.add_argument('--checkpoint_interval', type = int, default = 1, help = 'interval between model checkpoints')
    parser.add_argument('--load_name', type = str, default = '', help = 'load model name')
    # Training parameters
    parser.add_argument('--epochs', type = int, default = 120, help = 'number of epochs of training')
    ########################################
    # parser.add_argument('--resume', action='store_true') 
    parser.add_argument('--resume', default = False) #False 
    parser.add_argument('--resume_epoch', type = int, default = 90) 
    parser.add_argument('--load_model_root', type = str, default = './DeepFillv2_Pytorch-master/log/pre_train/models') 
    parser.add_argument('--load_model_G', type = str, default = 'deepfillv2_WGAN_G_epoch120_batchsize4.pth', help = 'G  load model name')
    parser.add_argument('--load_model_D', type = str, default = 'deepfillv2_WGAN_D_epoch120_batchsize4.pth', help = 'D  load model name')
    ########################
    parser.add_argument('--batch_size', type = int, default = 4, help = 'size of the batches')
    parser.add_argument('--lr_g', type = float, default = 2e-4, help = 'Adam: learning rate')
    parser.add_argument('--lr_d', type = float, default = 2e-4, help = 'Adam: learning rate')
    parser.add_argument('--b1', type = float, default = 0.5, help = 'Adam: beta 1')
    parser.add_argument('--b2', type = float, default = 0.999, help = 'Adam: beta 2')
    parser.add_argument('--weight_decay', type = float, default = 0, help = 'Adam: weight decay')
    parser.add_argument('--lr_decrease_epoch', type = int, default = 10, help = 'lr decrease at certain epoch and its multiple')
    parser.add_argument('--lr_decrease_factor', type = float, default = 0.5, help = 'lr decrease factor, for classification default 0.1')
    ########################
    parser.add_argument('--lambda_l1', type = float, default = 100, help = 'the parameter of L1Loss')
    parser.add_argument('--lambda_perceptual', type = float, default = 0, help = 'the parameter of FML1Loss (perceptual loss)')
    parser.add_argument('--lambda_gan', type = float, default = 1, help = 'the parameter of valid loss of AdaReconL1Loss; 0 is recommended')
    ###############################################
    parser.add_argument('--lambda_segmentation', type = float, default = 0.5, help = '') #0.05-0.5
    #####################
    parser.add_argument('--num_workers', type = int, default = 8, help = 'number of cpu threads to use during batch generation')
    # Network parameters
    #############################################
    # parser.add_argument('--in_channels', type = int, default = 4, help = 'input RGB image + 1 channel mask')
    parser.add_argument('--in_channels', type = int, default = 4+1, help = 'input RGB image + 1 channel mask + 1 channel label')
    ######################
    parser.add_argument('--out_channels', type = int, default = 3, help = 'output RGB image')
    parser.add_argument('--latent_channels', type = int, default = 48, help = 'latent channels')
    parser.add_argument('--pad_type', type = str, default = 'reflect', help = 'the padding type')
    parser.add_argument('--activation', type = str, default = 'lrelu', help = 'the activation type')
    parser.add_argument('--norm', type = str, default = 'none', help = 'normalization type')
    parser.add_argument('--init_type', type = str, default = 'kaiming', help = 'the initialization type')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'the initialization gain')
    # Dataset parameters
    parser.add_argument('--baseroot', type = str, default = "./data/JPEGImages_0", help = 'the training folder')
    ##############################################
    # parser.add_argument('--mask_type', type = str, default = 'free_form', help = 'mask type')
    parser.add_argument('--mask_type', type = str, default = 'my_random_mask', help = 'mask type')#'my_random_mask' , 'my_random_mask_for_fine_tuning'
    ####################
    parser.add_argument('--imgsize', type = int, default = 256, help = 'size of image')
    parser.add_argument('--margin', type = int, default = 10, help = 'margin of image')
    parser.add_argument('--mask_num', type = int, default = 20, help = 'number of mask')
    parser.add_argument('--bbox_shape', type = int, default = 30, help = 'margin of image for bbox mask')
    parser.add_argument('--max_angle', type = int, default = 4, help = 'parameter of angle for free form mask')
    parser.add_argument('--max_len', type = int, default = 40, help = 'parameter of length for free form mask')
    parser.add_argument('--max_width', type = int, default = 10, help = 'parameter of width for free form mask')
    
    #######################################
    from datetime import datetime
    current_time = datetime.now().strftime("%Y%m%d-%H%M")
    parser.add_argument('--log_root', default = './DeepFillv2_Pytorch-master/log/%s'%current_time, help = 'save path')
    #######################################
    
    opt = parser.parse_args()
    print(opt)
    
    
    # ----------------------------------------
    #       Choose CUDA visible devices
    # ----------------------------------------
    if opt.multi_gpu == True:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    
    # Enter main function
    import trainer
    if opt.gan_type == 'WGAN':
        trainer.WGAN_trainer(opt)
    
