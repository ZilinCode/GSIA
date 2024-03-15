from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self,  parser):
        parser = BaseOptions.initialize(self, parser)

        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of the test examples')
        parser.add_argument('--results_dir', type=str, default='./data_test/data_result', help='saves results here')
        parser.add_argument('--how_many', type=int, default=None, help='how many test images to run')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test')
        parser.add_argument('--nsampling', type=int, default=1, help='ramplimg # times for each images')
        # parser.add_argument('--save_number', type=int, default=10, help='choice # reasonable results based on the discriminator score')
        
        resume_uppath='./Pluralistic-Inpainting-master/checkpoints/pre_train'
        parser.add_argument('--resume_uppath', default=resume_uppath, help='resume model path') 
        parser.add_argument('--img_file_test', type=str, default='./data_test/data', help='testing dataset')
        
        parser.add_argument('--maskroot_test', type=str, default="./data_test/data_mask", help='the mask folder')

        self.isTrain = False

        return parser
