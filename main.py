import argparse
import os
from dataset import get_loader
from dataset import get_loader_target
from solver import Solver


def main(config):
    if config.mode == 'train':
        train_loader = get_loader(config.train_path, config.label_path, config.img_size, config.batch_size,
                                  filename=config.train_file,itertype='multi', num_thread=config.num_thread)
        target_loader = get_loader_target(config.target_path, config.img_size, config.batch_size, 
                                    filename=config.target_file,num_thread=config.num_thread)

        if config.val:
            val_loader = get_loader(config.val_path, config.val_label, config.img_size, config.batch_size,
                                    filename=config.val_file,num_thread=config.num_thread)
        run = 0
        while os.path.exists("%s/run-%d" % (config.save_fold, run)): run += 1
        os.mkdir("%s/run-%d" % (config.save_fold, run))
        os.mkdir("%s/run-%d/logs" % (config.save_fold, run))
        # os.mkdir("%s/run-%d/images" % (config.save_fold, run))
        os.mkdir("%s/run-%d/models" % (config.save_fold, run))
        os.mkdir("%s/run-%d/tensorboards" % (config.save_fold, run))
        config.save_fold = "%s/run-%d" % (config.save_fold, run)
        if config.val:
            train = Solver(train_loader, target_loader, val_loader, None, config)
        else:
            train = Solver(train_loader, target_loader, None, None, config)
        train.train_advent()
        # train.train_old()

    elif config.mode == 'test':
        test_loader = get_loader(config.test_path, config.test_label, config.img_size, config.batch_size, mode='test',
                                 filename=config.test_file, num_thread=config.num_thread)
        if not os.path.exists(config.test_fold): os.mkdir(config.test_fold)
        if not os.path.exists(config.test_map_fold): os.mkdir(config.test_map_fold)#设置saliency map的存储位置
        test = Solver(None, None, None, test_loader, config)
        test.test(100, use_crf=config.use_crf)
    else:
        raise IOError("illegal input!!!")


if __name__ == '__main__':
    # data_root = os.path.join(os.path.expanduser('~'), 'data')
    data_root = '/data0/liumengmeng/datasets'
    cg2_root = '/data1/liumengmeng/CG2'
    vgg_path = './weights/vgg16_feat.pth'
    
    # # -----MSRA-B dataset-----
    # image_path = os.path.join(data_root, 'MSRA-B/img')
    # label_path = os.path.join(data_root, 'MSRA-B/gt')
    # train_file = os.path.join(data_root, 'MSRA-B/train_id.txt')
    # valid_file = os.path.join(data_root, 'MSRA-B/val_id.txt')
    # test_file = os.path.join(data_root, 'MSRA-B/test_id.txt')

    # -----DUTS dataset-----
    # image_path = os.path.join(data_root, 'DUTS/imgs')
    # label_path = os.path.join(data_root, 'DUTS/gt')
    # train_file = os.path.join(data_root, 'DUTS/ImageSets/train_id.txt')
    # #valid_file = os.path.join(data_root, 'DUTS/ImageSets/val_id.txt')
    # test_file = os.path.join(data_root, 'DUTS/ImageSets/test_id.txt')

    # # -----my Composite Graph ONE dataset-----
    # image_path = os.path.join(data_root, 'CG/img')
    # label_path = os.path.join(data_root, 'CG/gt')
    # train_file = os.path.join(data_root, 'CG/id/train_id.txt')
    # valid_file = os.path.join(data_root, 'DUTS/ImageSets/test_id.txt')
    # image_path_2 = os.path.join(data_root, 'DUTS/imgs')
    # label_path_2 = os.path.join(data_root, 'DUTS/gt')
    # test_file = os.path.join(data_root, 'DUTS/ImageSets/test_id.txt')
    # target_path = os.path.join(data_root, 'DUTS/imgs')
    # target_file = os.path.join(data_root, 'DUTS/ImageSets/train_id.txt')

    # -----my Composite Graph TWO  dataset-----
    image_path = os.path.join(cg2_root, 'img')
    label_path = os.path.join(cg2_root, 'gt')
    train_file = os.path.join(cg2_root, 'id/train_id.txt')
    valid_file = os.path.join(data_root, 'DUTS/ImageSets/test_id.txt')
    image_path_2 = os.path.join(data_root, 'DUTS/imgs')
    label_path_2 = os.path.join(data_root, 'DUTS/gt')
    test_file = os.path.join(data_root, 'DUTS/ImageSets/test_id.txt')
    target_path = os.path.join(data_root, 'DUTS/imgs')
    target_file = os.path.join(data_root, 'DUTS/ImageSets/train_id.txt')


    parser = argparse.ArgumentParser()

    # Hyper-parameters
    parser.add_argument('--n_color', type=int, default=3)
    parser.add_argument('--img_size', type=int, default=320)  # 256
    parser.add_argument('--lr', type=float, default=1e-4)#-4 #1e-6
    parser.add_argument('--lr_d', type=float, default=1e-4)#-4 #1e-6
    parser.add_argument('--LAMBDA_ADV_MAIN', type=float, default=0.01) # loss_adv
    parser.add_argument('--clip_gradient', type=float, default=1.0)
    parser.add_argument('--cuda', type=bool, default=True)

    # Training settings
    parser.add_argument('--vgg', type=str, default=vgg_path)
    parser.add_argument('--train_path', type=str, default=image_path)
    parser.add_argument('--label_path', type=str, default=label_path)
    parser.add_argument('--train_file', type=str, default=train_file)
    parser.add_argument('--target_path', type=str, default=target_path)# target domain
    parser.add_argument('--target_file', type=str, default=target_file)# the id of target domain
    parser.add_argument('--early_stop', type=int, default=15000)# iteration number
    parser.add_argument('--iter_save', type=int, default=500)# save model every  epoch
    parser.add_argument('--iter_val', type=int, default=500) # equals  epoch
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1)  # 8 # 16
    parser.add_argument('--val', type=bool, default=True)
    parser.add_argument('--val_path', type=str, default=image_path_2)
    parser.add_argument('--val_label', type=str, default=label_path_2)
    parser.add_argument('--val_file', type=str, default=valid_file)
    parser.add_argument('--num_thread', type=int, default=4)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--save_fold', type=str, default='/data1/liumengmeng/DSS/DSS-results')
    parser.add_argument('--epoch_val', type=int, default=10)
    parser.add_argument('--epoch_save', type=int, default=10)
    parser.add_argument('--epoch_show', type=int, default=1)
    parser.add_argument('--pre_trained', type=str, default=None)

    # Testing settings
    parser.add_argument('--test_path', type=str, default=image_path_2)
    parser.add_argument('--test_label', type=str, default=label_path_2)
    parser.add_argument('--test_file', type=str, default=test_file)
    parser.add_argument('--model', type=str, default='./weights/final.pth')
    parser.add_argument('--test_fold', type=str, default='/data1/liumengmeng/DSS/DSS-results/test')
    parser.add_argument('--test_map_fold', type=str, default='/data1/liumengmeng/DSS/DSS-results/test/test_map')
    parser.add_argument('--use_crf', type=bool, default=False)

    # Misc
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--visdom', type=bool, default=False)

    config = parser.parse_args()
    if not os.path.exists(config.save_fold): os.mkdir(config.save_fold)
    main(config)
