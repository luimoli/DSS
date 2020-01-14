import argparse
import os
from dataset import get_loader
from dataset import get_loader_target
from solver import Solver


def main(config):
    if config.mode == 'train':
        train_loader = get_loader(config.train_path, config.label_path, config.img_size, config.batch_size,
                                  filename=config.train_file,itertype='multi', num_thread=config.num_thread)
        target_loader = get_loader_target(config.target_image_path, config.img_size, config.batch_size, 
                                    filename=config.target_file,num_thread=config.num_thread) #target without label
        # target_loader = get_loader(config.target_image_path,config.target_label_path, config.img_size, config.batch_size, 
        #                             filename=config.target_file,itertype='multi', num_thread=config.num_thread) #target with label

        if config.val:
            val_loader = get_loader(config.val_path, config.val_label, config.img_size, config.val_batch_size,
                                    filename=config.val_file, valsize='original', num_thread=config.num_thread)
            if not os.path.exists(config.val_fold): os.mkdir(config.val_fold)
        run = 0
        while os.path.exists("%s/run-%d" % (config.save_fold, run)): run += 1
        os.mkdir("%s/run-%d" % (config.save_fold, run))
        os.mkdir("%s/run-%d/logs" % (config.save_fold, run))
        # os.mkdir("%s/run-%d/images" % (config.save_fold, run))
        os.mkdir("%s/run-%d/models" % (config.save_fold, run))
        os.mkdir("%s/run-%d/tensorboards" % (config.save_fold, run))
        config.save_fold = "%s/run-%d" % (config.save_fold, run)
        # if not os.path.exists("%s/val-%d" % (config.val_fold, run)): os.mkdir("%s/val-%d" % (config.val_fold, run))
        # config.val_fold = "%s/val-%d" % (config.val_fold, run)
        if config.val:
            train = Solver(train_loader, target_loader, val_loader, None, config)
        else:
            train = Solver(train_loader, target_loader, None, None, config)
        train.train()
        # train.train_old()

    elif config.mode == 'test':
        test_loader = get_loader(config.test_path, config.test_label, config.img_size, config.batch_size, mode='test',
                                 filename=config.test_file, num_thread=config.num_thread)
                                 
        if not os.path.exists(config.test_fold): os.mkdir(config.test_fold)
        te = 0
        while os.path.exists("%s/test-%d" % (config.test_fold, te)): te += 1
        os.mkdir("%s/test-%d" % (config.test_fold, te))
        os.mkdir("%s/test-%d/test_map" % (config.test_fold, te))
        config.test_map_fold = "%s/test-%d/test_map" % (config.test_fold, te)
        config.test_fold = "%s/test-%d" % (config.test_fold, te)

        test = Solver(None, None, None, test_loader, config)
        test.test(100, use_crf=config.use_crf)
        # test.test_bg()
    else:
        raise IOError("illegal input!!!")


if __name__ == '__main__':
    # data_root = os.path.join(os.path.expanduser('~'), 'data')
    data_root = '/data1/liumengmeng/dataset'
    # cg2_root = '/data1/liumengmeng/dataset/CG2'
    # cg3tr_root = '/data1/liumengmeng/CG3-TR'
    # cg3te_root = '/data1/liumengmeng/CG3-TE'
    cg4_root = '/data1/liumengmeng/CG4'
    salicon_root = '/data1/liumengmeng/dataset/SALICON'
    soc_root = '/data1/liumengmeng/dataset/SOC'
    vgg_path = './weights/vgg16_feat.pth'
    
    # # -----MSRA-B dataset-----
    # image_path = os.path.join(data_root, 'MSRA-B/img')
    # label_path = os.path.join(data_root, 'MSRA-B/gt')
    # train_file = os.path.join(data_root, 'MSRA-B/train_id.txt')
    # valid_file = os.path.join(data_root, 'MSRA-B/val_id.txt')
    # test_file = os.path.join(data_root, 'MSRA-B/test_id.txt')

    # # -----DUTS dataset-----
    # image_path = os.path.join(data_root, 'DUTS/imgs')
    # label_path = os.path.join(data_root, 'DUTS/gt')
    # train_file = os.path.join(data_root, 'DUTS/ImageSets/train_id.txt')
    # valid_file = os.path.join(data_root, 'DUTS/ImageSets/test_id.txt')
    # # test_file = os.path.join(data_root, 'DUTS/ImageSets/test_id.txt')
    # val_image_path = os.path.join(data_root, 'DUTS/imgs')
    # val_label_path = os.path.join(data_root, 'DUTS/gt')

    # -----CG2 dataset-----
    # image_path = os.path.join(cg2_root, 'imgs')
    # label_path = os.path.join(cg2_root, 'gt')
    # train_file = os.path.join(cg2_root, 'ImageSets/train_id.txt')
    # valid_file = os.path.join(data_root, 'DUTS/ImageSets/test_id.txt')
    # val_image_path = os.path.join(data_root, 'DUTS/imgs')
    # val_label_path = os.path.join(data_root, 'DUTS/gt')
    # test_file = os.path.join(data_root, 'DUTS/ImageSets/test_id.txt')
    # target_path = os.path.join(data_root, 'DUTS/imgs')
    # target_file = os.path.join(data_root, 'DUTS/ImageSets/train_id.txt')

    # # ------source:CG2  target:DUTS  test/val:DUTS-TEST-----
    # image_path = os.path.join(cg2_root, 'imgs')
    # label_path = os.path.join(cg2_root, 'gt')
    # train_file = os.path.join(cg2_root, 'id/train_id.txt')
    # valid_file = os.path.join(data_root, 'DUTS/ImageSets/test_id.txt')
    # image_path_2 = os.path.join(data_root, 'DUTS/imgs')
    # label_path_2 = os.path.join(data_root, 'DUTS/gt')
    # test_file = os.path.join(data_root, 'DUTS/ImageSets/test_id.txt')
    # target_path = os.path.join(data_root, 'DUTS/imgs')
    # target_file = os.path.join(data_root, 'DUTS/ImageSets/train_id.txt')

    # # ------source:CG2  target:SALICON  test/val:DUTS-TEST -----
    # image_path = os.path.join(cg2_root, 'img')
    # label_path = os.path.join(cg2_root, 'gt')
    # train_file = os.path.join(cg2_root, 'id/train_id.txt')
    # valid_file = os.path.join(data_root, 'DUTS/ImageSets/test_id.txt')
    # val_image_path = os.path.join(data_root, 'DUTS/imgs')
    # val_label_path = os.path.join(data_root, 'DUTS/gt')
    # test_file = os.path.join(data_root, 'DUTS/ImageSets/test_id.txt')
    # target_image_path = os.path.join(salicon_root, 'img_tr')
    # target_label_path = os.path.join(salicon_root, 'gt_tr')
    # target_file = os.path.join(salicon_root, 'id/train_id.txt')

    # # ------SOC person / none person -----
    # image_path = os.path.join(soc_root, 'img')
    # label_path = os.path.join(soc_root, 'gt')
    # train_file = os.path.join(soc_root, 'id/person.txt')
    # valid_file = os.path.join(soc_root, 'id/person_none.txt')
    # val_image_path = os.path.join(soc_root, 'img')
    # val_label_path = os.path.join(soc_root, 'gt')
    # test_file = os.path.join(soc_root, 'id/person_none.txt')
    target_image_path = os.path.join(salicon_root, 'img_tr')
    target_label_path = os.path.join(salicon_root, 'gt_tr')
    target_file = os.path.join(salicon_root, 'id/train_id.txt')

    # #-------------CG3-TR  CG3-TE----------------------------
    # image_path = os.path.join(cg3tr_root, 'img')
    # label_path = os.path.join(cg3tr_root, 'gt')
    # train_file = os.path.join(cg3tr_root, 'ImageSets/5000_id.txt')
    # valid_file = os.path.join(cg3te_root, 'ImageSets/5000_id.txt')
    # val_image_path = os.path.join(cg3te_root, 'img')
    # val_label_path = os.path.join(cg3te_root, 'gt')
    # test_file = os.path.join(cg3te_root, 'ImageSets/5000_id.txt')
    # # test_image_path = os.path.join(cg3te_root, 'img')
    # # test_label_path = os.path.join(cg3te_root, 'gt')
    # # pos = test_label_path.find('gt') + 3

    # #--------------CG4--------------------------------------
    image_path = os.path.join(cg4_root, 'img')
    label_path = os.path.join(cg4_root, 'gt')
    train_file = os.path.join(cg4_root, 'ImageSets/total_id.txt')
    # val_image_path = os.path.join(data_root,'HKU-IS/imgs')
    # val_label_path = os.path.join(data_root,'HKU-IS/gt')
    # valid_file = os.path.join(data_root,'HKU-IS/ImageSets/total_id.txt')
    val_image_path = os.path.join(data_root, 'DUTS/imgs')
    val_label_path = os.path.join(data_root, 'DUTS/gt')
    valid_file = os.path.join(data_root, 'DUTS/ImageSets/test_id.txt') 

    #--------------------------------------------------------
    # # #DUT-OMRON
    # test_image_path = os.path.join(data_root,'DUT-OMRON/imgs')
    # test_label_path = os.path.join(data_root,'DUT-OMRON/gt')
    # test_file = os.path.join(data_root,'DUT-OMRON/ImageSets/test_id.txt')
    # pos = test_label_path.find('gt') + 3
    
    # DUTS
    test_image_path = os.path.join(data_root, 'DUTS/imgs')
    test_label_path = os.path.join(data_root, 'DUTS/gt')
    test_file = os.path.join(data_root, 'DUTS/ImageSets/test_id.txt')
    # test_file = os.path.join(data_root, 'DUTS/ImageSets/total_id.txt')
    pos = test_label_path.find('gt') + 3

    # # #ECSSD
    # test_image_path = os.path.join(data_root,'ECSSD/imgs')
    # test_label_path = os.path.join(data_root,'ECSSD/gt')
    # test_file = os.path.join(data_root,'ECSSD/ImageSets/test_id.txt')
    # pos = test_label_path.find('gt') + 3

    # # #HKU-IS
    # test_image_path = os.path.join(data_root,'HKU-IS/imgs')
    # test_label_path = os.path.join(data_root,'HKU-IS/gt')
    # test_file = os.path.join(data_root,'HKU-IS/ImageSets/total_id.txt')  ###
    # pos = test_label_path.find('gt') + 3

    # # #PASCAL-S
    # test_image_path = os.path.join(data_root,'PASCAL-S/imgs')
    # test_label_path = os.path.join(data_root,'PASCAL-S/gt')
    # test_file = os.path.join(data_root,'PASCAL-S/ImageSets/test_id.txt')
    # pos = test_label_path.find('gt') + 3

    # # #SOD
    # test_image_path = os.path.join(data_root,'SOD/imgs')
    # test_label_path = os.path.join(data_root,'SOD/gt')
    # test_file = os.path.join(data_root,'SOD/ImageSets/test_id.txt')
    # pos = test_label_path.find('gt') + 3


    parser = argparse.ArgumentParser()

    # Hyper-parameters
    parser.add_argument('--n_color', type=int, default=3)
    parser.add_argument('--img_size', type=int, default=256)  # 256
    parser.add_argument('--lr', type=float, default=1e-4)#-4 #1e-6
    # parser.add_argument('--lr_d', type=float, default=1e-4)#-4 #1e-6
    # parser.add_argument('--LAMBDA_ADV_MAIN', type=float, default=0.01) # loss_adv
    parser.add_argument('--clip_gradient', type=float, default=1.0)
    parser.add_argument('--cuda', type=bool, default=True)

    # Training settings
    parser.add_argument('--vgg', type=str, default=vgg_path)
    parser.add_argument('--train_path', type=str, default=image_path)
    parser.add_argument('--label_path', type=str, default=label_path)
    parser.add_argument('--train_file', type=str, default=train_file)
    parser.add_argument('--target_image_path', type=str, default=target_image_path)# target domain
    parser.add_argument('--target_label_path', type=str, default=target_label_path)# target domain
    parser.add_argument('--target_file', type=str, default=target_file)# the id of target domain
    parser.add_argument('--early_stop', type=int, default=70000)# iteration number
    parser.add_argument('--iter_save', type=int, default=1400)# save model every  epoch
    parser.add_argument('--iter_val', type=int, default=1400) # equals  epoch
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1)  # 8 # 16
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--val', type=bool, default=True)
    parser.add_argument('--val_path', type=str, default=val_image_path)
    parser.add_argument('--val_label', type=str, default=val_label_path)
    parser.add_argument('--val_file', type=str, default=valid_file)
    parser.add_argument('--num_thread', type=int, default=4)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--save_fold', type=str, default='/data1/liumengmeng/DSS/DSS-results')
    parser.add_argument('--val_fold', type=str, default='/data1/liumengmeng/DSS/DSS-results')
    parser.add_argument('--val_fold_sub', type=str, default=None)
    parser.add_argument('--epoch_val', type=int, default=10)
    parser.add_argument('--epoch_save', type=int, default=10)
    parser.add_argument('--epoch_show', type=int, default=1)
    parser.add_argument('--pre_trained', type=str, default=None)
    parser.add_argument('--add_adv', type=bool, default=False)

    # Testing settings
    parser.add_argument('--test_path', type=str, default=test_image_path)
    parser.add_argument('--test_label', type=str, default=test_label_path)
    parser.add_argument('--test_file', type=str, default=test_file)
    parser.add_argument('--model', type=str, default='/data1/liumengmeng/DSS/tmp/DSS-results-v4/run-6/models/best.pth')
    parser.add_argument('--test_fold', type=str, default='/data1/liumengmeng/DSS/DSS-results')
    parser.add_argument('--test_map_fold', type=str, default='/data1/liumengmeng/DSS/DSS-results')
    parser.add_argument('--test_map_save_pos', type=int, default=pos)
    parser.add_argument('--use_crf', type=bool, default=True)

    # Misc
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--visdom', type=bool, default=False)

    config = parser.parse_args()
    if not os.path.exists(config.save_fold): os.mkdir(config.save_fold)
    main(config)
