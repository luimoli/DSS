import math
import torch
from collections import OrderedDict
from torch.nn import utils, functional as F
from torch.optim import Adam
from torch.backends import cudnn
from torchvision import transforms
from torchvision.transforms import ToPILImage
from dssnet import build_model, weights_init
from loss import Loss
from tools.visual import Viz_visdom
from PIL import Image
import numpy as np

import os
import sys
from pathlib import Path
import os.path as osp
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch import nn
from torchvision.utils import make_grid
from tqdm import tqdm
from advent.model.discriminator import get_fc_discriminator
from advent.utils.func import adjust_learning_rate, adjust_learning_rate_discriminator
from advent.utils.func import loss_calc, bce_loss
from advent.utils.loss import entropy_loss
from advent.utils.func import prob_2_entropy
from advent.utils.viz_segmask import colorize_mask



class Solver(object):
    def __init__(self, train_loader, target_loader,val_loader, test_dataset, config):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_dataset = test_dataset
        self.targetloader = target_loader
        self.config = config
        self.beta = math.sqrt(0.3)  # for max F_beta metric
        # inference: choose the side map (see paper)
        self.select = [1, 2, 3, 6]
        self.device = torch.device('cpu')
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        self.TENSORBOARD_LOGDIR = f'{config.save_fold}/tensorboards'
        self.TENSORBOARD_VIZRATE = 100
        if self.config.cuda:
            cudnn.benchmark = True
            self.device = torch.device('cuda:0')
        if config.visdom:
            self.visual = Viz_visdom("DSS", 1)
        self.build_model()
        if self.config.pre_trained: self.net.load_state_dict(torch.load(self.config.pre_trained))
        if config.mode == 'train':
            self.log_output = open("%s/logs/log.txt" % config.save_fold, 'w')
            self.val_output = open("%s/logs/val.txt" % config.save_fold, 'w')
        else:
            self.net.load_state_dict(torch.load(self.config.model))
            self.net.eval()
            self.test_output = open("%s/test.txt" % config.test_fold, 'w')
            self.test_outmap = config.test_map_fold
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    # print the network information and parameter numbers
    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            if p.requires_grad: num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    # build the network
    def build_model(self):
        self.net = build_model().to(self.device)
        if self.config.mode == 'train': self.loss = Loss().to(self.device)
        self.net.train()
        self.net.apply(weights_init)
        if self.config.load == '': self.net.base.load_state_dict(torch.load(self.config.vgg))
        if self.config.load != '': self.net.load_state_dict(torch.load(self.config.load))
        self.optimizer = Adam(self.net.parameters(), self.config.lr)
        self.print_network(self.net, 'DSS')

    # update the learning rate
    def update_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    # evaluate MAE (for test or validation phase)
    def eval_mae(self, y_pred, y):
        return torch.abs(y_pred - y).mean()

    # TODO: write a more efficient version
    # get precisions and recalls: threshold---divided [0, 1] to num values
    def eval_pr(self, y_pred, y, num):
        prec, recall = torch.zeros(num), torch.zeros(num)
        thlist = torch.linspace(0, 1 - 1e-10, num)
        for i in range(num):
            y_temp = (y_pred >= thlist[i]).float()
            tp = (y_temp * y).sum()
            # prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / y.sum()
            prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)
        return prec, recall

    # validation: using resize image, and only evaluate the MAE metric
    def validation(self):
        avg_mae, avg_loss = 0.0, 0.0
        self.net.eval()
        with torch.no_grad():
            for i, data_batch in enumerate(self.val_loader):
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                prob_pred = self.net(images)
                # for side_num in range(len(prob_pred)):
                #         tmp = torch.sigmoid(prob_pred[side_num])[0]
                #         tmp = tmp.cpu().data
                #         img = ToPILImage()(tmp)
                #         img.save(self.config.val_fold_sub + '/' + self.val_loader.dataset.label_path[i][36:-4] +'_side_' + str(side_num) + '.png')
                avg_loss += self.loss(prob_pred[0], labels).item()
                # prob_pred = torch.mean(torch.cat([torch.sigmoid(prob_pred[i]) for i in self.select], dim=1), dim=1, keepdim=True)
                avg_mae += self.eval_mae(torch.sigmoid(prob_pred[0]), labels).item()
                # print(f'{i} --  {iloss}',file=self.val_output)
        self.net.train()
        return avg_mae / len(self.val_loader), avg_loss / len(self.val_loader)

    # test phase: using origin image size, evaluate MAE and max F_beta metrics
    def test(self, num, use_crf=False):
        avg_mae, img_num = 0.0, len(self.test_dataset)
        avg_prec, avg_recall = torch.zeros(num), torch.zeros(num)
        with torch.no_grad():
            for i, (img, labels) in enumerate(self.test_dataset):
                images = self.transform(img).unsqueeze(0)
                labels = labels.unsqueeze(0)
                shape = labels.size()[2:]
                images = images.to(self.device)
                prob_pred = self.net(images)
                # prob_pred = torch.mean(torch.cat([torch.sigmoid(prob_pred[i]) for i in self.select], dim=1), dim=1, keepdim=True)
                prob_pred = F.interpolate(torch.sigmoid(prob_pred[0]), size=shape, mode='bilinear', align_corners=True).cpu().data
                mae = self.eval_mae(prob_pred, labels)
                prec, recall = self.eval_pr(prob_pred, labels, num)
                tmp = prob_pred[0]
                img = ToPILImage()(tmp)
                img.save(self.test_outmap + '/' + self.test_dataset.label_path[i][36:])
                print("[%d] mae: %.4f" % (i, mae))
                print("[%d] mae: %.4f" % (i, mae), file=self.test_output)
                avg_mae += mae
                avg_prec, avg_recall = avg_prec + prec, avg_recall + recall
        avg_mae, avg_prec, avg_recall = avg_mae / img_num, avg_prec / img_num, avg_recall / img_num
        score = (1 + self.beta ** 2) * avg_prec * avg_recall / (self.beta ** 2 * avg_prec + avg_recall)
        score[score != score] = 0  # delete the nan
        print('average mae: %.4f, max fmeasure: %.4f' % (avg_mae, score.max()))
        print('average mae: %.4f, max fmeasure: %.4f' % (avg_mae, score.max()), file=self.test_output)


    def train_advent(self):
        ''' UDA training with advent
        '''
        num_classes = 1
        viz_tensorboard = os.path.exists(self.TENSORBOARD_LOGDIR)
        if viz_tensorboard:
            writer = SummaryWriter(log_dir=self.TENSORBOARD_LOGDIR)

        # DISCRIMINATOR NETWORK
        # seg maps, i.e. output, level
        # d_main = get_fc_discriminator(num_classes=num_classes)
        # d_main.train()
        # d_main.to(self.device)

        # # OPTIMIZERS
        # # discriminators' optimizers
        # optimizer_d_main = optim.Adam(d_main.parameters(), lr=self.config.lr_d,
        #                             betas=(0.9, 0.99))                               
        # labels for adversarial training-------------------------------------------------------
        # source_label = 0
        # target_label = 1
        trainloader_iter = enumerate(self.train_loader)
        targetloader_iter = enumerate(self.targetloader)
        best_mae = 1.0 if self.config.val else None 

        for i_iter in tqdm(range(self.config.early_stop)):
            
            # if i_iter >= 3000:
            #     self.update_lr(1e-5)

            # reset optimizers
            self.optimizer.zero_grad()
            # optimizer_d_main.zero_grad()

            # # adapt LR if needed
            # adjust_learning_rate(self.optimizer, i_iter, cfg)
            # adjust_learning_rate_discriminator(optimizer_d_aux, i_iter, cfg)
            # adjust_learning_rate_discriminator(optimizer_d_main, i_iter, cfg)

            # UDA Training--------------------------------------------------------------------------
            # # only train segnet. Don't accumulate grads in disciminators
            # for param in d_main.parameters():
            #     param.requires_grad = False
            
            # train on source with seg loss
            _, batch = trainloader_iter.__next__()
            imgs_src, labels_src = batch
            imgs_src, labels_src = imgs_src.to(self.device), labels_src.to(self.device)
            pred_src = self.net(imgs_src)
            loss_seg_src = self.loss(pred_src[0], labels_src) #side output 1
            loss = loss_seg_src
            loss.backward()
            utils.clip_grad_norm_(self.net.parameters(), self.config.clip_gradient)

            # train on target with seg loss 
            _, batch1 = targetloader_iter.__next__()
            imgs_trg, labels_trg = batch1
            imgs_trg, labels_trg = imgs_trg.to(self.device), labels_trg.to(self.device)
            pred_trg = self.net(imgs_trg)
            loss_seg_trg = self.loss(pred_trg[5], labels_trg) # side output 6
            loss = loss_seg_trg
            loss.backward()
            utils.clip_grad_norm_(self.net.parameters(), self.config.clip_gradient)          


            # if self.config.add_adv:
            #     # adversarial training ot fool the discriminator-------------------------------------------
            #     _, batch = targetloader_iter.__next__()
            #     images = batch
            #     images = images.to(self.device)
            #     pred_trg_main = self.net(images)
    
            #     d_out_main = d_main(torch.sigmoid(pred_trg_main))
            #     loss_adv_trg_main = bce_loss(d_out_main, source_label)
            #     loss_adv_trg = self.config.LAMBDA_ADV_MAIN * loss_adv_trg_main
            #     loss = loss_adv_trg
            #     loss.backward()

            #     # # d_out_main = d_main(prob_2_entropy(pred_trg_main[0]))
            #     # d_out_main = d_main(torch.sigmoid(pred_trg_main[0]))
            #     # loss_adv_trg_main = bce_loss(d_out_main, source_label)
            #     # loss_adv_trg = self.config.LAMBDA_ADV_MAIN * loss_adv_trg_main
            #     # for i in range(len(pred_trg_main) - 1):
            #     #     # d_out_main = d_main(prob_2_entropy(pred_trg_main[i+1]))
            #     #     d_out_main = d_main(torch.sigmoid(pred_trg_main[i+1]))
            #     #     loss_adv_trg_main = bce_loss(d_out_main, source_label)
            #     #     loss_adv_trg += self.config.LAMBDA_ADV_MAIN * loss_adv_trg_main
            #     # loss = loss_adv_trg
            #     # loss.backward()

            #     # Train discriminator networks--------------------------------------------------------------
            #     # enable training mode on discriminator networks
            #     for param in d_main.parameters():
            #         param.requires_grad = True
                
            #     # train with source
            #     pred_src_main = pred_src_main.detach()
            #     d_out_main = d_main(torch.sigmoid(pred_src_main))
            #     loss_d_main = bce_loss(d_out_main, source_label)
            #     loss_d_src = loss_d_main / 2
            #     loss_d = loss_d_src
            #     loss_d.backward()

            #     # train with target
            #     pred_trg_main = pred_trg_main.detach()
            #     d_out_main = d_main(torch.sigmoid(pred_trg_main))
            #     loss_d_main = bce_loss(d_out_main, target_label)
            #     loss_d_trg = loss_d_main / 2
            #     loss_d = loss_d_trg
            #     loss_d.backward()

            #     # # train with source
            #     # pred_src_main[0] = pred_src_main[0].detach()
            #     # # d_out_main = d_main(prob_2_entropy(pred_src_main[0]))
            #     # d_out_main = d_main(torch.sigmoid(pred_src_main[0]))
            #     # loss_d_main = bce_loss(d_out_main, source_label)
            #     # loss_d_src = loss_d_main / 2
            #     # for i in range(len(pred_src_main) - 1):
            #     #     pred_src_main[i+1] = pred_src_main[i+1].detach()
            #     #     # d_out_main = d_main(prob_2_entropy(pred_src_main[i+1]))
            #     #     d_out_main = d_main(torch.sigmoid(pred_src_main[i+1]))
            #     #     loss_d_main = bce_loss(d_out_main, source_label)
            #     #     loss_d_src += loss_d_main / 2
            #     # loss_d = loss_d_src
            #     # loss_d.backward()

            #     # # train with target
            #     # pred_trg_main[0] = pred_trg_main[0].detach()
            #     # # d_out_main = d_main(prob_2_entropy(pred_trg_main[0]))
            #     # d_out_main = d_main(torch.sigmoid(pred_trg_main[0]))
            #     # loss_d_main = bce_loss(d_out_main, target_label)
            #     # loss_d_trg = loss_d_main / 2
            #     # for i in range(len(pred_trg_main) - 1):
            #     #     pred_trg_main[i+1] = pred_trg_main[i+1].detach()
            #     #     # d_out_main = d_main(prob_2_entropy(pred_trg_main[i+1]))
            #     #     d_out_main = d_main(torch.sigmoid(pred_trg_main[i+1]))
            #     #     loss_d_main = bce_loss(d_out_main, target_label)
            #     #     loss_d_trg += loss_d_main / 2
            #     # loss_d = loss_d_trg
            #     # loss_d.backward()


            # optimizer.step()------------------------------------------------------------------------------
            self.optimizer.step()
            # optimizer_d_main.step()
                
            current_losses = {
                            'loss_seg_src': loss_seg_src,
                            'loss_srg_trg': loss_seg_trg}
                            # 'loss_adv_trg': loss_adv_trg,
                            # 'loss_d_src': loss_d_src,
                            # 'loss_d_trg': loss_d_trg}
            print_losses(current_losses, i_iter, self.log_output)

            if self.config.val and (i_iter + 1) % self.config.iter_val == 0:
                # val = i_iter + 1
                # os.mkdir("%s/val-%d" % (self.config.val_fold, val))
                # self.config.val_fold_sub = "%s/val-%d" % (self.config.val_fold, val)
                mae, loss_val = self.validation()
                log_vals_tensorboard(writer, best_mae,mae,loss_val, i_iter+1)
                tqdm.write('%d:--- Best MAE: %.4f, Curr MAE: %.4f ---' % ((i_iter + 1),best_mae, mae))
                print('  %d:--- Best MAE: %.4f, Curr MAE: %.4f ---' % ((i_iter + 1),best_mae, mae), file=self.log_output)
                print('  %d:--- Best MAE: %.4f, Curr MAE: %.4f ---' % ((i_iter + 1),best_mae, mae), file=self.val_output)
                if best_mae > mae:
                    best_mae = mae
                    torch.save(self.net.state_dict(), '%s/models/best.pth' % self.config.save_fold)
            
            if (i_iter + 1) % self.config.iter_save == 0 and i_iter != 0:
                # tqdm.write('taking snapshot ...')
                # torch.save(self.net.state_dict(), '%s/models/iter_%d.pth' % (self.config.save_fold, i_iter + 1))
                # torch.save(d_main.state_dict(), '%s/models/iter_Discriminator_%d.pth' % (self.config.save_fold, i_iter + 1))
                if i_iter >= self.config.early_stop - 1:
                    break
            
            sys.stdout.flush()
            
            if viz_tensorboard:
                log_losses_tensorboard(writer, current_losses, i_iter)
                # if i_iter % self.TENSORBOARD_VIZRATE == self.TENSORBOARD_VIZRATE - 1:
                #     draw_in_tensorboard(writer, images, i_iter, pred_trg_main, num_classes, 'T')
                #     draw_in_tensorboard(writer, images_source, i_iter, pred_src_main, num_classes, 'S')

        # torch.save(self.net.state_dict(), '%s/models/final.pth' % self.config.save_fold)



    def train_old(self):
        print(len(self.train_loader.dataset))
        iter_num = len(self.train_loader.dataset) // self.config.batch_size
        best_mae = 1.0 if self.config.val else None 
        for epoch in range(self.config.epoch):
            loss_epoch = 0
            for i, data_batch in enumerate(self.train_loader):
                if (i + 1) > iter_num: break
                self.net.zero_grad()
                x, y = data_batch
                x, y = x.to(self.device), y.to(self.device)
                y_pred = self.net(x)
                loss = self.loss(y_pred, y)
                loss.backward()
                utils.clip_grad_norm_(self.net.parameters(), self.config.clip_gradient)
                # utils.clip_grad_norm(self.loss.parameters(), self.config.clip_gradient)
                self.optimizer.step()
                loss_epoch += loss.item()
                print('epoch: [%d/%d], iter: [%d/%d], loss: [%.4f]' % (
                    epoch, self.config.epoch, i, iter_num, loss.item()))
                if self.config.visdom:
                    error = OrderedDict([('loss:', loss.item())])
                    self.visual.plot_current_errors(epoch, i / iter_num, error)

            if (epoch + 1) % self.config.epoch_show == 0:
                print('epoch: [%d/%d], epoch_loss: [%.4f]' % (epoch, self.config.epoch, loss_epoch / iter_num),
                      file=self.log_output)
                if self.config.visdom:
                    avg_err = OrderedDict([('avg_loss', loss_epoch / iter_num)])
                    self.visual.plot_current_errors(epoch, i / iter_num, avg_err, 1)
                    y_show = torch.mean(torch.cat([y_pred[i] for i in self.select], dim=1), dim=1, keepdim=True)
                    img = OrderedDict([('origin', x.cpu()[0] * self.std + self.mean), ('label', y.cpu()[0][0]),
                                       ('pred_label', y_show.cpu().data[0][0])])
                    self.visual.plot_current_img(img)

            # if self.config.val and (epoch + 1) % self.config.epoch_val == 0:
            #     mae = self.validation()
            #     print('--- Best MAE: %.2f, Curr MAE: %.2f ---' % (best_mae, mae))
            #     print('--- Best MAE: %.2f, Curr MAE: %.2f ---' % (best_mae, mae), file=self.log_output)
            #     if best_mae > mae:
            #         best_mae = mae
            #         torch.save(self.net.state_dict(), '%s/models/best.pth' % self.config.save_fold)
            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.net.state_dict(), '%s/models/epoch_%d.pth' % (self.config.save_fold, epoch + 1))
        torch.save(self.net.state_dict(), '%s/models/final.pth' % self.config.save_fold)

#-----------------------------------------------------------------------------------------------------------
def draw_in_tensorboard(writer, images, i_iter, pred_main, num_classes, type_):
    grid_image = make_grid(images[:3].clone().cpu().data, 3, normalize=True)
    writer.add_image(f'Image - {type_}', grid_image, i_iter)

    grid_image = make_grid(torch.from_numpy(np.array(colorize_mask(np.asarray(
        np.argmax(F.softmax(pred_main).cpu().data[0].numpy().transpose(1, 2, 0),
                  axis=2), dtype=np.uint8)).convert('RGB')).transpose(2, 0, 1)), 3,
                           normalize=False, range=(0, 255))
    writer.add_image(f'Prediction - {type_}', grid_image, i_iter)

    output_sm = F.softmax(pred_main).cpu().data[0].numpy().transpose(1, 2, 0)
    output_ent = np.sum(-np.multiply(output_sm, np.log2(output_sm)), axis=2,
                        keepdims=False)
    grid_image = make_grid(torch.from_numpy(output_ent), 3, normalize=True,
                           range=(0, np.log2(num_classes)))
    writer.add_image(f'Entropy - {type_}', grid_image, i_iter)

def to_numpy(tensor):
    if isinstance(tensor, (int, float)):
        return tensor
    else:
        return tensor.data.cpu().numpy()

def print_losses(current_losses, i_iter, file_):
    list_strings = []
    for loss_name, loss_value in current_losses.items():
        list_strings.append(f'{loss_name} = {to_numpy(loss_value):.5f} ')
    full_string = ' '.join(list_strings)
    # tqdm.write(f'iter = {i_iter} {full_string}')
    print(f'iter = {i_iter} {full_string}', file=file_)

def log_losses_tensorboard(writer, current_losses, i_iter):
    for loss_name, loss_value in current_losses.items():
        writer.add_scalar(f'loss/{loss_name}', to_numpy(loss_value), i_iter)

def log_vals_tensorboard(writer, best_mae, mae, loss_val, val_iter):
    writer.add_scalar(f'val/best_mae', best_mae, val_iter)
    writer.add_scalar(f'val/curr_mae', mae, val_iter)
    writer.add_scalar(f'val/loss_seg', loss_val, val_iter)
