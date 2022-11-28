# encoding: utf-8
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.optim as optim
import torch.backends.cudnn as cudnn
from dataset.KMNIST import get_train_dataloader, get_validation_dataloader, get_test_dataloader
# self-defined
from utils.init import *
from utils.logger import get_logger
from utils.evaluation import avg_accuracy, class_accuracy, visualize_val_accuracy, visualize_train_loss, \
    visualize_confusion_matrix
from config import *


class Classification:
    def __init__(self, model='ResNet-18', train_batch=64, test_batch=1000, epoch=30, ckpt_path='./models/',
                 class_num=10, log_img_path=''):
        self.dataloader_train = get_train_dataloader(batch_size=train_batch, shuffle=True, num_workers=4)
        self.dataloader_val = get_validation_dataloader(batch_size=train_batch, shuffle=False, num_workers=4)
        self.dataloader_test = get_test_dataloader(batch_size=test_batch, shuffle=False, num_workers=4)
        self.model = nn.DataParallel(get_model(model)).cuda()
        torch.backends.cudnn.benchmark = True
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        self.lr_scheduler_model = lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=1)
        self.max_epoch = epoch
        self.loss_log = []
        self.accuracy_log = []
        self.ckpt_path = ckpt_path + model + '.pkl'
        self.class_num = class_num
        self.log_img_path = log_img_path

    def train_epoch(self):
        self.model.train()  # set model to training mode
        train_loss = []
        with torch.autograd.enable_grad():
            for batch_idx, (img, lbl) in enumerate(self.dataloader_train):
                image = torch.autograd.Variable(img).cuda()
                label = torch.autograd.Variable(lbl).cuda()
                self.optimizer.zero_grad()
                output = self.model(image)  # forward
                if type(output) is tuple:
                    output = output[0]
                loss_tensor = self.criterion.forward(output, label)

                loss_tensor.backward()
                self.optimizer.step()  ##update parameters
                train_loss.append(loss_tensor.item())

        return train_loss

    def test_epoch(self, dataloader=None):
        self.model.eval()
        if dataloader is None:
            dataloader = self.dataloader_test

        gt = torch.FloatTensor().cuda()
        pred = torch.FloatTensor().cuda()

        loss_test = []
        with torch.autograd.no_grad():
            for batch_idx, (img, lbl) in enumerate(dataloader):
                # forward
                image = torch.autograd.Variable(img).cuda()
                label = torch.autograd.Variable(lbl).cuda()
                output = self.model(image)
                if type(output) is tuple:
                    output = output[0]
                loss_tensor = self.criterion.forward(output, label)
                loss_test.append(loss_tensor.item())
                _, pred_label = torch.max(output.data, 1)

                gt = torch.cat((gt, label.data), 0)
                pred = torch.cat((pred, pred_label.data), 0)  # todo

        return np.mean(loss_test), gt.cpu().numpy(), pred.cpu().numpy()

    def val_epoch(self):
        loss, gt, pred = self.test_epoch(self.dataloader_val)
        acc = avg_accuracy(gt, pred)
        return loss, acc

    def train_model(self):
        logger.info('********************begin training!********************')
        accuracy_max = 0.0
        for epoch in range(self.max_epoch):
            # train
            train_loss = self.train_epoch()
            train_loss = np.mean(train_loss)
            self.loss_log.append(train_loss)

            logger.info("Eopch: %5d train loss = %.6f" % (epoch + 1, train_loss))
            self.lr_scheduler_model.step()

            # validation
            val_loss, val_accuracy = self.val_epoch()

            logger.info("Eopch: %5d valuation loss = %.6f, ACC = %.6f" % (epoch + 1, val_loss, val_accuracy))
            self.accuracy_log.append(val_accuracy)

            # save checkpoint
            if accuracy_max < val_accuracy:
                accuracy_max = val_accuracy
                torch.save(self.model.state_dict(), self.ckpt_path)  # Saving torch.nn.DataParallel Models
                logger.info(' Epoch: {} model has been already save!'.format(epoch + 1))

            logger.info(
                'Training epoch: {} completed.'.format(epoch + 1))

        visualize_train_loss(self.loss_log, logger, self.log_img_path)
        visualize_val_accuracy(self.accuracy_log, logger, self.log_img_path)
        logger.info('Train Loss:')
        logger.info(','.join([str(x) for x in self.loss_log]))
        logger.info('Validation Accuracy:')
        logger.info(','.join([str(x) for x in self.accuracy_log]))

    def test_model(self):
        if os.path.isfile(self.ckpt_path):
            checkpoint = torch.load(self.ckpt_path)
            self.model.load_state_dict(checkpoint)
            logger.info("=> loaded model checkpoint: " + self.ckpt_path)

        logger.info('******* begin testing!*********')

        loss, gt, pred = self.test_epoch()
        logger.info("Test Averaged Loss = %.6f" % (loss))
        test_acc = avg_accuracy(gt, pred)
        logger.info("Test Averaged Accuracy = %.6f" % (test_acc))
        cm = visualize_confusion_matrix(gt, pred, logger, self.log_img_path)

        for i in range(self.class_num):
            logger.info("Class: %5d      Accuracy = %.6f" % (i, class_accuracy(cm, i)))


if __name__ == '__main__':
    # command parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ResMLP-12')
    parser.add_argument('--gpu', type=str, default=config['CUDA_VISIBLE_DEVICES'])
    parser.add_argument('--train_batch', type=int, default=64)
    parser.add_argument('--test_batch', type=int, default=1000)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--train', type=int, default=0)
    parser.add_argument('--test', type=int, default=1)
    parser.add_argument('--class_num', type=int, default=None)
    parser.add_argument('--ckpt_path', type=str, default='/userhome/cs2/mingzeng/codes/kmnist/models/')

    args = parser.parse_args()
    # set log
    logger = get_logger(config['LOG_PATH'] + args.model)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.class_num is None:
        class_num = config['N_CLASSES']
    else:
        class_num = args.class_num
    classification = Classification(model=args.model, train_batch=args.train_batch, test_batch=args.test_batch,
                                    epoch=args.epoch, ckpt_path=args.ckpt_path, class_num=class_num,
                                    log_img_path=config['LOG_PATH'] + args.model + '/')
    if args.train == 1:
        classification.train_model()
    if args.test == 1:
        classification.test_model()
