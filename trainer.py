import argparse
import os
import sys
import torch
import torch.optim.lr_scheduler as lrs
from torch.autograd import Variable  # 获取变量
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from datasets import CompletionDataset
from model.AGGNet import AGGNet
from loss import Loss

sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append('.../')
parser = argparse.ArgumentParser(description='PyTorch AGG-Net Training')

# net parameters
parser.add_argument('--datasets', default='NYUDepth', type=str, help='train dataset')
parser.add_argument('--model', default='default', type=str, help='model')
parser.add_argument('--kernel_size', default=3, type=str, help='model')
parser.add_argument('--num_layer', default=4, type=str, help='model')
parser.add_argument('--ratio_linear', default=4, type=str, help='model')

# optimizer parameters
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--momentum', default=0.95, type=float, help='sgd momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay (L2 penalty)')
parser.add_argument('--dampening', default=0.0, type=float, help='dampening for momentum')
parser.add_argument('--nesterov', '-n', action='store_true', help='enables Nesterov momentum')
parser.add_argument('--num_epoch', default=200, type=int, help='number of epoch for training')

# batch size
parser.add_argument('--batch_size_train', default=16, type=int, help='batch size for training')
parser.add_argument('--batch_size_eval', default=1, type=int, help='batch size for eval')
parser.add_argument('--resume', '-r', default=True, action='store_true', help='resume from checkpoint')

args = parser.parse_args()

# root path
root = r'\home\htx0601\AGGNet'
dataset_train = CompletionDataset(os.path.join(root, "datasets"), args.datasets, 'train', data_len=0)
dataset_test = CompletionDataset(os.path.join(root, "datasets"), args.datasets, 'test', data_len=0)
dir_name = "result_" + args.datasets + '_AGG-Net'
save_dir = os.path.join(root, dir_name)
save_img_dir = os.path.join(root, dir_name, "img")
train_loader = DataLoader(dataset_train, batch_size=args.batch_size_train, shuffle=True, num_workers=4, drop_last=True)
eval_loader = DataLoader(dataset_test, batch_size=args.batch_size_eval, shuffle=False, num_workers=0)
best_delta = 1  # best test rmse
utils.log_file_folder_make_lr(save_dir)

if args.datasets == 'Kitti':
    size_h = [256, 128, 64, 32, 16, 8]
    size_w = [1216, 608, 304, 152, 76, 38]
else:
    size_h = [192, 96, 48, 24, 12, 6]
    size_w = [288, 144, 72, 36, 18, 9]

if args.model == 'default':
    model = AGGNet(size_h, size_w, num_layer=args.num_layer, kernel_size=args.kernel_size).cuda()
    print(model)

if args.resume:
    # Load best model checkpoint.
    print("============> Load Pretrained Model <============")
    best_model_path = os.path.join(save_dir, 'latest_model.pth')
    print(best_model_path)
    assert os.path.isdir(save_dir), 'Error: no checkpoint directory found!'
    try:
        best_model_dict = torch.load(best_model_path)
        best_model_dict = utils.remove_moudle(best_model_dict)
        model.load_state_dict(utils.update_model(model, best_model_dict))
    except:
        print("Failed to load, start a new training process!")
print("============> Start Training <============")

optimizer = torch.optim.SGD(model.parameters(),
                            lr=args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay,
                            nesterov=args.nesterov,
                            dampening=args.dampening)
scheduler = lrs.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, min_lr=1e-5)  # set up scheduler
loss_fn = Loss()

if args.datasets == 'Kitti':
    unit = 10000
else:
    unit = 1


def train(epoch):
    model.train()
    total_step_train = 0
    train_loss = 0.0
    error_sum_train = {'RMSE': 0, 'RMSE_M': 0, 'ABS_REL': 0,
                       'DELTA1.02': 0, 'DELTA1.05': 0, 'DELTA1.10': 0,
                       'DELTA1.25': 0, 'DELTA1.25^2': 0, 'DELTA1.25^3': 0,
                       }
    tbar = tqdm(train_loader)
    for batch_idx, data in enumerate(tbar):
        rgb, depth, mask, gt = data['rgb'], data['depth'], data['mask'], data['gt']
        rgb, depth, mask, gt = Variable(rgb).cuda(), Variable(depth).cuda(), Variable(mask).cuda(), Variable(gt).cuda()
        optimizer.zero_grad()
        output = model(rgb, depth, mask, False)

        loss = loss_fn(output, gt, mask)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        error_str = 'Epoch: %d, loss=%.5f' % (epoch, train_loss / (batch_idx + 1))
        tbar.set_description(error_str)

        error_result = utils.evaluate_error(gt.cpu() * unit, output.cpu() * unit, mask.cpu() * unit, False)
        total_step_train += args.batch_size_train
        error_avg = utils.avg_error(error_sum_train,
                                    error_result,
                                    total_step_train,
                                    args.batch_size_train)

        if batch_idx % 50 == 0:
            utils.print_error('training_result: step(average)',
                              epoch,
                              batch_idx,
                              loss,
                              error_result,
                              error_avg,
                              print_out=True
                              )
            record_loss = utils.save_error(
                epoch,
                batch_idx,
                loss,
                error_result,
                error_avg,
                print_out=False
            )
            utils.log_loss_lr(save_dir, record_loss, 'train')

    error_avg = utils.avg_error(error_sum_train,
                                error_result,
                                total_step_train,
                                args.batch_size_train)

    for param_group in optimizer.param_groups:
        old_lr = float(param_group['lr'])
    utils.log_result_lr(save_dir, error_avg, epoch, old_lr, False, 'train')

    # saving the latest model
    tmp_name = "epoch_%02d.pth" % epoch
    save_name = os.path.join(save_dir, tmp_name)
    save_name_ = os.path.join(save_dir, "latest_model.pth")
    torch.save(model.state_dict(), save_name_)

    # saving the model on schedule
    if epoch % 10 == 0:
        torch.save(model.state_dict(), save_name)

    # update lr
    scheduler.step(error_avg['RMSE'], epoch)


def eval(epoch, save_img='False'):
    global best_delta
    is_best_model = False
    model.eval()
    total_step_val = 0
    eval_loss = 0.0
    error_sum_val = {'RMSE': 0, 'RMSE_M': 0, 'ABS_REL': 0,
                     'DELTA1.02': 0, 'DELTA1.05': 0, 'DELTA1.10': 0,
                     'DELTA1.25': 0, 'DELTA1.25^2': 0, 'DELTA1.25^3': 0,
                     }
    tbar = tqdm(eval_loader)
    for batch_idx, data in enumerate(tbar):
        with torch.no_grad():
            rgb, depth, mask, gt = data['rgb'], data['depth'], data['mask'], data['gt']
            rgb, depth, mask, gt = Variable(rgb).cuda(), Variable(depth).cuda(), Variable(mask).cuda(), Variable(
                gt).cuda()
            optimizer.zero_grad()
            output = model(rgb, depth, mask, False)

            loss = loss_fn(output, gt, mask)
            loss.backward()
            optimizer.step()

        loss = loss.data.cpu()
        eval_loss += loss.item()
        error_str = 'Epoch: %d, loss=%.4f' % (epoch, eval_loss / (batch_idx + 1))
        tbar.set_description(error_str)

        error_result = utils.evaluate_error(gt * unit, output * unit, mask, False)
        utils.save_eval_img(save_img_dir, batch_idx, depth[0].cpu() / 10, gt[0].cpu() / 10, output[0].cpu() / 10)
        total_step_val += args.batch_size_eval
        error_avg = utils.avg_error(error_sum_val, error_result, total_step_val, args.batch_size_eval)

        # eval_loss_record
        if batch_idx % args.batch_size_eval * 10 == 0:
            record_loss = utils.save_error(
                epoch,
                batch_idx,
                loss,
                error_result,
                error_avg,
                print_out=False
            )

            utils.log_loss_lr(save_dir, record_loss, 'eval')

    utils.print_error('eval_result: step(average)',
                      epoch, batch_idx, loss,
                      error_result, error_avg, print_out=True)
    # log best_model
    if utils.update_best_model(error_avg, best_delta):
        is_best_model = True
        best_delta = error_avg['RMSE'] if error_avg['RMSE'] < 10 else 10
    for param_group in optimizer.param_groups:
        old_lr = float(param_group['lr'])
    utils.log_result_lr(save_dir, error_avg, epoch, old_lr, is_best_model, 'eval')

    # saving best_model
    if is_best_model:
        print('==> saving best model at epoch %d' % epoch)
        best_model_pytorch = os.path.join(save_dir, 'best_model.pth')
        torch.save(model.state_dict(), best_model_pytorch)


if __name__ == '__main__':
    for epoch in range(1, args.num_epoch):
        train(epoch)
        eval(epoch)
        # break
