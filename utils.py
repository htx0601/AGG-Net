import math
import os
import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np


def update_model(my_model, pretrained_dict):
    my_model_dict = my_model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in my_model_dict}
    # 2. overwrite entries in the existing state dict
    my_model_dict.update(pretrained_dict)

    return my_model_dict


def update_best_model(error_avg, best_DELTA):
    if error_avg['RMSE'] <= best_DELTA:
        return True
    else:
        return False


def remove_moudle(remove_dict):
    for k, v in remove_dict.items():
        if 'module' in k:
            print("==> model dict with addtional module, remove it...")
            removed_dict = {k[7:]: v for k, v in remove_dict.items()}
        else:
            removed_dict = remove_dict
        break
    return removed_dict


def max_of_two(y_over_z, z_over_y):
    return torch.max(y_over_z, z_over_y)


def evaluate_error(gt, output, mask, mask_mode=False):
    error = {'RMSE': 0, 'RMSE_M': 0, 'ABS_REL': 0,
             'DELTA1.02': 0, 'DELTA1.05': 0, 'DELTA1.10': 0,
             'DELTA1.25': 0, 'DELTA1.25^2': 0, 'DELTA1.25^3': 0,
             }
    mask_gt = (gt >= 1e-3).detach()
    gt = gt[mask_gt]
    output = output[mask_gt]
    mask = (mask[mask_gt] == 1).detach()

    gt_m = gt[mask]
    out_m = output[mask]
    diff = torch.abs(output - gt)
    diff_m = torch.abs(out_m - gt_m)

    n_valid_element = gt.size(0)
    n_valid_mask = torch.sum(mask)

    if mask_mode:
        diff = diff_m
        n_valid_element = n_valid_mask
        gt = gt_m
        output = out_m

    gt = torch.clamp(gt, 1e-5, 100000)
    output = torch.clamp(output, 1e-5, 100000)
    diff = torch.clamp(diff, 1e-5, 10000)

    error['RMSE'] = float(math.sqrt(torch.sum(torch.pow(diff, 2)) / n_valid_element))
    error['RMSE_M'] = float(math.sqrt(torch.sum(torch.pow(diff_m, 2)) / n_valid_mask))
    rel_mat = torch.div(diff, gt)
    error['ABS_REL'] = float(torch.sum(rel_mat) / n_valid_element)
    y_over_z = torch.abs(torch.div(gt, output))
    z_over_y = torch.abs(torch.div(output, gt))
    max_ratio = max_of_two(y_over_z, z_over_y)
    error['DELTA1.02'] = torch.sum(max_ratio < 1.02) / float(n_valid_element)
    error['DELTA1.05'] = torch.sum(max_ratio < 1.05) / float(n_valid_element)
    error['DELTA1.10'] = torch.sum(max_ratio < 1.10) / float(n_valid_element)
    error['DELTA1.25'] = torch.sum(max_ratio < 1.25) / float(n_valid_element)
    error['DELTA1.25^2'] = torch.sum(max_ratio < 1.25 ** 2) / float(n_valid_element)
    error['DELTA1.25^3'] = torch.sum(max_ratio < 1.25 ** 3) / float(n_valid_element)
    return error


def avg_error(error_sum, error_step, total_step, batch_size):
    error_avg = {'RMSE': 0, 'RMSE_M': 0, 'ABS_REL': 0,
                 'DELTA1.02': 0, 'DELTA1.05': 0, 'DELTA1.10': 0,
                 'DELTA1.25': 0, 'DELTA1.25^2': 0, 'DELTA1.25^3': 0,
                 }
    for item, value in error_step.items():
        error_sum[item] += error_step[item] * batch_size
        error_avg[item] = error_sum[item] / float(total_step)
    return error_avg


def print_error(split, epoch, step, loss, error, error_avg, print_out=False):
    format_str = ('%s ===>\n\
                      Epoch: %d, step: %d, loss=%.5f\n\
                      RMSE=%.4f(%.4f)\tRMSE_M=%.4f(%.4f)\tABS_REL=%.4f(%.4f)\n\
                      DELTA1.02=%.4f(%.4f)\tDELTA1.05=%.4f(%.4f)\tDELTA1.10=%.4f(%.4f)\n\
                      DELTA1.25=%.4f(%.4f)\tDELTA1.25^2=%.4f(%.4f)\tDELTA1.25^3=%.4f(%.4f)\n')
    error_str = format_str % (split, epoch, step, loss,
                              error['RMSE'], error_avg['RMSE'], error['RMSE_M'], error_avg['RMSE_M'],
                              error['ABS_REL'], error_avg['ABS_REL'],
                              error['DELTA1.02'], error_avg['DELTA1.02'],
                              error['DELTA1.05'], error_avg['DELTA1.05'],
                              error['DELTA1.10'], error_avg['DELTA1.10'],
                              error['DELTA1.25'], error_avg['DELTA1.25'],
                              error['DELTA1.25^2'], error_avg['DELTA1.25^2'],
                              error['DELTA1.25^3'], error_avg['DELTA1.25^3'])
    if print_out:
        print(error_str)
    return error_str


def save_error(epoch, step, loss, error, error_avg, print_out=False):
    format_str = ('%d\t %d\t %.5f \
                  %.4f\t %.4f\t %.4f\t %.4f \
                  %.4f\t %.4f \
                  %.4f\t %.4f \
                  %.4f\t %.4f \
                  %.4f\t %.4f \
                  %.4f\t %.4f \n')
    error_str = format_str % (epoch, step, loss,
                              error['RMSE'], error_avg['RMSE'], error['RMSE_M'], error_avg['RMSE_M'],
                              error['ABS_REL'], error_avg['ABS_REL'],
                              error['DELTA1.02'], error_avg['DELTA1.02'],
                              error['DELTA1.05'], error_avg['DELTA1.05'],
                              error['DELTA1.10'], error_avg['DELTA1.10'],
                              error['DELTA1.25'], error_avg['DELTA1.25'], \
                              )
    if print_out:
        print(error_str)
    return error_str


def print_single_error(epoch, step, loss, error):
    format_str = ('%s ===>\n\
                  Epoch: %d, step: %d, loss=%.5f\n\
                  RMSE=%.4f\tRMSE_M=%.4f\tABS_REL=%.4f\n\
                  DELTA0.02=%.4f\tDELTA0.05=%.4f\tDELTA0.10=%.4f\n\
                  DELTA0.25=%.4f\tDELTA0.5=%.4f\tDELTA1=%.4f\n')
    print(format_str % ('eval_avg_error', epoch, step, loss,
                        error['RMSE'], error['RMSE_M'], error['ABS_REL'],
                        error['DELTA1.02'], error['DELTA1.05'], error['DELTA1.10'],
                        error['DELTA1.25'], error['DELTA1.25^2'], error['DELTA1.25^3']))


def log_file_folder_make(save_dir):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, 0o777)

    train_log_file = os.path.join(save_dir, 'log_train.txt')
    train_fd = open(train_log_file, 'w')
    train_fd.write('epoch\t bestModel\t RMSE\t RMSE_M\t ABS_REL\t \
                   DELTA1.02\t DELTA1.05\t DELTA1.10\t DELTA1.25\t \
                   DELTA1.25^2\t DELTA1.25^3\n')
    train_fd.close()

    eval_log_file = os.path.join(save_dir, 'log_eval.txt')
    eval_fd = open(eval_log_file, 'w')
    eval_fd.write('epoch\t bestModel\t RMSE\t RMSE_M\t ABS_REL\t \
                   DELTA1.02\t DELTA1.05\t DELTA1.10\t DELTA1.25\t \
                   DELTA1.25^2\t DELTA1.25^3\n')
    eval_fd.close()


def log_result(save_dir, error_avg, epoch, lr, best_model, split):
    format_str = '%.4f\t %.4f\t\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\n'
    train_log_file = os.path.join(save_dir, 'log_train.txt')
    eval_log_file = os.path.join(save_dir, 'log_eval.txt')
    if split == 'train':
        train_fd = open(train_log_file, 'a')
        train_fd.write(format_str % (epoch, best_model, error_avg['RMSE'], error_avg['RMSE_M'],
                                     error_avg['ABS_REL'], error_avg['DELTA1.02'], error_avg['DELTA1.05'],
                                     error_avg['DELTA1.10'], error_avg['DELTA1.25'], error_avg['DELTA1.25^2'],
                                     error_avg['DELTA1.25^3']))
        train_fd.close()
    elif split == 'eval':
        eval_fd = open(eval_log_file, 'a')
        eval_fd.write(format_str % (epoch, best_model, error_avg['RMSE'], error_avg['RMSE_M'],
                                    error_avg['ABS_REL'], error_avg['DELTA1.02'], error_avg['DELTA1.05'],
                                    error_avg['DELTA1.10'], error_avg['DELTA1.25'], error_avg['DELTA1.25^2'],
                                    error_avg['DELTA1.25^3']))
        eval_fd.close()


def log_file_folder_make_lr(save_dir):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, 0o777)
    train_log_file = os.path.join(save_dir, 'log_train.txt')
    train_fd = open(train_log_file, 'w')
    train_fd.write('epoch\t lr\t bestModel\t RMSE\t RMSE_M\t ABS_REL\t \
                       DELTA1.02\t DELTA1.05\t DELTA1.10\t DELTA1.25\t \
                       DELTA1.25^2\t DELTA1.25^3\n')
    train_fd.close()

    # train-loss
    train_loss_log_file = os.path.join(save_dir, 'log_loss_train.txt')
    train_loss_fd = open(train_loss_log_file, 'w')
    train_loss_fd.write('epoch\t step\t loss\t \
                        RMSE\t rmse\t RMSE_M\t ABS_REL\t \
                        DELTA1.02\t delta1.02\t \
                        DELTA1.05\t delta1.05\t \
                        DELTA1.10\t delta1.10\t \
                        DELTA1.25\t delta1.25\n')

    train_loss_fd.close()

    eval_log_file = os.path.join(save_dir, 'log_eval.txt')
    eval_fd = open(eval_log_file, 'w')
    eval_fd.write('epoch\t lr\t bestModel\t RMSE\t RMSE_M\t ABS_REL\t \
                       DELTA1.02\t DELTA1.05\t DELTA1.10\t DELTA1.25\t \
                       DELTA1.25^2\t DELTA1.25^3\n')
    eval_fd.close()

    # evl-loss
    eval_loss_log_file = os.path.join(save_dir, 'log_loss_eval.txt')
    eval_loss_fd = open(eval_loss_log_file, 'w')
    eval_loss_fd.write('epoch\t step\t loss\t \
                        RMSE\t mse\t RMSE_M\t ABS_REL\t \
                        DELTA1.02\t delta1.02\t \
                        DELTA1.05\t delta1.05\t \
                        DELTA1.10\t delta1.10\t \
                        DELTA1.25\t delta1.25\n')
    eval_loss_fd.close()


def log_result_lr(save_dir, error_avg, epoch, lr, best_model, split):
    format_str = '%.4f\t %.5f\t %.4f\t\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\n'
    train_log_file = os.path.join(save_dir, 'log_train.txt')
    eval_log_file = os.path.join(save_dir, 'log_eval.txt')
    if split == 'train':
        train_fd = open(train_log_file, 'a')
        train_fd.write(format_str % (epoch, lr, best_model, error_avg['RMSE'], error_avg['RMSE_M'], \
                                     error_avg['ABS_REL'], error_avg['DELTA1.02'], error_avg['DELTA1.05'], \
                                     error_avg['DELTA1.10'], error_avg['DELTA1.25'], error_avg['DELTA1.25^2'], \
                                     error_avg['DELTA1.25^3']))
        train_fd.close()
    elif split == 'eval':
        eval_fd = open(eval_log_file, 'a')
        eval_fd.write(format_str % (epoch, lr, best_model, error_avg['RMSE'], error_avg['RMSE_M'], \
                                    error_avg['ABS_REL'], error_avg['DELTA1.02'], error_avg['DELTA1.05'], \
                                    error_avg['DELTA1.10'], error_avg['DELTA1.25'], error_avg['DELTA1.25^2'], \
                                    error_avg['DELTA1.25^3']))
        eval_fd.close()


def log_loss_lr(save_dir, record_loss, split):
    train_loss_log_file = os.path.join(save_dir, 'log_loss_train.txt')
    eval_loss_log_file = os.path.join(save_dir, 'log_loss_eval.txt')
    if split == 'train':
        train_fd = open(train_loss_log_file, 'a')
        train_fd.write(record_loss)
        train_fd.close()
    elif split == 'eval':
        eval_fd = open(eval_loss_log_file, 'a')
        eval_fd.write(record_loss)
        eval_fd.close()


def save_eval_img(save_dir, index, input_depth, gt_depth, pred_depth, color=False):
    img_save_folder = os.path.join(save_dir, 'eval_result')
    if not os.path.isdir(img_save_folder):
        os.makedirs(img_save_folder, 0o777)

    save_name_depth = os.path.join(img_save_folder, "%04d_input.png" % index)
    save_name_gt = os.path.join(img_save_folder, "%04d_gt.png" % index)
    save_name_pred = os.path.join(img_save_folder, "%04d_pred.png" % index)

    save_depth = torch.squeeze(input_depth * 1.0, 0)
    save_gt = torch.squeeze(gt_depth * 1.0, 0)
    save_pred = torch.squeeze(pred_depth * 1.0, 0)

    save_depth = cv2.applyColorMap((save_depth.numpy()*255).astype(np.uint8), cv2.COLORMAP_RAINBOW)
    save_gt = cv2.applyColorMap((save_gt.numpy()*255).astype(np.uint8), cv2.COLORMAP_RAINBOW)
    save_pred = cv2.applyColorMap((save_pred.numpy()*255).astype(np.uint8), cv2.COLORMAP_RAINBOW)

    save_depth = Image.fromarray(save_depth)
    save_gt = Image.fromarray(save_gt)
    save_pred = Image.fromarray(save_pred)

    save_depth.save(save_name_depth)
    save_gt.save(save_name_gt)
    save_pred.save(save_name_pred)


def color_enhance():
    return transforms.Normalize(0.5, 0.5)


def transform_invert(img_, transform_train):
    if 'Normalize' in str(transform_train):
        norm_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform_train.transforms))
        mean = torch.tensor(norm_transform[0].mean, dtype=img_.dtype, device=img_.device)
        std = torch.tensor(norm_transform[0].std, dtype=img_.dtype, device=img_.device)
        img_.mul_(std).add_(mean)

    # img_ = np.array(img_) * 255

    return torch.IntTensor(img_)
