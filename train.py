import time
import json
import os 
import wandb
import logging
from collections import OrderedDict
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
from typing import List
from anomalib.utils.metrics import AUPRO, AUROC
from sklearn.metrics import average_precision_score,roc_auc_score
from data.copypaste import dynamic_copy_paste
_logger = logging.getLogger('train')
import cv2
from models.deviation_loss import DeviationLoss
import torch.autograd as autograd
autograd.set_detect_anomaly(True)
from torch import nn
import torchmetrics

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def read_image(outlier_data):
    res_outlier_data_image=torch.Tensor()
    res_outlier_data_mask=torch.Tensor()
    for item in outlier_data:
        img = cv2.imread(item)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, dsize=(256, 256))
        mask = cv2.imread(
            item.replace('test', 'ground_truth').replace('.png', '_mask.png').replace('.bmp','.bmp').replace('image','mask'),
            cv2.IMREAD_GRAYSCALE
        )
        # print(item.replace('test', 'ground_truth').replace('.jpg', '.png').replace('.bmp','.bmp'),
        #     )
        # print(mask.shape)
        mask = cv2.resize(mask, dsize=(256,256)).astype(np.bool).astype(np.int)
        tmp=torch.tensor(img).permute(2,0,1).unsqueeze(0)
        res_outlier_data_image=torch.cat([res_outlier_data_image,tmp/255.0],dim=0)
        res_outlier_data_mask=torch.cat([res_outlier_data_mask,torch.tensor(mask).unsqueeze(0)],dim=0)

    return res_outlier_data_image,res_outlier_data_mask

def training(model, trainloader, validloader, criterion, optimizer, scheduler, num_training_steps: int = 1000, loss_weights: List[float] = [0.6, 0.4], 
             log_interval: int = 1, eval_interval: int = 1, savedir: str = None, use_wandb: bool = False, device: str ='cpu') -> dict:   

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    l1_losses_m = AverageMeter()
    focal_losses_m = AverageMeter()
    
    # metrics
    auroc_image_metric = AUROC(num_classes=1, pos_label=1)
    auroc_pixel_metric = AUROC(num_classes=1, pos_label=1)
    aupro_pixel_metric = AUPRO()
    image_ap_metric = torchmetrics.AveragePrecision(pos_label=1)
    pixel_ap_metric = torchmetrics.AveragePrecision(pos_label=1)
    image_f1_score_metric = torchmetrics.F1Score()
    pixel_f1_score_metric = torchmetrics.F1Score()
    # criterion
    l1_criterion, focal_criterion = criterion
    l1_weight, focal_weight = loss_weights

    model.train()

    # set optimizer
    optimizer.zero_grad()

    # training
    best_score = 0
    step = 0
    train_mode = True
    outlier_data=trainloader.dataset.outlier_data
    outlier_data_image,outlier_data_mask=read_image(trainloader.dataset.outlier_data)
    while train_mode:
        end = time.time()
        for inputs, masks, targets in trainloader:
            # batch
        # for i_batch, sample_batched in enumerate(trainloader):
            inputs, masks, targets = inputs.to(device), masks.to(device), targets.to(device)
            inputs, masks,targets=dynamic_copy_paste( inputs, masks, targets,
                                                      outlier_data_image,outlier_data_mask
                                                      ,outlier_data)
            # inputs,masks,targets=sample_batched['image'].cuda(),sample_batched['mask'].cuda(),sample_batched['has_anomaly'].cuda()
            # masks = masks[:, 0, :]
            print()
            # targets = targets[:, 0]
            data_time_m.update(time.time() - end)

            # predict
            outputs = model(inputs,masks)
            # for i in range(16):
            #     ...: plt.imshow(masks[i].unsqueeze(0).permute(1, 2, 0).detach().cpu())
            #     ...: plt.show()
            #     ...: plt.imshow(outputs[i][1].unsqueeze(0).permute(1, 2, 0).detach().cpu())
            #     ...: plt.show()

            outputs = F.softmax(outputs, dim=1)
            l1_loss = l1_criterion(outputs[:, 1, :], masks)
            focal_loss = focal_criterion(outputs, masks.long())
            deviationLoss = DeviationLoss()
            # loss2 = deviationLoss(outputs, masks.unsqueeze(1))
            # loss3=loss_fn(classifier,targets.cuda())


            loss = (l1_weight * l1_loss) + (focal_weight * focal_loss)

            loss.backward()

            # update weight
            optimizer.step()
            optimizer.zero_grad()

            # log loss
            l1_losses_m.update(l1_loss.item())
            focal_losses_m.update(focal_loss.item())
            losses_m.update(loss.item())

            batch_time_m.update(time.time() - end)

            if (step+1) % log_interval == 0 or step == 0:
                _logger.info('TRAIN [{:>4d}/{}] '
                            'Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) '
                            'L1 Loss: {l1_loss.val:>6.4f} ({l1_loss.avg:>6.4f}) '
                            'Focal Loss: {focal_loss.val:>6.4f} ({focal_loss.avg:>6.4f}) '
                            'LR: {lr:.3e} '
                            'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                            'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                            step+1, num_training_steps,
                            loss       = losses_m,
                            l1_loss    = l1_losses_m,
                            focal_loss = focal_losses_m,
                            lr         = optimizer.param_groups[0]['lr'],
                            batch_time = batch_time_m,
                            rate       = inputs.size(0) / batch_time_m.val,
                            rate_avg   = inputs.size(0) / batch_time_m.avg,
                            data_time  = data_time_m))

            if ((step+1) % eval_interval == 0 and step != 0) or (step+1) == num_training_steps:
                eval_metrics = evaluate(
                    model        = model,
                    dataloader   = validloader,
                    criterion    = criterion,
                    log_interval = log_interval,
                    metrics      = [auroc_image_metric, auroc_pixel_metric, aupro_pixel_metric,image_ap_metric,pixel_ap_metric,image_f1_score_metric,pixel_f1_score_metric],
                    device       = device
                )
                model.train()

                # eval_log = dict([(f'eval_{k}', v) for k, v in eval_metrics.items()])

                # wandb
                # if use_wandb:
                #     wandb.log(eval_log, step=step)

                #  np.mean(list(eval_metrics.values()))
                if best_score <np.mean(list(eval_metrics.values())):
                    # save best score
                    state = {'best_step':step}
                    # state.update(eval_log)
                    json.dump(state, open(os.path.join(savedir, 'best_score.json'),'w'), indent='\t')

                    # save best model
                    torch.save(model.state_dict(), os.path.join(savedir, f'best_model.pt'))


                    torch.save(model.memory_bank, os.path.join(savedir, f'memory_bank.pt'))

                    _logger.info('Best Score {0:.3%} to {1:.3%}'.format(best_score, np.mean(list(eval_metrics.values()))))

                    best_score = np.mean(list(eval_metrics.values()))


            # scheduler
            if scheduler:
                scheduler.step()

            end = time.time()

            step += 1

            if step == num_training_steps:
                train_mode = False
                break

    # print best score and step
    _logger.info('Best Metric: {0:.3%} (step {1:})'.format(best_score, state['best_step']))

    # save latest model
    torch.save(model.state_dict(), os.path.join(savedir, f'latest_model.pt'))

    # save latest score
    state = {'latest_step':step}
    state.update(eval_log)
    json.dump(state, open(os.path.join(savedir, 'latest_score.json'),'w'), indent='\t')

    

        
def evaluate(model, dataloader, criterion, log_interval, metrics: list, device: str = 'cpu'):

    # metrics
    auroc_image_metric, auroc_pixel_metric, aupro_pixel_metric ,image_ap_metric,pixel_ap_metric,image_f1_score_metric,pixel_f1_score_metric= metrics

    # reset
    auroc_image_metric.reset(); auroc_pixel_metric.reset(); aupro_pixel_metric.reset()
    image_ap_metric.reset();pixel_ap_metric.reset();image_f1_score_metric.reset();pixel_f1_score_metric.reset()
    model.eval()

    best_threshold = 0.0
    min_missed_samples = len(dataloader.dataset)  # 初始值设置为数据集总数
    all_anomaly_scores = []
    all_targets = []
    all_outputs=[]
    all_images = []
    all_masks=[]
    misclassified_images = []
    save_dir=r'E:\mvtecprediction\small_mysdd'
    with torch.no_grad():
        for idx,(inputs, masks, targets) in enumerate(dataloader):
            inputs, masks, targets = inputs.to(device), masks.to(device), targets.to(device)
            # predict
        # for i_batch, sample_batched in enumerate(dataloader):
        #     inputs, masks, targets = sample_batched['image'].cuda(), sample_batched['mask'].cuda(), sample_batched[
        #         'has_anomaly'].cuda()
        #     masks = masks[:, 0, :]
        #     targets = targets[:, 0]
            outputs = model( inputs,masks)
            outputs = F.softmax(outputs, dim=1)
            for i in range(inputs.size(0)):
                image = inputs[i].cpu().numpy().transpose(1, 2, 0)
                # output = outputs[i, 1].cpu().numpy()    # 假设是分类任务
                # mask= masks[i].cpu().numpy()
                #
                # image_name = f"prediction_{i_batch * dataloader.batch_size + i}.png"
                # image_path = os.path.join(save_dir, image_name)
                #
                # heatmap = plt.get_cmap('jet')(output)
                # heatmap = np.delete(heatmap, 3, 2)
                # overlay = 0.6 * image + 0.4 * heatmap
                # 这里假设你使用matplotlib库来保存图片

                # plt.figure()
                # plt.subplot(1, 3, 1)
                # plt.axis('off')
                # plt.imshow(image)
                # plt.title("Input Image")
                #
                # plt.subplot(1, 3, 2)
                # plt.axis('off')
                # plt.imshow(overlay,interpolation='bilinear')
                # plt.title("Predicted Heatmap")
                #
                # plt.subplot(1, 3, 3)
                # plt.imshow(mask, cmap='gray')
                # plt.axis('off')
                # plt.title("Ground Truth Mask")
                #
                # plt.savefig(image_path)
                # plt.close()
            k=100

            anomaly_score = torch.topk(torch.flatten(outputs[:,1,:], start_dim=1), k)[0].mean(dim=1)
            all_anomaly_scores.append(anomaly_score)
            all_targets.append(targets)
            all_outputs.append(outputs[:,1,:])
            all_images.append(inputs)
            all_masks.append(masks)

            # anomaly_score=classifier.argmax(dim=1)
            # update metrics
            auroc_image_metric.update(
                preds  = anomaly_score.cpu(),
                target = targets.cpu()
            )
            auroc_pixel_metric.update(
                preds  = outputs[:,1,:].cpu(),
                target = masks.cpu()
            )
            aupro_pixel_metric.update(
                preds   = outputs[:,1,:].cpu(),
                target  = masks.cpu()
            )

            image_ap_metric.update(
                preds  = anomaly_score.cpu(),
                target = targets.cpu()
            )
            pixel_ap_metric.update(
                preds=outputs[:, 1,:].cpu(),
                target=masks.cpu()
            )
            image_f1_score_metric.update(
                preds=(anomaly_score>0.5).int().cpu(),
                target=targets.cpu()
            )
            pixel_f1_score_metric.update(
                preds=(outputs[:, 1,:].flatten()>0.5).int().cpu(),
                target=masks.flatten().cpu()
            )
    # imageAp = average_precision_score(
    #     torch.cat(all_targets).view(-1).cpu().numpy(),
    #     torch.cat(all_anomaly_scores).view(-1).cpu().numpy()
    # )
    # imageauroc = roc_auc_score(
    #     torch.cat(all_targets).view(-1).cpu().numpy(),
    #     torch.cat(all_anomaly_scores).view(-1).cpu().numpy()
    # )
    #
    # auroc_pixel = roc_auc_score(
    #     torch.cat(all_masks).view(-1).cpu().numpy(),
    #     torch.cat(all_outputs).view(-1).cpu().numpy()
    # )
    # ap_pixel=average_precision_score(
    #     torch.cat(all_masks).view(-1).cpu().numpy(),
    #     torch.cat(all_outputs).view(-1).cpu().numpy()
    # )
    # print("AUC Image:  " + str(imageauroc))
    # print("AP Image:  " + str(imageAp))
    # print("AUC Pixel:  " + str(auroc_pixel))
    # print("AP Pixel:  " + str(ap_pixel))
    # metrics
    metrics = {
        'AUROC-image':auroc_image_metric.compute().item(),
        'AUROC-pixel':auroc_pixel_metric.compute().item(),
        'AUPRO-pixel':aupro_pixel_metric.compute().item(),
        'image_AP':image_ap_metric.compute().item(),
        'pixel_AP':pixel_ap_metric.compute().item(),
        'image_F1Score' : image_f1_score_metric.compute().item(),
        'pixel_f1score': pixel_f1_score_metric.compute().item()
    }

    _logger.info('TEST: AUROC-image: %.3f%% | AUROC-pixel: %.3f%% | AUPRO-pixel: %.3f%%'
                 ' | image_AP: %.3f%%| image_F1: %.3f%% | pixel_AP: %.3f%%| pixel_F1: %.3f%%'  %
                (metrics['AUROC-image'], metrics['AUROC-pixel'], metrics['AUPRO-pixel'],
                 metrics['image_AP'],metrics['image_F1Score'],metrics['pixel_AP'],metrics['pixel_f1score']))

    # _logger.info('TEST:  AUROC-pixel: %.3f%% ' %
    #              (metrics['AUROC-pixel']))

    return metrics


# all_anomaly_scores = np.array(all_anomaly_scores)
    # all_targets = np.array(all_targets)
    # # 从正样本中找到最小的置信度（异常分数）
    # min_confidence_positive = np.min(all_anomaly_scores[all_targets == 1])
    # # 找到所有大于或等于该置信度的样本
    # predicted_positive_indices = np.where(all_anomaly_scores >= min_confidence_positive)[0]
    # # 计算误报的数量
    # false_positive_count = np.sum(all_targets[predicted_positive_indices] == 0)
    # # 计算误报的比例
    # false_positive_rate = false_positive_count / len(predicted_positive_indices)
    # print(f"误报的比例为：{false_positive_rate:.2f}")
    # desired_fnr = 0.70 # 假设您可以接受的最大漏报率为5%
    # best_threshold_fnr = 0
    # thresholds = torch.linspace(1, 0, 100).to(device)  # 100是示例，你可以根据需要调整步长和数量
    # for thresh in thresholds:
    #     predicted_anomalies = np.array(all_anomaly_scores) > np.array(thresh.cpu())
    #     actual_anomalies = np.array(all_targets) == 1  # 假设异常的标签为1
    #     actual_normals = np.array(all_targets) == 0  # 假设正常的标签为0
    #     TP = np.sum(np.logical_and(predicted_anomalies, actual_anomalies))
    #     FN = np.sum(np.logical_and(~predicted_anomalies, actual_anomalies))
    #     FP = np.sum(np.logical_and(predicted_anomalies, actual_normals))
    #     TN = np.sum(np.logical_and(~predicted_anomalies, actual_normals))
    #
    #     TPR = TP / (TP + FN)
    #     FPR = FP / (FP + TN)
    #     FNR = FN / (TP + FN)
    #     TNR = TN / (FP + TN)
    #     missed_samples = FN
    #
    #     # if missed_samples < min_missed_samples:
    #     # min_missed_samples = missed_samples
    #     # best_threshold = thresh
    #     if FNR <= desired_fnr:
    #         best_threshold_fnr = thresh
    #         break
    # misclassified_images = []
    # misclassified_outputs = []
    #
    # for i in range(len(all_targets)):
    #     # if predicted_anomalies[i] and actual_normals[i]:  # 误报的情况
    #     #     misclassified_images.append(all_images[i])
    #     if not predicted_anomalies[i] and actual_anomalies[i]:  # 漏报的情况，特别是正例但异常分低的
    #         misclassified_images.append(all_images[i])
    #         misclassified_outputs.append(all_outputs[i])
    #
    # import matplotlib.pyplot as plt
    #
    # for i, (orig_img, pred_img) in enumerate(zip(misclassified_images, misclassified_outputs)):
    #     plt.figure(figsize=(10, 5))
    #
    #     # 原始图像
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(orig_img[0], cmap='gray')  # 假设图片是单通道的
    #     plt.title(f"Original Image {i + 1}")
    #
    #     # 模型预测的输出
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(pred_img[1], cmap='gray')  # 假设您想显示第二个通道的输出，根据实际情况调整
    #     plt.title(f"Predicted Output {i + 1}")
    #
    #     plt.show()
    #     # print(missed_samples)
    #     # print(best_threshold)
    #     # print("TPR (True Positive Rate):", TPR)
    #     # print("FPR (False Positive Rate):", FPR)
    #     # print("FNR (False Negative Rate):", FNR)
    #     # print("TNR (True Negative Rate):", TNR)
    #
