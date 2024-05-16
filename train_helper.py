import os
import time
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import numpy as np
from datetime import datetime
import torch.nn.functional as F
from datasets.vehicle import Trancos
from torch import tensor_split
from Networks import OSTNet
from losses.ot_loss import OT_Loss
from losses.bay_loss import Bay_Loss
from losses.post_prob import Post_Prob
from utils.pytorch_utils import Save_Handle, AverageMeter
from utils.GAME import GAME_metric
import utils.log_utils as log_utils
import wandb


def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[
        1
    ]
    gt_discretes = torch.stack(transposed_batch[2], 0)
    st_sizes = torch.FloatTensor(transposed_batch[3])
    targets = transposed_batch[4]
    return images, points, gt_discretes, st_sizes, targets


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def setup(self):
        args = self.args
        sub_dir = (
            "{}-input-{}_batch_size-{}_lr-{}_wot-{}_wtv-{}_reg-{}_nIter-{}_normCood-{}".format(
                args.run_name,
                args.crop_size,
                args.batch_size,
                args.lr,
                args.wot,
                args.wtv,
                args.reg,
                args.num_of_iter_in_ot,
                args.norm_cood,
            )
        )

        self.save_dir = os.path.join("ckpts", sub_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        time_str = datetime.strftime(datetime.now(), "%m%d-%H%M%S")
        self.logger = log_utils.get_logger(
            os.path.join(self.save_dir, "train-{:s}.log".format(time_str))
        )
        log_utils.print_config(vars(args), self.logger)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            self.logger.info("using {} gpus".format(self.device_count))
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        downsample_ratio = 8

        image_trianval_list = [img[:-1] for img in open(os.path.join(args.data_dir, 'image_sets', 'trainval.txt'))]
        image_test_list = [img[:-1] for img in open(os.path.join(args.data_dir, 'image_sets', 'test.txt'))]

        if args.dataset.lower() == "trancos":
            self.datasets = {
                "train": Trancos(
                    os.path.join(args.data_dir),
                    args.crop_size,
                    downsample_ratio,
                    "train",image_trianval_list
                ),
                "val": Trancos(
                    os.path.join(args.data_dir),
                    args.crop_size,
                    downsample_ratio,
                    "val",image_test_list
                ),
            }
        else:
            raise NotImplementedError

        self.dataloaders = {
            x: DataLoader(
                self.datasets[x],
                collate_fn=(train_collate if x ==
                            "train" else default_collate),
                batch_size=(args.batch_size if x == "train" else 1),
                shuffle=(False if x == "train" else False),
                num_workers=args.num_workers,
                pin_memory=(True if x == "train" else False),
            )
            for x in ["train", "val"]
        }
        self.model = OSTNet.alt_gvt_large()
        self.model.to(self.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        self.start_epoch = 0

        # check if wandb has to log
        if args.wandb:
            self.wandb_run = wandb.init(
            config=args, project="CTTrans", name=args.run_name
        )
        else : 
            wandb.init(mode="disabled")
    

        if args.resume:
            self.logger.info("loading pretrained model from " + args.resume)
            suf = args.resume.rsplit(".", 1)[-1]
            if suf == "tar":
                checkpoint = torch.load(args.resume, self.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(
                    checkpoint["optimizer_state_dict"])
                self.start_epoch = checkpoint["epoch"] + 1
            elif suf == "pth":
                self.model.load_state_dict(
                    torch.load(args.resume, self.device))
        else:
            self.logger.info("random initialization")

        sigma = 8.0
        background_ratio = 1.0
        use_background = True

        self.post_prob = Post_Prob(sigma,
                                   args.crop_size,
                                   downsample_ratio,
                                   background_ratio,
                                   use_background,
                                   self.device)

        self.criterion = Bay_Loss(use_background, self.device)

        self.ot_loss = OT_Loss(
            args.crop_size,
            downsample_ratio,
            args.norm_cood,
            self.device,
            args.num_of_iter_in_ot,
            args.reg,
        )
        self.tv_loss = nn.L1Loss(reduction="none").to(self.device)
        self.mse = nn.MSELoss().to(self.device)
        self.mae = nn.L1Loss().to(self.device)
        self.save_list = Save_Handle(max_num=1)
        self.best_mae = np.inf
        self.best_mse = np.inf


    def train(self):
        """training process"""
        args = self.args
        for epoch in range(self.start_epoch, args.max_epoch + 1):
            self.logger.info(
                "-" * 5 + "Epoch {}/{}".format(epoch, args.max_epoch) + "-" * 5
            )
            self.epoch = epoch
            self.train_epoch()
            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                self.val_epoch()

    def train_epoch(self):
        epoch_ot_loss = AverageMeter()
        epoch_ot_obj_value = AverageMeter()
        epoch_wd = AverageMeter()
        epoch_count_loss = AverageMeter()
        epoch_tv_loss = AverageMeter()
        epoch_bay_loss = AverageMeter()
        epoch_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_start = time.time()
        self.model.train()

        for step, (inputs, points, gt_discrete,st_sizes, targets) in enumerate(self.dataloaders["train"]):
            inputs = inputs.to(self.device)
            st_sizes = st_sizes.to(self.device)
            gd_count = np.array([len(p) for p in points], dtype=np.float32)
            points = [p.to(self.device) for p in points]
            targets = [t.to(self.device) for t in targets]
            gt_discrete = gt_discrete.to(self.device)
            N = inputs.size(0)

            with torch.set_grad_enabled(True):
                outputs, outputs_normed = self.model(inputs)

                # Compute OT loss.
                ot_loss, wd, ot_obj_value = self.ot_loss(
                    outputs_normed, outputs, points
                )
                ot_loss = ot_loss * self.args.wot # weight on OT loss
                ot_obj_value = ot_obj_value * self.args.wot
                epoch_ot_loss.update(ot_loss.item(), N)
                epoch_ot_obj_value.update(ot_obj_value.item(), N)
                epoch_wd.update(wd, N)

                # Compute counting loss.
                count_loss = self.mae(
                    outputs.sum(1).sum(1).sum(1),
                    torch.from_numpy(gd_count).float().to(self.device),
                )
                epoch_count_loss.update(count_loss.item(), N)

                # Compute TV loss.
                gd_count_tensor = (
                    torch.from_numpy(gd_count)
                    .float()
                    .to(self.device)
                    .unsqueeze(1)
                    .unsqueeze(2)
                    .unsqueeze(3)
                )
                gt_discrete_normed = gt_discrete / (gd_count_tensor + 1e-6)
                tv_loss = (
                    self.tv_loss(outputs_normed, gt_discrete_normed)
                    .sum(1)
                    .sum(1)
                    .sum(1)
                    * torch.from_numpy(gd_count).float().to(self.device)
                ).mean(0) * self.args.wtv
                epoch_tv_loss.update(tv_loss.item(), N)

                prob_list = self.post_prob(points, st_sizes)
                bay_loss = self.criterion(prob_list, targets, outputs)
                epoch_bay_loss.update(bay_loss.item(), N)

                loss = ot_loss + count_loss + tv_loss + bay_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pred_count = (
                    torch.sum(outputs.view(N, -1),
                              dim=1).detach().cpu().numpy()
                )
                pred_err = pred_count - gd_count
                epoch_loss.update(loss.item(), N)
                epoch_mse.update(np.mean(pred_err * pred_err), N)
                epoch_mae.update(np.mean(abs(pred_err)), N)

                # log wandb
                wandb.log(
                    {
                        "train/TOTAL_loss": loss,
                        "train/count_loss": count_loss,
                        "train/tv_loss": tv_loss,
                        "train/pred_err": pred_err,
                    },
                    step=self.epoch,
                )

        self.logger.info(
            "Epoch {} Train, Loss: {:.2f}, OT Loss: {:.2e}, Wass Distance: {:.2f}, OT obj value: {:.2f}, "
            "Count Loss: {:.2f}, TV Loss: {:.2f}, bay_loss: {:.2f}, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec".format(
                self.epoch,
                epoch_loss.get_avg(),
                epoch_ot_loss.get_avg(),
                epoch_wd.get_avg(),
                epoch_ot_obj_value.get_avg(),
                epoch_count_loss.get_avg(),
                epoch_tv_loss.get_avg(),
                epoch_bay_loss.get_avg(),
                np.sqrt(epoch_mse.get_avg()),
                epoch_mae.get_avg(),
                time.time() - epoch_start,
            )
        )


    def val_epoch(self):
        args = self.args
        epoch_start = time.time()
        self.model.eval()  # Set model to evaluate mode
        epoch_res = []
        epoch_GAME_0 = []
        epoch_GAME_1 = []
        epoch_GAME_2 = []
        for step, (inputs, count, name,density) in enumerate(self.dataloaders["val"]):
            with torch.no_grad():
                inputs = inputs.to(self.device)
                crop_imgs, crop_masks = [], []
                b, c, h, w = inputs.size()
                rh, rw = args.crop_size, args.crop_size
                for i in range(0, h, rh):
                    gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
                    for j in range(0, w, rw):
                        gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
                        crop_imgs.append(inputs[:, :, gis:gie, gjs:gje])
                        mask = torch.zeros([b, 1, h, w]).to(self.device)
                        mask[:, :, gis:gie, gjs:gje].fill_(1.0)
                        crop_masks.append(mask)
                crop_imgs, crop_masks = map(
                    lambda x: torch.cat(x, dim=0), (crop_imgs, crop_masks)
                )

                crop_preds = []
                nz, bz = crop_imgs.size(0), args.batch_size
                for i in range(0, nz, bz):
                    gs, gt = i, min(nz, i + bz)
                    crop_pred, _ = self.model(crop_imgs[gs:gt])

                    _, _, h1, w1 = crop_pred.size()
                    crop_pred = (
                        F.interpolate(
                            crop_pred,
                            size=(h1 * 8, w1 * 8),
                            mode="bilinear",
                            align_corners=True,
                        )
                        / 64
                    )

                    crop_preds.append(crop_pred)
                crop_preds = torch.cat(crop_preds, dim=0)

                # splice them to the original size
                idx = 0
                pred_map = torch.zeros([b, 1, h, w]).to(self.device)
                for i in range(0, h, rh):
                    gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
                    for j in range(0, w, rw):
                        gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
                        pred_map[:, :, gis:gie, gjs:gje] += crop_preds[idx]
                        idx += 1
                # for the overlapping area, compute average value
                mask = crop_masks.sum(dim=0).unsqueeze(0)
                outputs = pred_map / mask

                GAME_mini_batch_0 = GAME_metric(outputs, density, 0)
                GAME_mini_batch_1 = GAME_metric(outputs, density, 1)
                GAME_mini_batch_2 = GAME_metric(outputs, density, 2)
                epoch_GAME_0.append(GAME_mini_batch_0)
                epoch_GAME_1.append(GAME_mini_batch_1)
                epoch_GAME_2.append(GAME_mini_batch_2)

                res = count[0].item() - torch.sum(outputs).item()
                epoch_res.append(res)

        Game_0 = sum(epoch_GAME_0) / len(epoch_GAME_0)
        Game_1 = sum(epoch_GAME_1) / len(epoch_GAME_1)
        Game_2 = sum(epoch_GAME_2) / len(epoch_GAME_2)
        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))

        self.logger.info(
            "Epoch {} Val, MSE: {:.2f}, MAE: {:.2f}, epoch_Game_0: {:.2f}, epoch_Game_1: {:.2f}, epoch_Game_2: {:.2f}, Cost {:.1f} sec".format(
                self.epoch, mse, mae, Game_0, Game_1, Game_2, time.time() - epoch_start
            )
        )

        # log wandb
        wandb.log({"val/MSE": mse, "val/MAE": mae}, step=self.epoch)

        model_state_dic = self.model.state_dict()
        if mae < self.best_mae:
            self.best_mse = mse
            self.best_mae = mae
            self.logger.info(
                "save best mse {:.2f} mae {:.2f} model epoch {}".format(
                    self.best_mse, self.best_mae, self.epoch
                )
            )
            print("Saving best model at {} epoch".format(self.epoch))
            model_path = os.path.join(
                self.save_dir, "best_model_mae-{:.2f}_epoch-{}.pth".format(
                    self.best_mae, self.epoch)
            )
            if(os.path.exists("*.pth")):
                os.remove('*.pth')
            torch.save(
                model_state_dic,
                model_path,
            )
            self.save_list.append(model_path)

            if args.wandb:
                artifact = wandb.Artifact("model", type="model")
                artifact.add_file(model_path)
                
                self.wandb_run.log_artifact(artifact)



def tensor_divideByfactor(img_tensor, factor=32):
    _, _, h, w = img_tensor.size()
    h, w = int(h // factor * factor), int(w // factor * factor)
    img_tensor = F.interpolate(
        img_tensor, (h, w), mode="bilinear", align_corners=True)

    return img_tensor


def cal_new_tensor(img_tensor, min_size=256):
    _, _, h, w = img_tensor.size()
    if min(h, w) < min_size:
        ratio_h, ratio_w = min_size / h, min_size / w
        if ratio_h >= ratio_w:
            img_tensor = F.interpolate(
                img_tensor,
                (min_size, int(min_size / h * w)),
                mode="bilinear",
                align_corners=True,
            )
        else:
            img_tensor = F.interpolate(
                img_tensor,
                (int(min_size / w * h), min_size),
                mode="bilinear",
                align_corners=True,
            )
    return img_tensor


if __name__ == "__main__":
    import torch

    print(torch.__file__)
    x = torch.ones(1, 3, 768, 1152)
    y = tensor_split(x)
    print(y.size())
