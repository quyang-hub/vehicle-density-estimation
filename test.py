import argparse
import torch
import os
import numpy as np
import datasets.vehicle as vehicle
from Networks import OSTNet
from utils.GAME import GAME_metric
import torch.nn.functional as F

def tensor_divideByfactor(img_tensor, factor=32):
    _, _, h, w = img_tensor.size()
    h, w = int(h//factor*factor), int(w//factor*factor)
    img_tensor = F.interpolate(img_tensor, (h, w), mode='bilinear', align_corners=True)
    return img_tensor

def cal_new_tensor(img_tensor, min_size=256):
    _, _, h, w = img_tensor.size()
    if min(h, w) < min_size:
        ratio_h, ratio_w = min_size / h, min_size / w
        if ratio_h >= ratio_w:
            img_tensor = F.interpolate(img_tensor, (min_size, int(min_size / h * w)), mode='bilinear', align_corners=True)
        else:
            img_tensor = F.interpolate(img_tensor, (int(min_size / w * h), min_size), mode='bilinear', align_corners=True)
    return img_tensor

parser = argparse.ArgumentParser(description='Test ')
parser.add_argument('--device', default='0', help='assign device')
parser.add_argument('--batch-size', type=int, default=16,
                        help='train batch size')
parser.add_argument('--crop-size', type=int, default=256,
                    help='the crop size of the train image')
parser.add_argument('--model-path', type=str, required=True,default='ckpts/Trancos/model.pth',
                    help='saved model path')
parser.add_argument('--data-path', type=str,default='dataset/TRANCOS_v3/',
                    help='dataset path')
parser.add_argument('--dataset', type=str, default='Trancos',
                    help='dataset name: Trancos')
parser.add_argument('--pred-density-map-path', type=str, default='inference_results',
                    help='save predicted density maps when pred-density-map-path is not empty.')

def test(args, isSave = True):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device  # set vis gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = args.model_path
    crop_size = args.crop_size
    data_path = args.data_path
    image_test_list = [img[:-1] for img in open(os.path.join(args.data_path, 'image_sets', 'test.txt'))]
    dataset = vehicle.Trancos(os.path.join(data_path), crop_size, 8, method='val',image_list=image_test_list)
    dataloader = torch.utils.data.DataLoader(dataset, 1, shuffle=False,
                                             num_workers=1, pin_memory=True)

    model = OSTNet.alt_gvt_large()
    model.to(device)
    model.load_state_dict(torch.load(model_path, device))
    model.eval()
    image_errs = []
    result = []
    epoch_GAME_0 = []
    epoch_GAME_1 = []
    epoch_GAME_2 = []
    epoch_GAME_3 = []
    for inputs, count, name, density in dataloader:
        with torch.no_grad():
            inputs = inputs.to(device)
            crop_imgs, crop_masks = [], []
            b, c, h, w = inputs.size()
            rh, rw = args.crop_size, args.crop_size
            for i in range(0, h, rh):
                gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
                for j in range(0, w, rw):
                    gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
                    crop_imgs.append(inputs[:, :, gis:gie, gjs:gje])
                    mask = torch.zeros([b, 1, h, w]).to(device)
                    mask[:, :, gis:gie, gjs:gje].fill_(1.0)
                    crop_masks.append(mask)
            crop_imgs, crop_masks = map(lambda x: torch.cat(x, dim=0), (crop_imgs, crop_masks))

            crop_preds = []
            nz, bz = crop_imgs.size(0), args.batch_size
            for i in range(0, nz, bz):
                gs, gt = i, min(nz, i + bz)
                crop_pred, _ = model(crop_imgs[gs:gt])

                _, _, h1, w1 = crop_pred.size()

                crop_pred = F.interpolate(crop_pred, size=(h1 * 8, w1 * 8), mode='bilinear', align_corners=True) / 64

                crop_preds.append(crop_pred)
            crop_preds = torch.cat(crop_preds, dim=0)

            # splice them to the original size
            idx = 0
            pred_map = torch.zeros([b, 1, h, w]).to(device)
            for i in range(0, h, rh):
                gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
                for j in range(0, w, rw):
                    gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
                    pred_map[:, :, gis:gie, gjs:gje] += crop_preds[idx]
                    idx += 1
            mask = crop_masks.sum(dim=0).unsqueeze(0)
            outputs = pred_map / mask


            GAME_mini_batch_0 = GAME_metric(outputs, density, 0)
            GAME_mini_batch_1 = GAME_metric(outputs, density, 1)
            GAME_mini_batch_2 = GAME_metric(outputs, density, 2)
            GAME_mini_batch_3 = GAME_metric(outputs, density, 3)

            epoch_GAME_0.append(GAME_mini_batch_0)
            epoch_GAME_1.append(GAME_mini_batch_1)
            epoch_GAME_2.append(GAME_mini_batch_2)
            epoch_GAME_3.append(GAME_mini_batch_3)

            img_err = count[0].item() - torch.sum(outputs).item()
            print("Img name: ", name, "Error: ", img_err, "GT count: ", count[0].item(), "Model out: ", torch.sum(outputs).item())
            image_errs.append(img_err)
            result.append([name, count[0].item(), torch.sum(outputs).item(), img_err])

    Game_0 = sum(epoch_GAME_0) / len(epoch_GAME_0)
    Game_1 = sum(epoch_GAME_1) / len(epoch_GAME_1)
    Game_2 = sum(epoch_GAME_2) / len(epoch_GAME_2)
    Game_3 = sum(epoch_GAME_3) / len(epoch_GAME_3)
    image_errs = np.array(image_errs)
    mse = np.sqrt(np.mean(np.square(image_errs)))
    mae = np.mean(np.abs(image_errs))
    print('{}: mae {:.2f}, mse {:.2f}, GAME_0 {:.2f}, GAME_1 {:.2f}, GAME_2 {:.2f}, GAME_3 {:.2f}\n'.format(model_path, mae, mse,Game_0,Game_1,Game_2,Game_3))


if __name__ == '__main__':
    args = parser.parse_args()
    test(args, isSave=True)


