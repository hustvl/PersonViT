import os
from config import cfg
import argparse
import logging
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from datasets import make_dataloader
from model import make_model
from utils.logger import setup_logger
from utils.metrics import R1_mAP_eval

def draw_rank(query_imgs, rank_imgs, matchs, dst_file):
    n_row = len(rank_imgs)
    if not n_row:
        return
    n_col = len(rank_imgs[0])

    fig, axs = plt.subplots(n_row, n_col+1, figsize=(2 * (n_col+1), 4 * (n_row)))
    for i in range(n_row):
        qimg = query_imgs[i]
        ax = axs[i, 0]
        ax.axis('off')
        ax.imshow(qimg)
        if i == 0:
            ax.set_title('query')
        for j in range(n_col):
            ax = axs[i, j+1]
            ax.axis('off')
            rimg = rank_imgs[i][j]
            match = matchs[i][j]
            ax.imshow(rimg)
            if i == 0:
                ax.set_title(f'{j}')

            h, w = rimg.shape[:2]
            # draw color
            linewidth=5
            rect = patches.Rectangle((0, 0), w-linewidth, h-linewidth, linewidth=linewidth,
                    edgecolor='green' if match else 'red', fill=False)
            ax.add_patch(rect)
    fig.savefig(dst_file)
    plt.clf()
    plt.close()

def torch2img(rgb_pt_img, cfg):
    mean = np.array(cfg.INPUT.PIXEL_MEAN)
    std = np.array(cfg.INPUT.PIXEL_STD)
    img = torch.einsum('chw->hwc', rgb_pt_img)
    img = torch.clip((img * std + mean) * 255, 0, 255).int()
    return img


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query,
                 outdir,
                 nrank=10,
                 nrow=5,
                 ap_thresh=0,
                 limit=0):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, distmat, pids, camids, _, _, all_AP, indices = evaluator._compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    dataset = val_loader.dataset
    for ib, bs in enumerate(range(0, num_query, nrow)):
        query_imgs = []
        rank_imgs = []
        matchs = []
        bmap = np.mean(all_AP[bs: bs+nrow])
        dst_file = os.path.join(outdir, f'{ib:05d}-{bmap:.3f}.jpg')
        if ib > limit:
            break
        for i in range(bs, min(bs+nrow, num_query)):
            qimg, qpid, qcamid = dataset[i][:3]
            query_imgs.append(torch2img(qimg, cfg))
            cur_rank_imgs = []
            cur_matchs = []
            for j in range(nrank):
                pi = num_query + indices[i, j]
                pimg, ppid, pcamid = dataset[pi][:3]
                cur_rank_imgs.append(torch2img(pimg, cfg))
                cur_matchs.append(ppid == qpid)
            rank_imgs.append(cur_rank_imgs)
            matchs.append(cur_matchs)

        draw_rank(query_imgs, rank_imgs, matchs, dst_file)
        logger.info('rank visualization is saved to %s', dst_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--nrank', type=int, default=10)
    parser.add_argument('--nrow', type=int, default=5)
    parser.add_argument('--outdir', type=str, default='vis.rank')
    parser.add_argument('--ap_thresh', type=float, default=0)
    parser.add_argument('--limit', type=int, default=500)

    args = parser.parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)
    model.load_param(cfg.TEST.WEIGHT)


    if cfg.DATASETS.NAMES == 'VehicleID':
        for trial in range(10):
            train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
            rank_1, rank5 = do_inference(cfg,
                 model,
                 val_loader,
                 num_query)
            if trial == 0:
                all_rank_1 = rank_1
                all_rank_5 = rank5
            else:
                all_rank_1 = all_rank_1 + rank_1
                all_rank_5 = all_rank_5 + rank5

            logger.info("rank_1:{}, rank_5 {} : trial : {}".format(rank_1, rank5, trial))
        logger.info("sum_rank_1:{:.1%}, sum_rank_5 {:.1%}".format(all_rank_1.sum()/10.0, all_rank_5.sum()/10.0))
    else:
       do_inference(cfg,
                 model,
                 val_loader,
                 num_query,
                 outdir,
                 args.nrank,
                 args.nrow,
                 args.ap_thresh,
                 args.limit)

