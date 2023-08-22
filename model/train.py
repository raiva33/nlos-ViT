import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'

import random
import logging
import numpy as np
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
import torch.distributed as dist
from tensorboardX import SummaryWriter
#import lpips
import cv2
import matplotlib.pyplot as plt
from model import Rec_Transformer
from utils.scheduler import WarmupCosineSchedule
from utils.data_utils import get_loader

writer = SummaryWriter('runs/test_hole_local_bo')
logger = logging.getLogger(__name__)


class AverageMeter(object):
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


def save_model(cfg, model):
    torch.save(model.state_dict(), cfg.dir.save_model_dir)
    # torch.save(model.state_dict(), cfg.basic.save_model_dir)


def load_model(cfg, model):
    loaded_dict = torch.load(cfg.dir.load_model_dir)
    model_dict = model.state_dict()
    loaded_dict = {k: v for k, v in loaded_dict.items() if k in model_dict}
    model_dict.update(loaded_dict)
    model.load_state_dict(model_dict)


def setup(cfg):
    model = Rec_Transformer()
    num_params = count_parameters(model)
    logger.info("Total Parameter: \t%2.1fM" % num_params)

    if cfg.train.load == True:
        load_model(cfg, model)

    return model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000


def set_seed(cfg):
    torch.manual_seed(cfg.basic.seed)
    torch.cuda.manual_seed_all(cfg.basic.seed)
    np.random.seed(cfg.basic.seed)
    random.seed(cfg.basic.seed)
    torch.backends.cudnn.deterministic = True


def valid(cfg, model, val_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()
    model.eval()
    epoch_iterator = tqdm(val_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True)
    MSE = torch.nn.MSELoss()
    MSE.cuda()
    #LPIPS = lpips.LPIPS(net='alex')
    #LPIPS.cuda()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.cuda() for t in batch)
        x, y = batch
        with torch.no_grad():
            outputs = model(x)
            MSE_loss=MSE(outputs, y.to(torch.float))
            #LPIPS_loss=LPIPS(outputs, y.to(torch.float))
            #eval_loss = cfg.loss.MSE_t * MSE_loss + cfg.loss.LPIPS_t * LPIPS_loss
            eval_loss = cfg.loss.MSE_t * MSE_loss
            eval_losses.update(eval_loss.item())

        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)


    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)

    writer.add_scalar("Validation_Loss", eval_losses.avg, global_step)


    return eval_losses.avg


def train(cfg):
    """ Train the model """
    model = setup(cfg)
    #freezing layers
    '''
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.Decoder.parameters():
        parameter.requires_grad = True
    '''
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(trainable_params)
    untrainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad==False)
    print(untrainable_params)

    model.cuda()
    #model = torch.nn.DataParallel(model)

    # Prepare dataset
    train_loader, val_loader = get_loader(cfg)

    # Prepare optimizer and scheduler
    if cfg.optimizer.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=cfg.optimizer.learning_rate,
                                    momentum=0.9,
                                    weight_decay=cfg.optimizer.weight_decay)
    if cfg.optimizer.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=cfg.optimizer.learning_rate,
                                      weight_decay=cfg.optimizer.weight_decay)
    t_total = cfg.train.num_steps

    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=cfg.scheduler.warmup_steps, t_total=t_total)

    model.zero_grad()
    losses = AverageMeter()
    MSE = torch.nn.MSELoss()
    MSE.cuda()
    #LPIPS = lpips.LPIPS(net='alex')
    #LPIPS.cuda()
    global_step=0
    best_losses = 999999

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", cfg.train.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", cfg.train.train_batch_size)
    count = 0
    while True:
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True)
        
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.cuda() for t in batch)

            x, y = batch
            print(x.shape,y.shape)

            outputs = model(x)
            MSE_loss = MSE(outputs, y.to(torch.float))            
            #LPIPS_loss = LPIPS(outputs, y.to(torch.float))
            #loss = cfg.loss.MSE_t * MSE_loss + cfg.loss.LPIPS_t * LPIPS_loss
            loss = cfg.loss.MSE_t * MSE_loss
            loss.mean().backward()
            losses.update(loss.mean().item())

            
            writer.add_scalar("Training_Loss", losses.val, global_step)

            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if cfg.scheduler.use==True:
                scheduler.step()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            epoch_iterator.set_description(
                "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
            )
            if global_step % cfg.train.eval_every == 0:
                eval_losses=valid(cfg, model, val_loader, global_step)
                save_model(cfg, model)
                best_losses = eval_losses

                model.train()

            if global_step % t_total == 0:
                break
        losses.reset()

        if global_step % t_total == 0:
            break

    logger.info("Best Loss: \t%f" % best_losses)
    logger.info("End Training!")
    output = outputs.to('cpu').detach().numpy().copy()
    print("---------------------")
    print(output.shape)
    for i in range(output.shape[0]):
        img = output[i]
        img = np.squeeze(img)
        print(img.shape)
        plt.imshow(img)
        f_name = "/code/end_training" + str(i) + ".png"
        plt.savefig(f_name)

    
    print(y.shape)
    print(y.to(torch.float).shape)
    gt = y.to('cpu').detach().numpy().copy()
    for i in range(gt.shape[0]):
        img = gt[i]
        img = np.squeeze(img)
        print(img.shape)
        plt.imshow(img)
        f_name = "/code/gt" + str(i) + ".png"
        plt.savefig(f_name)
    print("------------")

def main():
    cfg = OmegaConf.load('./configs.yaml')
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    set_seed(cfg)
    train(cfg)

if __name__ == "__main__":
    main()