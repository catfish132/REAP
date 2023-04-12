import torch
import argparse
import numpy as np
from modules.tokenizers import Tokenizer
from modules.dataloaders import R2DataLoader
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler, build_noamopt_optimizer, build_plateau_optimizer
from modules.trainer_rl import Trainer
from modules.loss import compute_loss, RewardCriterion
from models.r2gen import R2GenModel
from opts import *


def main():
    # parse arguments
    args = parse_agrs()

    # fix random seeds
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # create tokenizer
    tokenizer = Tokenizer(args)

    # create data loader
    train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=True)
    val_dataloader = R2DataLoader(args, tokenizer, split='val', shuffle=False)
    test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)

    # build model architecture
    model = R2GenModel(args, tokenizer)

    # get function handles of loss and metrics
    criterion = RewardCriterion()
    metrics = compute_scores

    # build optimizer, learning rate scheduler
    ve_optimizer, ed_optimizer = build_plateau_optimizer(args, model)
    # lr_scheduler = build_lr_scheduler(args, optimizer)

    # build trainer and start to train
    trainer = Trainer(model, criterion, metrics, ve_optimizer, ed_optimizer, args, train_dataloader, val_dataloader, test_dataloader)
    trainer.train()

if __name__ == '__main__':
    main()
