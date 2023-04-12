import torch
import numpy as np
from modules.tokenizers import Tokenizer
from modules.dataloaders import R2DataLoader
from modules.metrics import compute_scores
from modules.tester import Tester
from modules.loss import compute_loss
from models.r2gen import R2GenModel
from opts import parse_agrs


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
    test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)

    # build model architecture
    model = R2GenModel(args, tokenizer)

    # get function handles of loss and metrics
    criterion = compute_loss
    metrics = compute_scores

    # build trainer and start to train
    tester = Tester(model, criterion, metrics, args, test_dataloader)
    tester.test()  # just test and get metrics
    # tester.get_tags()  # get the medical concepts predicted by the model
    # tester.plot_heatmap()  # plot the heatmap

if __name__ == '__main__':
    main()