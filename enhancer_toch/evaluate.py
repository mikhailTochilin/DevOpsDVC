
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import logging
import sys
import os
from pystoi import stoi
import torch

from .data import NoisyCleanSet
from .enhance import add_flags, get_estimate
from . import distrib, pretrained
from .utils import bold, LogProgress

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
        'enhancer_toch.evaluate',
        description='Speech enhancement using Demucs - Evaluate model performance')
add_flags(parser)
parser.add_argument('--data_dir', help='directory including noisy.json and clean.json files')
parser.add_argument('--matching', default="sort", help='set this to dns for the dns dataset.')
parser.add_argument('-v', '--verbose', action='store_const', const=logging.DEBUG,
                    default=logging.INFO, help="More loggging")


def evaluate(args, model=None, data_loader=None):
    total_stoi = 0
    total_cnt = 0
    updates = 5

    # Load model
    if not model:
        model = pretrained.get_model(args).to(args.device)
    model.eval()

    # Load data
    if data_loader is None:
        dataset = NoisyCleanSet(args.data_dir,
                                matching=args.matching, sample_rate=model.sample_rate)
        data_loader = distrib.loader(dataset, batch_size=1, num_workers=2)
    pendings = []
    with ProcessPoolExecutor(args.num_workers) as pool:
        with torch.no_grad():
            iterator = LogProgress(logger, data_loader, name="Eval estimates")
            for i, data in enumerate(iterator):
                # Get batch data
                noisy, clean = [x.to(args.device) for x in data]
                # If device is CPU, we do parallel evaluation in each CPU worker.
                if args.device == 'cpu':
                    pendings.append(
                        pool.submit(_estimate_and_run_metrics, clean, model, noisy, args))
                else:
                    estimate = get_estimate(model, noisy, args)
                    estimate = estimate.cpu()
                    clean = clean.cpu()
                    pendings.append(
                        pool.submit(_run_metrics, clean, estimate, args, model.sample_rate))
                total_cnt += clean.shape[0]

        for pending in LogProgress(logger, pendings, updates, name="Eval metrics"):
            stoi_i = pending.result()

            total_stoi += stoi_i

    metrics = [total_stoi]
    res = distrib.average([m/total_cnt for m in metrics], total_cnt)
    logger.info(bold(f'Test set performance: STOI={res[0]}.'))
    return res[0]


def _estimate_and_run_metrics(clean, model, noisy, args):
    estimate = get_estimate(model, noisy, args)
    return _run_metrics(clean, estimate, args, sr=model.sample_rate)


def _run_metrics(clean, estimate, args, sr):
    estimate = estimate.numpy()[:, 0]
    clean = clean.numpy()[:, 0]
    stoi_i = get_stoi(clean, estimate, sr=sr)
    return stoi_i



def get_stoi(ref_sig, out_sig, sr):
    """Calculate STOI.
    Args:
        ref_sig: numpy.ndarray, [B, T]
        out_sig: numpy.ndarray, [B, T]
    Returns:
        STOI
    """
    stoi_val = 0
    for i in range(len(ref_sig)):
        stoi_val += stoi(ref_sig[i], out_sig[i], sr, extended=False)
    return stoi_val


def main():
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stderr, level=args.verbose)
    logger.debug(args)
    stoi = evaluate(args)

    js = {"stoi": stoi}

    with open("metrics.json", 'w') as file:
        json.dump(js, file, indent=2)
    # json.dump({'pesq': pesq, 'stoi': stoi}, sys.stdout)
    # sys.stdout.write('\n')


if __name__ == '__main__':
    main()
