"""Run a fresh compact control linked to an explicitly named main experiment."""
import argparse
from pathlib import Path

from .v3_compact_optimization import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--main',type=Path,required=True,
                        help='fresh main run; launch compact registration before its held-out scoring starts')
    args = parser.parse_args()
    run(args.dataset,args.output,main=args.main)


if __name__ == '__main__':
    main()
