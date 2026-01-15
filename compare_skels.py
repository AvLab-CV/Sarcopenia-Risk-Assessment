import argparse
from pathlib import Path

import numpy as np

from visualize_skeleton import visualize_skeleton

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input1", type=Path)
    parser.add_argument("input2", type=Path)
    args = parser.parse_args()

    print(f"Loading from {args.input1}")
    skels1 = dict(np.load(args.input1))
    print(f"Loading from {args.input2}")
    skels2 = dict(np.load(args.input2))

    i = 0
    skel1 = skels1[list(skels1.keys())[i]]
    skel2 = skels2[list(skels2.keys())[i]]
    visualize_skeleton([skel1, skel2], [f"input1[{i}]", f"input2[{i}]"])


if __name__ == "__main__":
    main()
