import argparse
from pathlib import Path

import numpy as np

from seq_transformation import seq_transformation


def npz_dict_to_test_partition(
    skels: dict[str, np.ndarray],
    *,
    label: int = 0,
) -> dict:
    keys = sorted(skels.keys())
    test_X = [skels[k] for k in keys]
    test_Y = [int(label) for _ in range(len(test_X))]
    test_clips = keys
    return dict(test_X=test_X, test_Y=test_Y, test_clips=test_clips)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_npz",
        type=Path,
        help="Single .npz containing a dict of skeleton arrays (key -> np.ndarray).",
    )
    parser.add_argument(
        "output_npz",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    input_npz: Path = args.input_npz
    output_npz: Path = args.output_npz

    if input_npz.suffix != ".npz":
        raise SystemExit(f"Expected a .npz input, got: {input_npz}")

    with np.load(input_npz, allow_pickle=True) as npz:
        skels = {k: npz[k] for k in npz.files}

    pkl = npz_dict_to_test_partition(skels, label=0)
    seq_npz = seq_transformation(pkl)

    print(f"{input_npz} -> {output_npz}")
    np.savez(output_npz, **seq_npz)


if __name__ == "__main__":
    main()

