#!/usr/bin/env python

# from https://github.com/harubaru/waifu-diffusion/blob/main/scripts/prune.py

import os, sys, argparse, torch

def pruneModel(inFile,outFile,keepOnlyEma=False):
    inFileSize = os.path.getsize(inFile)
    print(f"{inFileSize*1e-9:.2f}GB {inFile}")


    sd = torch.load(inFile, map_location="cpu")
    print(f"loaded keys={sd.keys()}")

	# nsd にキーをコピー
    nsd = dict()
    for k in sd.keys():
        if k != "optimizer_states":
            nsd[k] = sd[k]
        else:
            print(f"removing optimizer states for path {inFile}")

    if "global_step" in sd:
        print(f"This is global step {sd['global_step']}.")
    if keepOnlyEma:
        sd = nsd["state_dict"].copy()
        # infer ema keys
        ema_keys = {k: "model_ema." + k[6:].replace(".", ".") for k in sd.keys() if k.startswith("model.")}
        new_sd = dict()

        for k in sd:
            if k in ema_keys:
                new_sd[k] = sd[ema_keys[k]].half()
            elif not k.startswith("model_ema.") or k in ["model_ema.num_updates", "model_ema.decay"]:
                new_sd[k] = sd[k].half()

        assert len(new_sd) == len(sd) - len(ema_keys)
        nsd["state_dict"] = new_sd
    else:
        sd = nsd['state_dict'].copy()
        new_sd = dict()
        for k in sd:
            new_sd[k] = sd[k].half()
        nsd['state_dict'] = new_sd

    print(f"save to… {outFile}")
    torch.save(nsd, outFile)
    outFileSize = os.path.getsize(outFile)
    print(f"{outFileSize*1e-9:.2f}GB {outFile}")

    desc = "by removing optimizer states"
    if keepOnlyEma:
        desc += " and non-EMA weights"
    reduceBytes = inFileSize - outFileSize
    print(f"{reduceBytes*1e-9:.2f} GB by {desc}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--inFile",
        help="input ckpt file.",
        type=str,
    )
    parser.add_argument(
        "--outFile",
        help="output ckpt file.",
        type=str,
    )
    parser.add_argument(
        "--keepOnlyEma",
        help="set if keepOnlyEma. default is false.",
        action='store_true',
        default=False,
    )

    args = parser.parse_args()
    if args.inFile is None:
        parser.print_help()
        sys.exit(1)

    if args.outFile is None:
         if args.keepOnlyEma:
            args.outFile = f"{os.path.splitext(args.inFile)[0]}-ema-pruned.ckpt"
         else:
            args.outFile = f"{os.path.splitext(args.inFile)[0]}-pruned.ckpt"

    pruneModel(
        inFile = args.inFile,
        outFile = args.outFile,
        keepOnlyEma = args.keepOnlyEma,
    )
