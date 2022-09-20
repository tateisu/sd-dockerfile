#!/usr/bin/env python
import os,sys,re
import argparse
import torch
from pprint import pprint

specFile = sys.argv[1] if len(sys.argv)>1 else None

if not specFile or not os.path.isfile(specFile):
    print("usage: mixModel.py specFile")
    sys.exit(1)

outFile = None
specs=[]
with open(specFile,encoding="utf-8") as f:
    lno=0
    while True:
        ++lno
        line = f.readline()
        if line == '':
            break
        line = re.sub(r'[\x0d\x0a]+', '', line)
        line = re.sub(r'//.*', '', line)
        line = re.sub(r'\s+$', '', line)
        if line == '':
            continue

        m = re.search(r'^\s*--outFile(?:=|\s+)(.+)', line)
        if m:
            outFile = m.group(1)
            continue

        m = re.search(r'^\s*([\d.]+)\s*(.+)', line)
        if m:
            weight = float(m.group(1))
            inFile = m.group(2)
            if not os.path.isfile(inFile):
                print(f"{specFile} {lno} : missing inFile {inFile}")
                sys.exit(1)
            if weight < 0.01:
                print(f"{specFile} {lno} : too small weight. {weight} must >= 0.01")
                sys.exit(1)
            specs.append( [weight,inFile] )
            continue

        print(f"{specFile} {lno} : ?? {line}")
        sys.exit(1)

if not specs:
    print("empty merge specs.")
    sys.exit(1)
elif not outFile:
    print("outFile not specified.")
    sys.exit(1)

keys = dict()
for spec in specs:
    print(f"load {spec[1]} …")
    model = torch.load(spec[1])
    theta = model['state_dict']
    spec.append(model) # 2
    spec.append(theta) # 3
    for key in theta.keys():
        if 'model' in key:
            keys[key]=1

for key in iter(keys):
    filtered = list(filter(lambda s: key in s[3], specs))
    weightSum = sum(map( lambda s: s[0], filtered))
    value = None
    for spec in filtered:
        delta = (spec[0]/weightSum) * spec[3][key]
        if value is None:
            value = delta
        else:
            value = value + delta
    specs[0][3][key] = value

print(f"Save checkpoint: {outFile}")
torch.save(specs[0][2], outFile)
