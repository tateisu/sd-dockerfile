#!/usr/bin/env python3
# /h/ was here
import os
import sys
import zipfile
import builtins
import io
import pickle
import collections
import torch
import numpy
import _codecs

def encode(*args):
    out = _codecs.encode(*args)
    print(f"INFO: encode({args}) = {out}")
    return out

class RestrictedUnpickler(pickle.Unpickler):
    def persistent_load(self, saved_id):
        assert saved_id[0] == 'storage'
        return torch.storage._TypedStorage()

    def find_class(self, module, name):
        print(f"INFO: finding class {module} {name}")
        if module == 'collections' and name == 'OrderedDict':
            return getattr(collections, name)
        if module == 'torch._utils' and name == '_rebuild_tensor_v2':
            return torch._utils._rebuild_tensor_v2
        if module == 'torch' and name in ['FloatStorage', 'HalfStorage']:
            return torch.FloatStorage
        if module == 'torch' and name in ['IntStorage']:
            return torch.IntStorage
        if module == 'torch' and name in ['LongStorage']:
            return torch.LongStorage
        if module == 'numpy.core.multiarray' and name == 'scalar':
            return numpy.core.multiarray.scalar
        if module == 'numpy' and name == 'dtype':
            return numpy.dtype
        if module == '_codecs' and name == 'encode':
            return encode
        # Forbid everything else.
        raise pickle.UnpicklingError(f"forbidden: module={module}, name={name}")

# To test that it catches this RCE:
# restricted_loads(b"cos\nsystem\n(S'echo hello world'\ntR.")

def check(f,inFile):
    bytes = f.read()
    d = RestrictedUnpickler(io.BytesIO(bytes)).load()
    print(f"INFO: {inFile} {dir(d)}")
    print(f"INFO: {inFile} {d.keys()}")
    print(f"INFO: {inFile} {d['callbacks']}")

files = sys.argv[1:]
if not files:
    print(f"usage: {sys.argv[0]} <model.ckpt>")
    sys.exit(1)

for inFile in files:
    # open model.ckpt as zip format
    with zipfile.ZipFile(inFile) as zf:
        targetEntry = 'archive/data.pkl'
        if targetEntry not in zf.namelist():
            print(f"INFO: {inFile} has no {targetEntry}")
            continue
        try:
            with zf.open(targetEntry) as f:
                check(f,inFile)
        except Exception as ex:
            print(f"ERROR: {inFile} :{ex}")
