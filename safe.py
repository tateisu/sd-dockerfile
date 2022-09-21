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
    print(f'encode({args}) = {out}')
    return out

class RestrictedUnpickler(pickle.Unpickler):
    def persistent_load(self, saved_id):
        assert saved_id[0] == 'storage'
        return torch.storage._TypedStorage()

    def find_class(self, module, name):
        print(f'find class {module} {name}')
        if module == 'collections' and name == 'OrderedDict':
            return getattr(collections, name)
        if module == 'torch._utils' and name == '_rebuild_tensor_v2':
            return torch._utils._rebuild_tensor_v2
        if module == 'torch' and name in ['FloatStorage', 'HalfStorage']:
            return torch.FloatStorage
        if module == 'numpy.core.multiarray' and name == 'scalar':
            return numpy.core.multiarray.scalar
        if module == 'numpy' and name == 'dtype':
            return numpy.dtype
        if module == '_codecs' and name == 'encode':
            return encode
        # Forbid everything else.
        raise pickle.UnpicklingError("global '%s/%s' is forbidden" % (module, name))

def restricted_loads(s):
    """Helper function analogous to pickle.loads()."""
    return 

# To test that it catches this RCE:
# restricted_loads(b"cos\nsystem\n(S'echo hello world'\ntR.")

def check(f):
    bytes = f.read()
    d = RestrictedUnpickler(io.BytesIO(bytes)).load()
    print(dir(d))
    print(d.keys())
    print(d['callbacks'])

inFile = sys.argv[1] if len(sys.argv)>1 else None
if inFile is None:
    print(f"usage: {sys.argv[0]} <model.ckpt>")
    sys.exit(1)

targetEntry = 'archive/data.pkl'

# open model.ckpt as zip format
with zipfile.ZipFile(inFile) as zf:

    if targetEntry not in zf.namelist():
        print(f"missing {targetEntry} in {inFile}")
        sys.exit(1)

    with zf.open(targetEntry) as f:
        check(f)
