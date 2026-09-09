"""Lightweight artifact identities shared by CPU and training environments."""
import hashlib
SOURCE_SHA = '313d615ca34c8ac3a183384e1f87e8274d748e5445af110013a335f02ae42e32'

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()
