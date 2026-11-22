"""Explicitly acquire pinned public assets outside Git; optional for supplied local assets."""
from pathlib import Path
from .data import digest

KEV_REPO='jaredpalmer/kev-0.8b'
KEV_REV='9a45d25eb2ab761841196625383fa1dff0e56c1e'
BASE_REPO='Qwen/Qwen3.5-0.8B-Base'
BASE_REV='dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68'
KEV_CODE='65b1e763943ace53fd3a4aae00d8d08dbd849772'
FILES=('adapter_config.json','adapter_model.safetensors','tokenizer_config.json','tokenizer.json')
ORIGINAL_DIGESTS={
    'head.pt':'f400bd12802b2b105ae45d6b03774a158a3db4fccff42413734ddca2e5c920b6',
    'adapter_model.safetensors':'9b908623acb162118575f4e7a94524f9c139c335be4bfb74d6cfceca01e1885a',
    'tokenizer.json':'06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523',
}
BASE_DIGEST='c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c'

def fetch(root):
    from huggingface_hub import snapshot_download
    base=root/'qwen-base';kev=root/'official-kev'
    snapshot_download(BASE_REPO,revision=BASE_REV,local_dir=base,allow_patterns=['config.json','tokenizer.json','tokenizer_config.json','*.safetensors','*.safetensors.index.json'])
    snapshot_download(KEV_REPO,revision=KEV_REV,local_dir=kev,allow_patterns=['head.pt',*FILES])
    for n,sha in ORIGINAL_DIGESTS.items():
        if digest(kev/n)!=sha:raise ValueError('Pinned Kev asset digest mismatch: '+n)
    if digest(base/'model.safetensors')!=BASE_DIGEST:raise ValueError('Pinned base weight digest mismatch')
    return {'base':str(base),'checkpoint':str(kev)}
