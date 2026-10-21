"""Immutable tokenizer cache, explicitly built; never created during training."""
import json
from pathlib import Path
import numpy as np
from ..artifacts import sha256_file, write_json

def cache_key(config):
    from ..models.runtime import asset_identity
    return {'tokenizer_files':{k:v for k,v in asset_identity(config)['files'].items() if k.startswith('tokenizer/') or k.startswith('vendor/')},'data_contract':config.values['data'],'encoder_sha256':sha256_file(Path(__file__).with_name('runner.py'))}

def prepare_cache(config,*,dataset,output_dir,device='mps'):
    from ..data.dataset import load_dataset
    from ..models.runtime import load_official_assets,resolve_device
    from .runner import encode_samples
    out=config.output_path(output_dir)
    if out.exists():raise FileExistsError(out)
    manifest,samples=load_dataset(config,dataset,compact=True)
    resolved=resolve_device(device);backbone,tokenizer,_=load_official_assets(config,device=resolved)
    encoded,proof=encode_samples(tokenizer,samples,config,resolved,use_cache=False)
    out.mkdir(parents=True)
    arrays={'s1':np.stack([r['_s1'].numpy() for r in encoded]),'s2':np.stack([r['_s2'].numpy() for r in encoded]),'sample_hashes':np.asarray([r['sample_sha256'] for r in encoded],dtype='U64')}
    for name,array in arrays.items():np.save(out/(name+'.npy'),array,allow_pickle=False)
    metadata=[{k:v for k,v in row.items() if k not in ('normalized_context','normalized_future','_s1','_s2')} for row in encoded]
    (out/'validated_rows.json').write_text(json.dumps(metadata,separators=(',',':'))+'\n')
    mp=config.output_path(dataset)
    if mp.is_dir():mp=mp/'manifest.json'
    record={'schema':'frozen_token_cache_v1','dataset_manifest_sha256':sha256_file(mp),'dataset_population_sha256':manifest['population_sha256'],'bindings':cache_key(config),'prefix_proof':proof,'rows':len(encoded),'files':{name:{'path':name+'.npy','sha256':sha256_file(out/(name+'.npy'))} for name in arrays},'compact_rows_file':{'path':'validated_rows.json','sha256':sha256_file(out/'validated_rows.json')},'deep_validated_rows':len(samples),'optimizer_steps':0,'device':str(resolved)}
    write_json(config,out/'manifest.json',record)
    return record

def cached_tokens(config,samples):
    import torch
    directory=config.output_path(config.values['training']['token_cache']);m=json.loads((directory/'manifest.json').read_text())
    if m.get('schema')!='frozen_token_cache_v1' or m['bindings']!=cache_key(config):raise ValueError('Tokenizer cache bindings changed')
    if not m['prefix_proof']['prefix_token_ids_exact']:raise ValueError('Cache lacks exact prefix proof')
    if set(m['files']) != {'s1','s2','sample_hashes'}:raise ValueError('Corrupt cache file set')
    arrays={}
    for name,entry in m['files'].items():
        path=config.output_path(directory/entry['path'])
        if not path.is_relative_to(directory) or sha256_file(path)!=entry['sha256']:raise ValueError('Cache file identity mismatch')
        arrays[name]=np.load(path,allow_pickle=False,mmap_mode='r')
    mapping={str(h):i for i,h in enumerate(arrays['sample_hashes'])}
    if len(mapping)!=m['rows'] or len(arrays['s1'])!=m['rows'] or arrays['s1'].shape!=arrays['s2'].shape:raise ValueError('Corrupt token cache shape')
    if arrays['s1'].shape != (m['rows'],config.context_length+config.horizon) or arrays['s1'].dtype.kind not in 'iu' or arrays['s2'].dtype.kind not in 'iu':raise ValueError('Corrupt token cache dimensions or dtype')
    encoded=[]
    for sample in samples:
        if sample['sample_sha256'] not in mapping:raise ValueError('Sample absent from frozen tokenizer cache')
        i=mapping[sample['sample_sha256']];row=dict(sample)
        for name in ('s1','s2'):row['_'+name]=torch.from_numpy(np.array(arrays[name][i],copy=True)).long()
        encoded.append(row)
    return encoded,{'checked_samples':len(encoded),'prefix_token_ids_exact':True,'context_encoded_alone_and_with_future':True,'tokenizer_frozen':True,'cache_manifest_sha256':sha256_file(directory/'manifest.json'),'verification':'explicit_precomputed_full_prefix_proof_and_per_sample_content_hash_binding'}


def load_model_dataset(config,dataset):
    """Reuse explicitly validated cache, while verifying immutable input bytes.

    Cache-produced metadata is a derivative, not new market truth. Each use is
    bound to source shard bytes, ordered sample hashes, config and encoder.
    An immutable symbol subset may reuse its parent's exact sample identities.
    """
    from ..data.dataset import load_dataset,_hash,SCHEMA,SAMPLE_SCHEMA
    if not config.values['training'].get('token_cache'):
        return load_dataset(config,dataset)
    mp=config.output_path(dataset)
    if mp.is_dir():mp=mp/'manifest.json'
    manifest=json.loads(mp.read_text())
    if manifest.get('schema')!=SCHEMA or manifest.get('sample_schema')!=SAMPLE_SCHEMA or manifest.get('universe')!='sh_sz_a_v1':raise ValueError('Invalid cached dataset schema')
    if manifest['manifest_sha256']!=_hash({k:v for k,v in manifest.items() if k!='manifest_sha256'}) or manifest['config_sha256']!=_hash(manifest['config']):raise ValueError('Cached dataset manifest identity mismatch')
    for entry in manifest['splits'].values():
        relative=Path(entry['path']);path=config.output_path(mp.parent/relative)
        if relative.is_absolute() or '..' in relative.parts or not path.is_relative_to(mp.parent) or sha256_file(path)!=entry['sha256']:raise ValueError('Cached dataset shard changed')
    directory=config.output_path(config.values['training']['token_cache']);cache=json.loads((directory/'manifest.json').read_text())
    if cache.get('schema')!='frozen_token_cache_v1' or cache['bindings']!=cache_key(config) or cache.get('deep_validated_rows')!=cache['rows']:raise ValueError('Cache lacks matching deep validation proof')
    entry=cache['compact_rows_file'];path=config.output_path(directory/entry['path'])
    if not path.is_relative_to(directory) or sha256_file(path)!=entry['sha256']:raise ValueError('Validated compact metadata changed')
    all_rows=json.loads(path.read_text());symbols=set(manifest['symbols']);rows=[r for r in all_rows if r['symbol'] in symbols]
    if len(rows)!=manifest['total_samples'] or _hash([r['sample_sha256'] for r in rows])!=manifest['population_sha256']:raise ValueError('Cache population ordering/identities differ')
    if {r['split'] for r in rows}!=set(manifest['splits']):raise ValueError('Cached split set differs')
    if any(sum(r['split']==name for r in rows)!=entry['row_count'] for name,entry in manifest['splits'].items()):raise ValueError('Cached per-split counts differ')
    return manifest,rows
