"""Post-validation memory compaction and causal normalization side information."""
import numpy as np

def context_features(sample):
    """Only origin context stats: close-relative means/stds and log scale."""
    stats=sample['normalization'];mean=np.asarray(stats['mean'],dtype=float);std=np.asarray(stats['std'],dtype=float)
    close=float(sample['raw_context'][-1][3]);epsilon=float(stats['epsilon'])
    # Price ratios preserve percent scale; log levels explicitly restore the
    # missing coordinates. Volume/amount scales are causal and bounded.
    values=np.r_[(mean[:4]/close-1)*10.,std[:4]/close*10.,np.log1p(mean)/10.,np.log1p(std)/10.]
    if close<=0 or not np.isfinite(values).all():raise ValueError('Invalid causal context statistics')
    return np.clip(values,-5,5).astype(np.float32).tolist()

def compact_sample(sample):
    drop={'raw_context','raw_future','provenance','quality','context_dates','future_dates','normalization'}
    result={k:v for k,v in sample.items() if k not in drop}
    result['context_features']=context_features(sample)
    for field in ('normalized_context','normalized_future'):result[field]=np.asarray(sample[field],dtype=np.float32)
    keep={'schema','schema_sha256','continuous_names','continuous_targets','continuous_mask','event_names','event_targets','event_mask','structure_targets','structure_mask'}
    result['teachers']={name:{k:v for k,v in teacher.items() if k in keep} for name,teacher in sample['teachers'].items()}
    return result
