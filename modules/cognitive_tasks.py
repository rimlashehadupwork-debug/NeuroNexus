"""Non-diagnostic cognitive task helpers."""
from datetime import datetime

def reaction_summary(times_ms):
    vals=[float(x) for x in times_ms if x is not None and float(x)>0]
    if not vals: return {'trials':0,'median_ms':None,'variability_ms':None}
    import statistics
    return {'trials':len(vals),'median_ms':round(statistics.median(vals),1),'variability_ms':round(statistics.pstdev(vals),1) if len(vals)>1 else 0.0}
