import numpy as np
def auc_fast(y,s):
    # y bool array, s float
    order=np.argsort(s,kind='mergesort'); s=s[order]; y=y[order]
    n=len(s); ranks=np.empty(n,float); i=0
    while i<n:
        j=i
        while j+1<n and s[j+1]==s[i]: j+=1
        ranks[i:j+1]=(i+j)/2.0+1.0
        i=j+1
    npos=y.sum(); nneg=n-npos
    if npos==0 or nneg==0: return np.nan
    return (ranks[y].sum()-npos*(npos+1)/2)/(npos*nneg)
