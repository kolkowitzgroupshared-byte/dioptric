import numpy as np
from scipy.signal import lombscargle, find_peaks, savgol_filter
root=r"\\192.168.0.197\G\nvdata\pc_NVOffice\branch_master\sc_spin_echo_physics_fit_52G_nv_pillar_array\2026_09"
z=np.load(root+r"\2026_09_21-21_09_41-spin_echo_old_protocol_ranked_52G_fit_checkpoint.npz",allow_pickle=True)
t=z["times_us"].astype(float)
y=z["norm_counts"].astype(float)
m=(t>=24)&(t<=47.5)
tt=t[m]
freq=np.linspace(5,1000,20000)
omega=2*np.pi*freq/1000.0
print("window",len(tt),tt.min(),tt.max(),np.median(np.diff(tt)))
for nv in [0,16,28,54,101]:
    yy=y[nv,m]
    smooth=savgol_filter(yy,21,3)
    resid=yy-smooth
    power=lombscargle(tt,resid-resid.mean(),omega,normalize=True)
    inds,_=find_peaks(power,distance=150)
    inds=inds[np.argsort(power[inds])[-12:]][::-1]
    print("NV",nv,[(round(freq[i],1),round(power[i],3)) for i in inds[:8]])
