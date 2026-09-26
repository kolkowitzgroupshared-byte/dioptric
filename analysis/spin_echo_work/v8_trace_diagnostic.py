"""Compare saved V6 and initial physical V8 traces without changing source data."""
import ast
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from analysis.spin_echo_work import sc_spin_echo_physics_fit_52G_v8b_multic13 as v8
ROOT = Path(r"\\192.168.0.197\G\nvdata\pc_NVOffice\branch_master\sc_spin_echo_physics_fit_52G_nv_pillar_array\2026_09")
ck = np.load(next(ROOT.glob("*ranked_52G_fit_checkpoint.npz")), allow_pickle=True)
t = ck["times_us"].astype(float)
y = ck["norm_counts"].astype(float)
e = ck["norm_counts_ste"].astype(float)
v6 = pd.read_csv(next(ROOT.glob("*v6_all_equal_footing_sites.csv")))
candidates = pd.read_csv(sorted(ROOT.glob("*v8b_multic13_*_subset_0-16-28-54-93_quick_candidate_fits.csv"))[-1])
catalog = json.load(open(REPO / "analysis/spin_echo_work/essem_freq_kappa_catalog_22A_52G.json"))
lookup = {(tuple(r["orientation"]), int(r["site_index"])): r for r in catalog}

def old_predict(tt, p):
    b, c, T, w, T2ms, beta, alpha, slope, chirp, a, f0, ph0, f1, ph1 = p
    carrier = np.exp(-(tt / (1000 * T2ms)) ** beta)
    comb = np.zeros_like(tt)
    for k in range(7):
        mu = k*T*(1+k*chirp)
        wk = w*(1+k*slope)
        if wk > 0:
            comb += (1+k)**(-alpha) * np.exp(-((tt-mu)/wk)**4)
    return b - c*carrier*comb + a*carrier*comb*(np.cos(2*np.pi*f0*tt+ph0)+np.cos(2*np.pi*f1*tt+ph1))

fig, axes = plt.subplots(2,2,figsize=(15,10),sharex="col")
for col, nv in enumerate([28,54]):
    ori = tuple(ast.literal_eval(v6[v6.nv_index==nv].orientation.iloc[0]))
    r6 = v6[(v6.nv_index==nv)&(v6.site_rank==1)].iloc[0]
    p6 = np.array(json.loads(r6.popt_json),float)
    rr = candidates[candidates.nv_index==nv]
    td = np.linspace(t.min(),t.max(),1800)
    ax, rz = axes[:,col]
    ax.errorbar(t,y[nv],yerr=e[nv],fmt=".",ms=4,color="black",alpha=.7)
    ax.plot(td,old_predict(td,p6),color="C0",label=f"V6 site {int(r6.site_id)}")
    for o, color in [(0,"C1"),(1,"C2"),(2,"C3")]:
        row = rr[(rr.model_order==o)&(rr.rank_within_order==1)].iloc[0]
        theta = np.array(json.loads(row.theta_json),float)
        bg,etas,dt = v8.unpack_theta(theta,o)
        sites = []
        for j in range(1,o+1):
            site = dict(lookup[ori,int(row[f"c13_{j}_site_id"])])
            site["fI_kHz"] = site["fI_Hz"]/1e3
            site["fm_kHz"] = site["omega_ms_Hz"]/1e3
            sites.append(site)
        pred = v8.model_from_parts(td,bg,sites,etas,dt)
        ax.plot(td,pred,color=color,label=f"V8 N{o} chi²r {row.red_chi2:.2f}",alpha=.9)
        rz.plot(t,(y[nv]-v8.model_from_parts(t,bg,sites,etas,dt))/e[nv],".",ms=3,color=color,alpha=.65,label=f"V8 N{o}")
    rz.plot(t,(y[nv]-old_predict(t,p6))/e[nv],".",ms=3,color="C0",alpha=.65,label="V6")
    ax.set(title=f"NV{nv}, orientation {ori}, V6 chi²r {r6.red_chi2:.2f}",ylabel="normalized contrast",ylim=(-.15,1.15))
    ax.axvspan(23,47,alpha=.04,color="grey"); ax.grid(alpha=.2); ax.legend(fontsize=8)
    rz.set(xlabel="total echo evolution (µs)",ylabel="weighted residual",ylim=(-7,7))
    rz.axhline(0,color="black",lw=.5);rz.grid(alpha=.2)
fig.tight_layout()
out = ROOT / "2026_09_23_v8b_trace_diagnostic_28_54.png"
fig.savefig(out,dpi=170)
print(out)
