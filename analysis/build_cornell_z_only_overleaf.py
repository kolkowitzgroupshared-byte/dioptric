from pathlib import Path
import copy,csv,json,shutil,zipfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from analysis import arbitrary_gate_z_only as zo
from analysis import arbitrary_gate_tomography as arb

THREE="2026_10_27-14_57_44-johnson-nv0_2026_10_02"
DIRECT="2026_10_27-16_51_38-johnson-nv0_2026_10_02"
ROOT=Path(r"\\192.168.0.197\G\nvdata\pc_Purcell\branch_master")
INPUT=ROOT/"arbitrary_gate_tomography"/"2026_10"
OUT=ROOT/"cornell_z_only_overleaf"/"2026_10"
NAME="Cornell_Z_Only_2026_10_27_Overleaf"
THRESHOLD=2.5
SNR=3.0

def prepare():
    tri,_=arb.load_run(THREE,data_dir=INPUT)
    direct,_=arb.load_run(DIRECT,data_dir=INPUT)
    mt=tri["cornell_manifest"]
    md=direct["cornell_manifest"]
    assert (tri["tomography_mode"],mt["readout_protocol"])==("virtual_z","virtual_z_mixed_projection")
    assert (direct["tomography_mode"],md["readout_protocol"])==("virtual_z_z_only","virtual_z_z_only")
    assert mt["pi_pulse_ns"]==md["pi_pulse_ns"]==96
    assert mt["mw_chain"]==md["mw_chain"]
    assert mt["csv_sha256"]==md["csv_sha256"]
    for a,b in zip(mt["gates"],md["gates"]):
        assert a["pulses"]==b["pulses"]
        assert not a["final_projection_by_input"]["+Z"]
        assert np.isclose(a["expected_projected_z"][0],b["expected_direct_z"])
        assert np.isclose(a["predicted_projected_z"][0],b["predicted_direct_z"])
    cp=copy.deepcopy(mt)
    cp["readout_protocol"]="virtual_z_z_only"
    for g in cp["gates"]:
        g["expected_direct_z"]=g["expected_projected_z"][0]
        g["predicted_direct_z"]=g["predicted_projected_z"][0]
    reduced=dict(tri)
    reduced["tomography_mode"]="virtual_z_z_only"
    reduced["cornell_manifest"]=cp
    reduced["counts"]=np.asarray(tri["counts"])[:,[0,3],...]
    ra=zo.analyze(reduced,min_ref_contrast=THRESHOLD,min_ref_snr=SNR,reference_scope="per_gate")
    rb=zo.analyze(direct,min_ref_contrast=THRESHOLD,min_ref_snr=SNR,reference_scope="per_gate")
    names=[nv["name"] for nv in direct["nv_list"]]
    oldnames=[nv["name"] for nv in tri["nv_list"]]
    shared=[n for n in names if n in set(oldnames)]
    assert shared and len(shared)==len(set(shared))
    aa=np.array([oldnames.index(n) for n in shared])
    bb=np.array([names.index(n) for n in shared])
    a={"label":"Earlier 3-input: +Z direct Z","z":ra["measured_z"][:,aa],
       "sem":ra["z_sem"][:,aa],"valid":ra["valid_ref"][:,aa],
       "runs":tri["num_runs"],"reps":tri["num_reps"],"stem":THREE}
    b={"label":"Dedicated Z-only","z":rb["measured_z"][:,bb],
       "sem":rb["z_sem"][:,bb],"valid":rb["valid_ref"][:,bb],
       "runs":direct["num_runs"],"reps":direct["num_reps"],"stem":DIRECT}
    return shared,ra["expected_z"],ra["compiled_z"],[a,b]

def metrics(a,ideal):
    good=a["valid"]&np.isfinite(a["z"])&np.isfinite(a["sem"])
    vals=np.where(good,a["z"],np.nan)
    with np.errstate(invalid="ignore"):
        return {"accepted":int(good.sum()),"total":int(good.size),
        "nvs_all_gates":int(good.all(axis=0).sum()),
        "median_abs_error":float(np.nanmedian(np.abs(vals-ideal[:,None]))),
        "median":np.nanmedian(vals,axis=1),
        "low":np.nanpercentile(vals,25,axis=1),
        "high":np.nanpercentile(vals,75,axis=1),
        "n_by_gate":good.sum(axis=1)}

def draw_summary(path,ideal,compiled,arrays):
    x=np.arange(1,len(ideal)+1)
    fig,axs=plt.subplots(3,1,figsize=(9.6,10.5),sharex=True,
        gridspec_kw={"height_ratios":[2.0,1,0.7]},layout="constrained")
    axs[0].plot(x,ideal,"k--o",label="Ideal Rzz",lw=1.5)
    axs[0].plot(x,compiled,":s",color="0.45",label="Compiled Rzz",lw=1.3)
    for a,c in zip(arrays,["#286181","#d07733"]):
        m=metrics(a,ideal)
        axs[0].plot(x,m["median"],"-o",color=c,label=a["label"],lw=1.5)
        axs[0].fill_between(x,m["low"],m["high"],color=c,alpha=.17)
        diff=np.where(a["valid"],np.abs(a["z"]-ideal[:,None]),np.nan)
        axs[1].plot(x,np.nanmedian(diff,axis=1),"-o",color=c,label=a["label"])
        axs[2].plot(x,m["n_by_gate"],"-o",color=c,label=a["label"])
    axs[0].set_ylabel(r"Direct $\langle\sigma_z\rangle$")
    axs[0].set_title("Cornell Experiment 0: +Z input and direct Z output only")
    axs[0].legend(fontsize=8,ncol=2)
    axs[1].set_ylabel(r"Median $|Z-Z_{\mathrm{ideal}}|$")
    axs[2].set_ylabel("Accepted NVs")
    axs[2].set_xlabel("Cornell gate")
    axs[2].set_xticks(x)
    for ax in axs: ax.grid(alpha=.16)
    fig.savefig(path,bbox_inches="tight")
    plt.close(fig)

def draw_heatmap(path,ideal,arrays):
    fig,axes=plt.subplots(1,2,figsize=(11,6.5),sharex=True,sharey=True,layout="constrained")
    for ax,a in zip(axes,arrays):
        heat=np.where(a["valid"],np.abs(a["z"]-ideal[:,None]),np.nan)
        im=ax.imshow(heat.T,origin="lower",aspect="auto",vmin=0,vmax=2,
                     interpolation="nearest")
        ax.set_title(a["label"])
        ax.set_xticks(np.arange(len(ideal)),labels=list(range(1,len(ideal)+1)))
        ax.set_xlabel("Cornell gate")
        ax.set_ylabel("Common NV index")
    fig.colorbar(im,ax=axes.tolist(),shrink=.8,label="Absolute direct Z error")
    fig.savefig(path,bbox_inches="tight")
    plt.close(fig)

def draw_nvs(path,ideal,compiled,arrays,names):
    xx=np.arange(1,len(ideal)+1)
    with PdfPages(path,metadata={"Title":"Cornell Experiment 0: direct Z individual NVs"}) as out:
        for i,name in enumerate(names):
            fig,ax=plt.subplots(figsize=(8.4,5.5),layout="constrained")
            ax.plot(xx,ideal,"k--o",lw=1.4,ms=3,label="Ideal Z")
            ax.plot(xx,compiled,":s",color="0.45",lw=1.3,ms=3,label="Compiled Z")
            highest=1.6
            for a,c,shift in zip(arrays,["#286181","#d07733"],[-.07,.07]):
                valid=a["valid"][:,i] & np.isfinite(a["z"][:,i]) & np.isfinite(a["sem"][:,i])
                y=a["z"][:,i];sem=a["sem"][:,i]
                if valid.any():
                    ax.errorbar(xx[valid]+shift,y[valid],yerr=sem[valid],
                        fmt="o",ms=4,capsize=2.2,color=c,ecolor=c,label=a["label"]+" (1 SEM)")
                    highest=max(highest,float(np.nanmax(np.abs(y[valid])+sem[valid]))+.25)
                if (~valid).any():
                    ax.scatter(xx[~valid]+shift,np.full((~valid).sum(),-1.36),
                        color=c,marker="x",s=18)
            ax.set_ylim(-min(highest,5),min(highest,5))
            ax.set_xticks(xx)
            ax.set_xlim(.5,len(xx)+.5)
            ax.set_xlabel("Cornell gate")
            ax.set_ylabel(r"Direct $\langle\sigma_z\rangle$")
            ax.set_title(f"NV {i+1}/{len(names)}: {name}")
            ax.grid(alpha=.2);ax.legend(fontsize=8)
            fig.text(.11,.023,f"Only +Z input and direct Z readout; contrast >= {THRESHOLD:g} photons, SNR >= {SNR:g}.",fontsize=8)
            out.savefig(fig,bbox_inches="tight",pad_inches=.18)
            plt.close(fig)

def tex_document(n,measures):
    a,b=measures
    pre=r"""\documentclass[11pt,letterpaper]{article}
\usepackage[margin=0.84in]{geometry}
\usepackage{amsmath,amssymb,booktabs,graphicx,xcolor,caption}
\usepackage{hyperref,pdfpages,microtype}
\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue}
\graphicspath{{figures/}}
\newif\ifallNVs
\allNVstrue % Change to \allNVsfalse to omit the 119-page appendix
\title{\textbf{Cornell Experiment 0: Direct-Z Responses}\\
\large Optical $|0\rangle$ preparation and SCC readout}
\author{Purcell Widefield NV Platform --- Experimental Analysis}
\date{October 27, 2026 acquisitions; compiled October 28, 2026}
\begin{document}
\maketitle
\begin{abstract}
This report examines only the direct-Z population response of the ten Cornell arbitrary gates, from optical $|0\rangle$ input to SCC readout. Two independently acquired datasets are displayed separately: the earlier three-input acquisition, restricted to its unprojected +Z channel, and the later dedicated Z-only acquisition. No +X or +Y state results are included. Error bars on individual-NV curves are propagated $\pm1$ SEM. The experiment tests $R_{zz}$ population transfer, not complete arbitrary-gate or phase fidelity.
\end{abstract}
\section{Protocol and theoretical target}
The measured sequence is optical spin polarization, compiled resonant microwave gate, SCC, and camera photon counting. No initial or final $\pi/2$ pulse is used for the +Z signal. For gate $U_g$,
\begin{equation}
R_{zz}^{(g)}=\langle 0|U_g^\dagger\sigma_zU_g|0\rangle,\quad
U_g=\exp[-i\Omega_g\hat{\mathbf n}_g\cdot\boldsymbol{\sigma}].
\end{equation}
If $\hat{\mathbf n}_g$ has polar angle $\theta_g$,
\begin{equation}
R_{zz,\mathrm{ideal}}^{(g)}
=\cos(2\Omega_g)+(1-\cos(2\Omega_g))\cos^2\theta_g.
\end{equation}
Rotations about Z leave this observable unchanged, so virtual-Z phase errors cannot be independently identified by this experiment.
\section{Source acquisitions}
\begin{table}[h]\centering\small
\begin{tabular}{@{}p{1.15in}p{3.0in}rrr@{}}\toprule
Protocol & Timestamp and sample & Runs & Reps & Accepted pairs\\\midrule
"""
    pre+=f"Earlier mixed & \\texttt{{\\detokenize{{{THREE}}}}} & 10 & 200 & {a['accepted']}/{a['total']}\\\\\n"
    pre+=f"Dedicated Z & \\texttt{{\\detokenize{{{DIRECT}}}}} & 10 & 200 & {b['accepted']}/{b['total']}\\\\\n"
    pre+=r"""\bottomrule\end{tabular}
\caption{Each acquisition has 119 NVs and ten gate measurements. The earlier raw file stored three signals and a dual-rail reference; only experiment 0 (+Z direct Z) and experiment 3 (reference) are used. The dedicated file stored one signal and one reference.}
\end{table}
Both acquisitions use microwave chain 1 and the same 96 ns $\pi$-pulse calibration and compiled gates. They were not interleaved and should not be interpreted as a causal before/after comparison.
\section{SCC normalization and uncertainty}
For signal photon counts $S$ and even/odd reference counts $R_0,R_1$, each gate and NV is normalized within its original file:
\begin{equation}
Z=1-2\frac{S-R_0}{R_1-R_0},\qquad D=R_1-R_0.
\end{equation}
With independent means' standard errors $s_S,s_0,s_1$, first-order error propagation gives
\begin{equation}
s_Z^2=\left(\frac{2s_S}{D}\right)^2
+\left(\frac{2(R_1-S)s_0}{D^2}\right)^2
+\left(\frac{2(S-R_0)s_1}{D^2}\right)^2.
\end{equation}
This report uses per-gate references, minimum contrast of 2.5 photons and minimum reference SNR of 3. Points failing those conditions are excluded. Out-of-range normalized Z estimates are not artificially clipped.
\section{Results: Z only}
\begin{figure}[h!]\centering
\includegraphics[width=.95\linewidth]{z_summary.pdf}
\caption{Top: Cornell ideal and compiled Z responses versus median measured direct Z, with interquartile ranges across NVs (shaded; not SEM). Middle: median absolute Z error per gate. Bottom: accepted NV count.}
\end{figure}
\clearpage
\begin{table}[h!]\centering
\begin{tabular}{@{}lrrr@{}}\toprule
Acquisition & Accepted gate-NV pairs & Median $|Z-Z_{\rm ideal}|$ & NVs with 10 valid gates\\\midrule
"""
    pre+=f"Earlier +Z subset & {a['accepted']}/{a['total']} & {a['median_abs_error']:.3f} & {a['nvs_all_gates']}\\\\\n"
    pre+=f"Dedicated Z-only & {b['accepted']}/{b['total']} & {b['median_abs_error']:.3f} & {b['nvs_all_gates']}\\\\\n"
    pre+=r"""\bottomrule\end{tabular}
\caption{Descriptive population response statistics only.}
\end{table}
\begin{figure}[h!]\centering
\includegraphics[width=.95\linewidth]{z_error_heatmaps.pdf}
\caption{Per-NV absolute error in direct Z, per gate. White cells have rejected/nonfinite SCC references. Both panels use the same scale.}
\end{figure}
\clearpage
\section{Scientific interpretation}
Direct-Z measurements of optical $|0\rangle$ inputs constrain the population transfer of a compiled gate and reveal microwave amplitude, timing, spin-polarization or SCC-reference errors. They do not verify relative phases or provide process tomography. Comparing these two non-interleaved acquisitions is descriptive only.
\section{Reproducibility}
The ZIP archive includes source filenames, normalized direct-Z observations and propagated error bars for every NV and gate in data/direct_z_points.csv, the per-gate statistics and the original figures. Each run was normalized independently. The appendix contains direct-Z error bars for all common NVs and no +X/+Y panels.
\ifallNVs
\clearpage
\section{Individual-NV direct-Z responses}
Colored points show measured direct Z with $\pm1$ SEM; crosses at the bottom indicate rejected references. Black and grey lines represent ideal and compiled responses.
\includepdf[pages=-,pagecommand={},fitpaper=false]{figures/individual_nv_z_only.pdf}
\fi
\end{document}
"""
    return pre

def main():
    names,ideal,compiled,arrays=prepare()
    OUT.mkdir(parents=True,exist_ok=True)
    project=OUT/NAME
    project.mkdir(parents=True,exist_ok=True)
    (project/"figures").mkdir(exist_ok=True)
    (project/"data").mkdir(exist_ok=True)
    results=[metrics(a,ideal) for a in arrays]
    (project/"main.tex").write_text(tex_document(len(names),results),encoding="utf-8")
    (project/"README.md").write_text(
        "Upload the ZIP to Overleaf via New Project > Upload Project. Main file: main.tex.\n"
        "Compile with pdfLaTeX. To omit individual NV appendix, change allNVstrue to allNVsfalse.\n"
        f"Sources: {THREE} (only first signal and fourth reference), {DIRECT} (first signal and second reference).\n"
        "No +X or +Y response plots appear. Data filters: contrast 2.5 photons, SNR 3.0.\n",
        encoding="utf-8")
    with (project/"data"/"direct_z_points.csv").open("w",newline="",encoding="utf-8") as f:
        wr=csv.writer(f)
        wr.writerow(["nv_index","nv_name","gate","source","measured_z","one_sem","valid","ideal_z","compiled_z"])
        for i,name in enumerate(names):
            for gate in range(len(ideal)):
                for a in arrays:
                    wr.writerow([i,name,gate+1,a["stem"],a["z"][gate,i],
                                 a["sem"][gate,i],bool(a["valid"][gate,i]),ideal[gate],compiled[gate]])
    with (project/"data"/"gate_statistics.csv").open("w",newline="",encoding="utf-8") as f:
        wr=csv.writer(f)
        wr.writerow(["source","gate","ideal","compiled","median_z","q25","q75","accepted_nv"])
        for a,r in zip(arrays,results):
            for i in range(len(ideal)):
                wr.writerow([a["stem"],i+1,ideal[i],compiled[i],r["median"][i],
                             r["low"][i],r["high"][i],r["n_by_gate"][i]])
    summary={"sample":"johnson NV array","num_nvs":len(names),"num_gates":len(ideal),
             "sources":[{"file":a["stem"],"label":a["label"],"runs":a["runs"],
                         "reps":a["reps"],"accepted":r["accepted"],"total":r["total"],
                         "median_absolute_z_error":r["median_abs_error"],
                         "nvs_all_gates":r["nvs_all_gates"]}
                        for a,r in zip(arrays,results)],
             "min_ref_contrast":THRESHOLD,"min_ref_snr":SNR,
             "disclaimer":"Z-only population responses do not establish full gate fidelity."}
    (project/"data"/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("SAVING SUMMARY...",flush=True)
    summarypdf=project/"figures"/"z_summary.pdf"
    draw_summary(summarypdf,ideal,compiled,arrays)
    print("SAVING HEATMAP...",flush=True)
    draw_heatmap(project/"figures"/"z_error_heatmaps.pdf",ideal,arrays)
    print("SAVING",len(names),"NV PAGES...",flush=True)
    draw_nvs(project/"figures"/"individual_nv_z_only.pdf",ideal,compiled,arrays,names)
    archive=OUT/(NAME+".zip")
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as out:
        for f in project.rglob("*"):
            if f.is_file():out.write(f,arcname=f.relative_to(project))
    shutil.copyfile(summarypdf,OUT/(NAME+"_summary_preview.pdf"))
    print("BUNDLE",archive,archive.stat().st_size,"bytes")
    print("PREVIEW",OUT/(NAME+"_summary_preview.pdf"))
    print("FULL_INDIVIDUAL_PAGES",project/"figures"/"individual_nv_z_only.pdf")
    print("ACCEPTED","three",results[0]["accepted"],"direct",results[1]["accepted"])
    print("MEDIAN_ERR","three",results[0]["median_abs_error"],"direct",results[1]["median_abs_error"])
if __name__=="__main__":
    main()

