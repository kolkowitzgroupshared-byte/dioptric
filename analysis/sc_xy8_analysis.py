# -*- coding: utf-8 -*-
"""
Widefield xy experiment
Created on November 29th, 2023
@author: Saroj Chand
"""

import time
import sys
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
from scipy.optimize import curve_fit, least_squares
import utils.tool_belt as tb
from majorroutines.widefield import base_routine
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import widefield


def stretched_exp(tau, a, t2, n, b):
    n = 3.0
    return a * (1 - np.exp(-((tau / t2) ** n))) + b


def residuals(params, x, y, yerr):
    return (stretched_exp(x, *params) - y) / yerr


def process_and_fit_xy8(nv_list, taus, norm_counts, norm_counts_ste):
    num_nvs = len(nv_list)
    T2_list = []
    n_list = []
    chi2_list = []
    param_errors_list = []
    fit_params = []
    for nv_ind in range(num_nvs):
        nv_counts = norm_counts[nv_ind]
        nv_counts_ste = norm_counts_ste[nv_ind]

        # Optional skip for low contrast
        # if np.ptp(nv_counts) < 0.05 or np.mean(nv_counts_ste) > 0.2:
        #     print(f"NV {nv_ind} skipped: low contrast or noisy")
        #     continue

        # Initial guesses
        a0 = np.clip(np.ptp(nv_counts), 0.1, 1.0)
        t2_0 = np.median(taus)
        n0 = 1.0
        # b0 = np.min(nv_counts)
        b0 = np.mean(nv_counts[-4:])
        p0 = [a0, t2_0, n0, b0]

        bounds = (
            [0, 1e-1, 0.01, -10.0],
            [1.5, 1e5, 11.0, 10.0],
        )
        try:
            popt, pcov = curve_fit(
                stretched_exp,
                taus,
                nv_counts,
                p0=p0,
                bounds=bounds,
                sigma=nv_counts_ste,
                absolute_sigma=True,
                maxfev=20000,
            )

            # χ²
            residuals = stretched_exp(taus, *popt) - nv_counts
            red_chi_sq = np.sum((residuals / nv_counts_ste) ** 2) / (
                len(taus) - len(popt)
            )
            # Parameter uncertainties
            param_errors = np.sqrt(np.diag(pcov))

            T2_list.append(popt[1])
            n_list.append(popt[2])
            fit_params.append(popt)
            chi2_list.append(red_chi_sq)
            param_errors_list.append(param_errors)
            print(
                f"NV {nv_ind} - T2 = {popt[1]:.1f} ns, n = {popt[2]:.2f}, χ² = {red_chi_sq:.2f}"
            )

        except Exception as e:
            print(f"NV {nv_ind} fit failed: {e}")
            continue

    return (
        fit_params,
        T2_list,
        n_list,
        chi2_list,
        param_errors_list,
    )


def process_and_plot_xy8(nv_list, taus, norm_counts, norm_counts_ste, seq_xy):
    num_nvs = len(nv_list)
    T2_list = []
    n_list = []
    nv_indices = []
    chi2_list = []
    T2_errs = []
    fit_params = []
    contrast_list = []
    b_list = []
    a_list = []
    for nv_ind in range(num_nvs):
        nv_counts = norm_counts[nv_ind]
        nv_counts_ste = norm_counts_ste[nv_ind]
        max_counts = np.max(nv_counts)
        # Skip low contrast or high noise
        # if np.ptp(nv_counts) < 0.05 or np.mean(nv_counts_ste) > 0.1:
        #     print(f"NV {nv_ind} skipped: low contrast or noisy")
        #     continue

        a0 = np.clip(np.ptp(nv_counts), 0.1, 1.0)
        t2_0 = np.median(taus)
        n0 = 1.0
        b0 = np.min(nv_counts)
        p0 = [a0, t2_0, n0, b0]

        bounds = (
            [0, 1e-1, 0.01, -10.0],
            [1.5, 8e3, 6.0, 10.0],
        )  # Lower bounds  # Upper bounds

        try:
            popt, pcov = curve_fit(
                stretched_exp,
                taus,
                nv_counts,
                p0=p0,
                bounds=bounds,
                sigma=nv_counts_ste,
                absolute_sigma=True,
                maxfev=60000,
            )

            residuals = stretched_exp(taus, *popt) - nv_counts
            red_chi_sq = np.sum((residuals / nv_counts_ste) ** 2) / (
                len(taus) - len(popt)
            )
            # if red_chi_sq > 1.0:
            #     pcov *= red_chi_sq
            param_errors = np.sqrt(np.diag(pcov))
            # if red_chi_sq > 10 or np.isnan(popt).any():
            #     print(f"NV {nv_ind} rejected: high χ² or NaNs")
            #     continue
            contrast_list.append(popt[0])
            fit_params.append(popt)
            T2_list.append(popt[1])
            n_list.append(popt[2])
            nv_indices.append(nv_ind)
            chi2_list.append(red_chi_sq)
            T2_errs.append(param_errors[1])
            b_list.append(popt[3])
            a_list.append(popt[0])
            T2 = round(popt[1], 1)
            n = round(popt[2], 2)
            b = round(popt[3], 2)
            print(f"NV {nv_ind} - T2 = {T2} us, n = {n}, b= {b}, χ² = {red_chi_sq:.2f}")
        except Exception as e:
            print(f"NV {nv_ind} fit failed: {e}")
            # continue
        # fit funtions
        # fit_funtion = lambda x: stretched_exp(x, *popt)
        # fit_functions.append(fit_funtion)
        # plotting
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.errorbar(
            taus,
            nv_counts,
            yerr=np.abs(nv_counts_ste),
            fmt="o",
            capsize=3,
            label=f"NV {nv_ind}",
        )
        # fit funtions
        # if popt is not None:
        #     tau_fit = tau_fit = np.logspace(
        #         np.log10(min(taus)), np.log10(max(taus)), 200
        #     )
        #     fit_vals = max_counts - stretched_exp(tau_fit, *popt)
        #     ax.plot(tau_fit, fit_vals, "-", label="Fit")
        ax.set_title(
            f"{seq_xy} : NV {nv_ind} - T2 = {T2} µs, n = {n}", fontsize=15
        )
        ax.set_xlabel("τ (µs)", fontsize=15)
        ax.set_ylabel("Norm. NV⁻ Population", fontsize=15)
        ax.tick_params(axis="both", labelsize=15)
        # ax.set_xscale("symlog", linthresh=1e5)
        # ax.set_xscale("log")
        # ax.set_yscale("log")
        # ax.legend()
        # ax.grid(True)
        # ax.spines["right"].set_visible(False)
        # ax.spines["top"].set_visible(False)
        plt.show(block=True)
    ## plot contrast

    print(f"list_T2 = {T2_list}")
    print(f"list_b = {b_list}")
    print(f"list_a = {a_list}")
    # sys.exit()
    fig, ax = plt.subplots(figsize=(6, 5))
    nv_indices_T1 = np.arange(len(contrast_list))
    ax.bar(
        nv_indices_T1,
        contrast_list,
        color="teal",
        edgecolor="k",
        alpha=0.7,
        label="T1",
    )

    ax.set_xlabel("NV Index", fontsize=14)
    ax.set_ylabel("Contrast", fontsize=14)
    ax.set_title("XY8 Fit Contrast per NV", fontsize=15)
    ax.tick_params(labelsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    # plt.tight_layout()
    plt.show()

    ## plot T2
    # Define outliers
    # outlier_indices = [0, 9, 10, 36]
    # Create a copy of T2_list for plotting with clipped or masked values
    T2_list_plot = T2_list.copy()
    T2_clipped_vals = []

    # Replace outlier T2s with np.nan or a capped value for visualization
    # for i in outlier_indices:
    #     T2_clipped_vals.append(T2_list_plot[i])
    #     T2_list_plot[i] = np.nan  # or use: T2_list_plot[i] = np.percentile(T2_list, 95)

    median_T2_us = np.median(T2_list)
    print(f"median T2: {median_T2_us}us")
    nv_indices = np.arange(num_nvs)
    # sys.exit()
    ##plot
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(
        nv_indices,
        T2_list_plot,
        c=chi2_list,
        s=40,
        edgecolors="blue",
    )
    # Error bars using errorbar (no color map here)
    if T2_errs is not None:
        ax.errorbar(
            nv_indices,
            T2_list_plot,
            yerr=T2_errs,
            fmt="none",
            ecolor="gray",
            elinewidth=1,
            capsize=3,
            alpha=0.7,
            zorder=1,
        )
    # Add median line
    ax.axhline(
        median_T2_us,
        color="r",
        linestyle="--",
        linewidth=0.8,
        label=f"Median T2 ≈ {median_T2_us:.1f} µs",
    )

    # # Annotate χ² values if provided
    # if n_list is not None:
    #     for idx, n in zip(nv_indices, n_list):
    #         ax.annotate(
    #             f"n={n:.2f}",
    #             (idx, T2_list_plot[idx]),
    #             textcoords="b_list points",
    #             xytext=(0, 5),
    #             ha="center",
    #             fontsize=6,
    #             color="black",
    #         )
    # Stretching exponent note
    ax.text(
        0.01,
        0.95,
        "n = 1.0",
        transform=ax.transAxes,
        fontsize=10,
        color="dimgray",
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="whitesmoke"),
    )
    # Labels and formatting
    ax.set_xlabel("NV Index", fontsize=15)
    ax.set_ylabel("T2 (µs)", fontsize=15)
    ax.tick_params(axis="both", which="major", labelsize=15)
    # ax.set_yscale("log")
    ax.set_title(f"T2  per NV ({seq_xy}-1, 68 MHz Orientation)", fontsize=15)
    ax.grid(True, which="both", ls="--", alpha=0.6)
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Reduced χ²", fontsize=15)
    cbar.ax.tick_params(labelsize=15)
    ax.legend(fontsize=11)
    plt.tight_layout
    plt.show()
    # sys.exit()
    ### plot all
    # sns.set(style="whitegrid")
    # num_cols = 8
    # num_nvs = len(nv_list)
    # num_rows = int(np.ceil(num_nvs / num_cols))
    # # Full plot
    # fig, axes = plt.subplots(
    #     num_rows,
    #     num_cols,
    #     figsize=(num_cols * 1.8, num_rows * 3),
    #     sharex=True,
    #     sharey=False,
    #     constrained_layout=True,
    #     gridspec_kw={"wspace": 0.0, "hspace": 0.0},
    # )
    # axes = axes.flatten()
    # # axes = axes[::-1]
    # taus_fit = np.logspace(np.log10(min(taus)), np.log10(max(taus)), 200)
    # for nv_idx, ax in enumerate(axes):
    #     if nv_idx >= len(norm_counts):  # Safe bound check
    #         ax.axis("off")
    #         continue

    #     nv_counts = norm_counts[nv_idx]
    #     max_counts = np.max(nv_counts)

    #     sns.scatterplot(
    #         x=taus,
    #         y=max_counts - nv_counts + b_list[nv_idx],
    #         ax=ax,
    #         color="blue",
    #         # label=f"NV {nv_idx}(T2 = {T2_list[nv_idx]:.2f} ± {T2_errs[nv_idx]:.2f} us)",
    #         label=f"NV {nv_idx}(T2 = {T2_list[nv_idx]:.2f} us)",
    #         s=10,
    #         alpha=0.7,
    #     )
    #     # Plot error bars separately for clarity
    #     ax.errorbar(
    #         taus,
    #         max_counts - nv_counts + b_list[nv_idx],
    #         yerr=norm_counts_ste[nv_idx],
    #         fmt="o",
    #         alpha=0.9,
    #         ecolor="gray",
    #         markersize=0.1,
    #     )
    #     # Plot fitted curve if available
    #     popt = fit_params[nv_idx]
    #     if popt is not None:
    #         fit_vals = max_counts - stretched_exp(taus_fit, *popt) + b_list[nv_idx]
    #         sns.lineplot(
    #             x=taus_fit,
    #             y=fit_vals,
    #             ax=ax,
    #             # color="blue",
    #             # label='Fit',
    #             lw=1,
    #         )
    #     ax.legend(fontsize="xx-small")
    #     ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    #     # ax.tick_params(labelleft=False)
    #     # Set yticks to min and max of current y-data
    #     y_min, y_max = np.min(max_counts - nv_counts + b_list[nv_idx]), np.max(
    #         max_counts - nv_counts + b_list[nv_idx]
    #     )
    #     # y_min, y_max = b_list[nv_idx], a_list[nv_idx] + b_list[nv_idx]
    #     # ax.set_yticks([round(y_min, 1), round(y_max, 1)])
    #     # ax.set_yticklabels([f"{y_min:.1f}", f"{y_max:.1f}"])
    #     # ax.tick_params(labelleft=True, labelsize=8, color="blue", pad=0)
    #     # Move tick labels inside the plot
    #     # Adjust label position and alignment

    #     # Set y-tick formatter to 2 decimal places
    #     ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    #     ax.tick_params(axis="y", labelsize=8, direction="in", pad=-10)
    #     for label in ax.get_yticklabels():
    #         label.set_horizontalalignment("right")
    #         label.set_x(0.01)  # Fine-tune this as needed
    #         label.set_zorder(100)
    #     # Set y-tick formatter to 2 decimal places
    #     # ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    #     # # Adjust label position and alignment
    #     # for label in ax.get_yticklabels():
    #     #     label.set_horizontalalignment("right")
    #     #     label.set_x(0.15)  # Fine-tune this as needed
    #     # for label in ax.get_yticklabels():
    #     #     label.set_horizontalalignment("right")
    #     #     label.set_x(0.15)
    #     # Optional: adjust tick line size and font
    #     ax.set_xscale("log")
    #     # # Add NV index within the plot at the center
    #     axes_grid = np.array(axes).reshape((num_rows, num_cols))
    #     # Loop over each column
    #     for col in range(num_cols):
    #         # Go from bottom row upwards
    #         for row in reversed(range(num_rows)):
    #             if row * num_cols + col < len(axes):  # Check if subplot exists
    #                 ax = axes_grid[row, col]
    #                 # Apply ticks
    #                 # tick_positions = np.logspace(
    #                 #     np.log10(taus[0]), np.log10(taus[-1] ), 4
    #                 # )
    #                 # ax.set_xticks(tick_positions)
    #                 # ax.set_xticklabels(
    #                 #     [f"{tick:.0f}" for tick in tick_positions],
    #                 #     rotation=45,
    #                 #     fontsize=9,
    #                 # )
    #                 for label in ax.get_xticklabels():
    #                     label.set_y(0.01)
    #                 ax.set_xlim(
    #                     4,
    #                 )
    #                 ax.tick_params(axis="x", labelsize=9, pad=0)
    #                 ax.set_xlabel("Time (μs)", fontsize=11, labelpad=1)
    #                 break  # Done for this column

    # fig.text(
    #     0.005, 0.5, "NV$^{-}$ Population", va="center", rotation="vertical", fontsize=11
    # )
    # fig.suptitle(f"{seq_xy}-1 Fits ({all_file_ids_str})", fontsize=12)
    # fig.tight_layout(pad=0.4, rect=[0.01, 0.01, 0.99, 0.99])
    plt.show(block=True)


def contrast_plot():
    # fmt:off
    contrast_list_T1 = [1.4371898492290658, 1.9855801828512756, 1.189948127119836, 1.1367476486414552, 2.0971847975297577, 1.0784495454684617, 2.061270960870385, 0.6557912312980084, 1.0970016329065504, 0.9086577559992178, 1.454423765860711, 1.072604032150468, 1.3139609887718264, 0.9823402724043178, 0.9273340413971323, 0.9646318715309131, 1.0326198663756367, 0.5502675304984411, 0.8246732536652994, 1.336547335910902, 1.1602295593668301, 1.7429553322370919, 0.606875538379173, 1.565779215998888, 1.6951645571068565, 0.7187507290940647, 2.171690151958945, 1.4974879677398811, 1.5407869182088632, 0.34014153979504363, 1.1094557327352172, 1.5791989767480112, 1.0299690912703412, 0.9848842254073304, 0.8056769816690995, 0.5652958122528015, 1.1225283700800104, 1.3500640082119226, 1.5289842904327884, 1.2878676785699383, 0.7998587608222976, 1.2843089676866188, 1.4541038671143658, 1.8071104322355616, 1.2002773642744078, 0.7250266406978021, 1.0362652289344805, 1.0582838452547434, 1.1908165409611722, 1.1486888187538424, 0.8464202073587321, 1.6341050197571885, 1.1030000142672916, 2.260828780908791, 0.6144139408216054, 1.050733951289048, 1.163328463251989, 0.8421395480075121, 1.2066417769076572, 0.5091025014571177, 1.420515897378421, 0.5486035585658687, 0.9241221548410897, 1.5225170715278613, 0.5322537375181111, 0.5650302604446495, 0.6763075545243682, 0.4529550045080959, 1.3211476494873053, 0.867683377411849, 1.0031772594435013, 1.9128573285063342, 0.5076066077437745]
    contrast_list_T2_68MHz = [8.615028305090697e-12, 0.15402741249984986, 0.2810018448147397, 0.2148108791573274, 0.32729816092045044, 0.1748456775577632, 0.22065194636964822, 0.17282968213543004, 0.22195626116574954, 0.2762961901167496, 0.2633045748885817, 0.21413573539075853, 0.18832479923828607, 0.23515144519684783, 0.21629180338633122, 0.24884199584283204, 0.16906125460663868, 0.32874156459566745, 0.26883214673393807, 0.18024179403910393, 0.24474271122697588, 0.24716077783748686, 0.23151505524616017, 0.23051755131187915, 0.23390189503472675, 0.2617489740116334, 0.28150921118420386, 0.20095601817630318, 0.23861061441762252, 0.18869488354581054, 0.2056954226810145, 0.2749228708648838, 0.140282414374628, 0.16338672888384143, 0.2516154197400596, 0.20227730297252133, 0.10902897835600525, 0.1832204564177516, 0.29207300232794003, 0.22024240429158187]
    contrast_list_T2_185MHz =[1.4999999996301037, 0.34128675880697973, 0.255602390332656, 0.7268644663608335, 0.2805573583375788, 0.2623285262125501, 0.41678508185128416, 0.2544229770942468, 0.469510723907074, 0.2893940749318923, 0.21030988284311472, 0.2340024742439338, 0.22326020676036612, 0.20955899788375318, 0.3743709977250483, 0.2924579476978601, 0.24944733012016282, 0.23998741163318846, 0.7445527009965711, 0.2785309438946973, 0.2710058081951797, 0.2809511274419595, 0.23140910785534163, 0.3103242983387597, 0.19386973884572473, 0.4284162061551819, 0.24667476085127882, 0.24979568508218897, 0.23346538852566254, 0.2517254204791441, 0.24654892016674268, 0.25727239858644857, 0.23384147746331843, 0.2810815530400289, 0.2692580801300352]
    contrast_list_T2 = contrast_list_T2_185MHz + contrast_list_T2_68MHz

    # b
    b_list_T1 = [0.03696981759922204, 0.15389155599135537, 0.02398028622680929, 0.22858485426117583, 0.04733540885424477, -0.05338990088662019, 0.0373000196360804, 0.09249653953939091, 0.09964306305554625, -0.0013928503472253023, 0.012024253043728169, -0.0006482352667783434, -0.09129337469866194, 0.08284606876503961, -0.07343411908336538, 0.1898544337395348, -0.002068799095084259, -0.0003844920999426842, 0.1274417772473542, 0.017532777510053544, 0.12240667082942644, 0.08376606046413897, 0.038482967461192716, 0.11131586074515792, 0.3356231404653557, 0.008836137697552478, 0.32288727696744934, 0.07850168131327961, 0.04396328439949872, 0.01546204960705676, -0.034350065242008884, 0.14900078801978425, 0.07393099090232871, 0.007990647229140292, 0.14752990445017883, 0.051073081443276616, 0.004118123471940301, 0.1326243570845902, 0.1888425408883814, 0.10016116732930695, 0.00831039491453627, 0.08055803198703244, -0.040009610909473456, 0.06006514604516922, 0.12401939661686741, -0.020140540924190906, 0.11768499792335792, -0.11311927577847518, 0.016177522671440615, 0.05998005462339177, -0.07990325182879812, 0.0973113271147902, -0.032867622273546844, -0.06301061479112091, 0.07147108805948674, 0.00971035560482675, -0.04865631004187237, -0.012085334533774798, 0.08877713295956069, 0.18529790351207198, 0.01893890307476978, 0.005525235593350504, 0.04236789410746585, 0.03414397636982159, -0.012951923497842558, 0.06490248025946428, 0.058323021311236005, 0.009733987564629015, 0.1326339277304399, -0.059791723770699017, 0.09146127342585197, 0.10944617682327314, 0.09925141887941882]
    b_list_T2_68MHz = [4.816204705903527, 0.6168113115706303, 0.6806912089156919, 0.7837102558528242, 0.6848742408312919, 0.6812828028524054, 1.0305001862161933, 0.7509916951003324, 1.1255085406138763, 0.9505223369275717, 0.6718816474415381, 0.6811218578965943, 0.7187575264105771, 0.7575692488538706, 0.571223175368567, 0.6722545770071162, 0.6460782562950039, 0.6627568983344697, 1.3002570845564692, 0.6630892878477505, 0.6709280824618071, 0.7746947574773559, 0.6631390068226853, 0.6418783472492819, 0.7548362671462352, 0.6957109979299595, 0.6587818398887894, 0.6561761201770238, 0.674590316145524, 0.7125319425124211, 0.6780046252896814, 0.6761158978152059, 0.6795021962223963, 0.718253256565652, 0.6694519243227017]
    b_list_T2_185MHz = [3.2097885939012967, 0.5318683640601995, 0.9977724132048927, 0.4658247817727291, 0.5117997932837548, 0.5355250133273287, 0.5312984385330984, 0.539617940805706, 0.52504909149491, 0.5595174552327852, 0.5512700611579663, 0.49145204639089113, 0.5203094480757041, 0.5224896868889846, 0.8622103202554361, 0.5170437850026282, 0.5644333345197982, 0.5087679469082755, 0.4991728825252808, 0.4858985289169379, 0.5011151289839816, 0.5244695053368857, 0.452741079259692, 0.48933299854277623, 0.5059121364866588, 0.4972570149886221, 0.9943006485738779, 0.5136522085251156, 0.5281717502388724, 0.5154836833721107, 0.5167480058852033, 0.9416637149841658, 0.5948832527871032, 0.5949870673520313, 0.5167546383431316, 0.5343420264047861, 0.5479554895298352, 0.508433792889998, 0.5396022866244736, 0.5551040051383119]
    b_list_T2 = b_list_T2_185MHz + b_list_T2_68MHz
    # fmt:on
    # print(f"contrast_list_T2 = {contrast_list}")
    # sys.exit()
    fig, ax = plt.subplots(figsize=(6, 5))
    nv_indices_T2 = np.arange(len(contrast_list_T2))
    nv_indices_T1 = np.arange(len(contrast_list_T1))
    ax.bar(
        nv_indices_T2,
        contrast_list_T2,
        color="orange",
        edgecolor="k",
        alpha=0.7,
        label="XY8(T2)",
    )
    ax.bar(
        nv_indices_T1,
        b_list_T1,
        color="teal",
        edgecolor="k",
        alpha=0.7,
        label="S(0,-1)(T1)",
    )
    ax.set_xlabel("NV Index", fontsize=14)
    ax.set_ylabel("Contrast", fontsize=14)
    ax.set_title("Contrast per NV", fontsize=15)
    ax.tick_params(labelsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(
        nv_indices_T2,
        b_list_T2,
        color="orange",
        edgecolor="k",
        alpha=0.7,
        label="XY8(T2)",
    )
    ax.bar(
        nv_indices_T1,
        b_list_T1,
        color="teal",
        edgecolor="k",
        alpha=0.7,
        label="S(0,-1)(T1)",
    )
    ax.set_xlabel("NV Index", fontsize=14)
    ax.set_ylabel("Baseline Value", fontsize=14)
    ax.set_title("Baseline Value per NV", fontsize=15)
    ax.tick_params(labelsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_T2_on_T1():
    # fmt: off
    list_T2 = [37.26895793472622, 1140.8252920245307, 6743.746190548513, 664.5028904539821, 3598.680083737381, 2017.300341798606, 1728.5089467676432, 3285.9478691598215, 7424.210040726823, 362.89633046176385, 3180.7682611950745, 483.28108973828995, 971.0769759711598, 2083.6894182759843, 1377.7004622562004, 4756.182698853826, 4453.92332900852, 726.8021283279614, 2401.2456604559407, 438.1463745212289, 946.00477469283, 243.27590547510712, 651.372099431156, 516.3593364623061, 1894.859991659015, 804.2455211917787, 3650.2650457098835, 2659.6079663694477, 3462.180284292328, 1046.045826175307, 811.0439716745412, 3574.696871620105, 228.99936457655616, 2236.600481452815, 1042.594322183478, 2570.6667222367837, 3265.076821615608, 6258.685108710562, 7998.557574158693, 1890.640992115312]
    list_T1 = [74.04099447283399, 1538.1259802678626, 4654.912804451828, 1910.1062178609386, 4867.4706559585675, 6536.506043273024, 2084.2740439608783, 5017.027633224041, 6925.071662831214, 5543.119332902354, 5793.355127992046, 5123.3569317152405, 2431.2396665044585, 4828.509327874994, 1677.21064733692, 4120.092871182978, 5054.750078964543, 715.9936282829332, 5548.02243974114, 1615.8199354087687, 4392.378914736074, 2298.074819874097, 2285.177837561421, 1071.800148605042, 4080.104356982668, 4230.051936566056, 6270.201095771297, 5337.0082023572795, 5773.634964552831, 3189.582325638845, 3256.0389102520558, 5.3897054640812705, 3058.347922335626, 6063.878498759282, 3469.9199892879387, 5448.505841969119, 4866.498062895886, 4459.097168009739, 4808.90134309902, 4689.345200489028]
    ratio = np.array(list_T2) / np.array(list_T1)
    median = np.median(ratio)
    ratio[31] = np.nan # dud nv
    # diff = [diff[i] for i in range(len(diff)) if  diff[i] > 0]
    # fmt: on
    nv_indices = np.arange(len(ratio))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(
        nv_indices,
        ratio,
        color="blue",
        edgecolor="k",
        alpha=0.7,
        label=f"T₂ / T₁ Ratio (median:{median:.2f})",
    )
    ax.set_xlabel("NV Index", fontsize=14)
    ax.set_ylabel("T2 / T1 Ratio", fontsize=14)
    ax.set_title("T2 / T1 Ratio per NV", fontsize=15)
    ax.tick_params(labelsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.show()


def stretched_exp_fixed_n(tau, a, t2, b, n_fixed):
    return a * (1 - np.exp(-((tau / t2) ** n_fixed))) + b


def plot_xy8_fits_fixed_n(
    nv_list, taus, norm_counts, norm_counts_ste, n_fixed, save_prefix=None
):

    num_nvs = len(nv_list)
    T2_list, fit_params = [], []
    b_list, a_list = [], []
    chi2_list, T2_error_list = []
    for nv_ind in range(num_nvs):
        y = norm_counts[nv_ind]
        yerr = norm_counts_ste[nv_ind]
        a0 = np.clip(np.ptp(y), 0.1, 1.0)
        t2_0 = np.median(taus)
        b0 = np.mean(y[-4:])
        p0 = [a0, t2_0, b0]
        bounds = ([0, 1e-1, -10], [1.5, 1e5, 10])

        try:
            popt, pcov = curve_fit(
                lambda tau, a, t2, b: stretched_exp_fixed_n(tau, a, t2, b, n_fixed),
                taus,
                y,
                p0=p0,
                bounds=bounds,
                sigma=yerr,
                absolute_sigma=True,
                maxfev=20000,
            )
            T2_list.append(popt[1])
            fit_params.append(popt)
            param_errors = np.sqrt(np.diag(pcov))
            a_list.append(popt[0])
            b_list.append(popt[2])
            T2_error_list.append(param_errors[1])
            # Calculate reduced chi-squared
            residuals = stretched_exp_fixed_n(taus, *popt, n_fixed) - y
            red_chi_sq = np.sum((residuals / yerr) ** 2) / (len(taus) - len(popt))
            T2_list.append(popt[1])
            chi2_list.append(red_chi_sq)
        except Exception as e:
            print(f"Fit failed for NV {nv_ind} at n={n_fixed}: {e}")
            T2_list.append(np.nan)
            fit_params.append(None)
            T2_error_list.append(np.nan)
            a_list.append(np.nan)
            b_list.append(np.nan)
            chi2_list.append(np.nan)

    # === Plotting ===
    sns.set(style="whitegrid")
    num_cols = 8
    num_rows = int(np.ceil(num_nvs / num_cols))
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(num_cols * 1.8, num_rows * 3),
        sharex=True,
        sharey=False,
        constrained_layout=True,
        gridspec_kw={"wspace": 0.0, "hspace": 0.0},
    )
    axes = axes.flatten()
    taus_fit = np.logspace(np.log10(min(taus)), np.log10(max(taus)), 200)

    for nv_idx, ax in enumerate(axes):
        if nv_idx >= len(nv_list):
            ax.axis("off")
            continue
        y = norm_counts[nv_idx]
        yerr = norm_counts_ste[nv_idx]
        max_y = np.max(y)
        a = a_list[nv_idx]
        b = b_list[nv_idx]
        popt = fit_params[nv_idx]

        sns.scatterplot(
            x=taus,
            y=max_y - y + b,
            ax=ax,
            color="blue",
            label=f"NV {nv_idx} (T2 = {T2_list[nv_idx]:.2f} µs)",
            s=10,
            alpha=0.7,
        )
        ax.errorbar(
            taus,
            max_y - y + b,
            yerr=yerr,
            fmt="o",
            alpha=0.9,
            ecolor="gray",
            markersize=0.1,
        )
        if popt is not None:
            y_fit = max_y - stretched_exp_fixed_n(taus_fit, *popt, n_fixed=n_fixed) + b
            sns.lineplot(
                x=taus_fit,
                y=y_fit,
                ax=ax,
                lw=1,
            )

        ax.legend(fontsize="xx-small")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.tick_params(axis="y", labelsize=8, direction="in", pad=-10)
        for label in ax.get_yticklabels():
            label.set_horizontalalignment("right")
            label.set_x(0.01)

        ax.set_xscale("log")
        ax.set_xlabel("Time (μs)", fontsize=9, labelpad=1)

    fig.text(
        0.005, 0.5, "NV$^{-}$ Population", va="center", rotation="vertical", fontsize=11
    )
    fig.suptitle(f"XY8 Fits (n = {n_fixed})", fontsize=13)
    fig.tight_layout(pad=0.4, rect=[0.01, 0.01, 0.99, 0.99])
    # if save_prefix:
    #     plt.savefig(f"{save_prefix}_n{n_fixed}.png", dpi=300)
    plt.show()

    return T2_list


def full_xy8_fitting_over_n(
    nv_list, taus, norm_counts, norm_counts_ste, n_values=[1, 2, 3, 4, 5, 6]
):
    median_T2s = []
    all_T2s_by_n = {}

    for n in n_values:
        print(f"\n=== Fitting with n = {n} ===")
        T2_list = plot_xy8_fits_fixed_n(
            nv_list,
            taus,
            norm_counts,
            norm_counts_ste,
            n_fixed=n,
            save_prefix="xy8_fits",
        )
        T2_clean = np.array(T2_list)
        T2_clean = T2_clean[~np.isnan(T2_clean)]
        median_T2 = np.median(T2_clean)
        print(f"Median T2 for n={n}: {median_T2:.2f} µs")
        median_T2s.append(median_T2)
        all_T2s_by_n[n] = T2_clean

    # Final summary plot
    plt.figure(figsize=(6, 5))
    plt.plot(n_values, median_T2s, "o-", lw=2)
    plt.xlabel("Fixed Stretching Exponent n", fontsize=15)
    plt.ylabel("Median T2 (µs)", fontsize=15)
    plt.title("Median T2 vs. Fixed n", fontsize=15)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(n_values)
    plt.tight_layout()
    plt.show()

    return all_T2s_by_n


def fit_and_plot_all_T2_with_chi2(
    nv_list, taus, norm_counts, norm_counts_ste, n_values=[1, 2, 3, 4]
):
    all_T2s = {}
    all_chi2s = {}
    num_nvs = len(nv_list)

    for n_fixed in n_values:
        T2_list = []
        chi2_list = []

        for nv_ind in range(num_nvs):
            y = norm_counts[nv_ind]
            yerr = norm_counts_ste[nv_ind]

            a0 = np.clip(np.ptp(y), 0.1, 1.0)
            t2_0 = np.median(taus)
            b0 = np.mean(y[-4:])
            p0 = [a0, t2_0, b0]
            bounds = ([0, 1e-1, -10], [1.5, 1e5, 10])

            try:
                popt, pcov = curve_fit(
                    lambda tau, a, t2, b: stretched_exp_fixed_n(tau, a, t2, b, n_fixed),
                    taus,
                    y,
                    p0=p0,
                    bounds=bounds,
                    sigma=yerr,
                    absolute_sigma=True,
                    maxfev=20000,
                )
                residuals = stretched_exp_fixed_n(taus, *popt, n_fixed) - y
                red_chi_sq = np.sum((residuals / yerr) ** 2) / (len(taus) - len(popt))
                T2_list.append(popt[1])
                chi2_list.append(red_chi_sq)

            except Exception as e:
                print(f"Fit failed for NV {nv_ind} at n={n_fixed}: {e}")
                T2_list.append(np.nan)
                chi2_list.append(np.nan)

        all_T2s[n_fixed] = T2_list
        all_chi2s[n_fixed] = chi2_list

        # === Plotting for this n ===
        T2_arr = np.array(T2_list)
        chi2_arr = np.array(chi2_list)
        nv_indices = np.arange(num_nvs)
        median_T2 = np.nanmedian(T2_arr)

        fig, ax = plt.subplots(figsize=(6, 5))
        scatter = ax.scatter(
            nv_indices,
            T2_arr,
            c=chi2_arr,
            edgecolors="black",
            s=50,
            label="T2 per NV",
        )
        ax.axhline(
            median_T2,
            color="r",
            linestyle="--",
            label=f"Median T2 ≈ {median_T2:.1f} µs",
        )
        ax.set_xlabel("NV Index", fontsize=15)
        ax.set_ylabel("T2 (µs)", fontsize=15)
        ax.set_title(f"T2 per NV (n = {n_fixed})", fontsize=15)
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.tick_params(labelsize=14)

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Reduced χ²", fontsize=13)
        cbar.ax.tick_params(labelsize=12)
        ax.legend(fontsize=11)
        plt.tight_layout
        plt.show()

    return all_T2s, all_chi2s


def fit_for_fixed_n_range(
    nv_list, taus, norm_counts, norm_counts_ste, n_values=[1, 2, 3, 4]
):
    medians_by_n = []
    all_T2s_by_n = {}

    for n_fixed in n_values:
        T2s = []

        for nv_ind in range(len(nv_list)):
            y = norm_counts[nv_ind]
            yerr = norm_counts_ste[nv_ind]
            a0 = np.clip(np.ptp(y), 0.1, 1.0)
            t2_0 = np.median(taus)
            b0 = np.mean(y[-4:])
            p0 = [a0, t2_0, b0]
            bounds = ([0, 1e-1, -10], [1.5, 1e5, 10])

            try:
                popt, _ = curve_fit(
                    lambda tau, a, t2, b: stretched_exp_fixed_n(tau, a, t2, b, n_fixed),
                    taus,
                    y,
                    p0=p0,
                    bounds=bounds,
                    sigma=yerr,
                    absolute_sigma=True,
                    maxfev=20000,
                )
                T2s.append(popt[1])
            except Exception as e:
                print(f"Fit failed for NV {nv_ind} at n={n_fixed}: {e}")
                T2s.append(np.nan)

        T2s = np.array(T2s)
        T2s = T2s[~np.isnan(T2s)]
        all_T2s_by_n[n_fixed] = T2s
        median_T2 = np.median(T2s)
        medians_by_n.append(median_T2)
        print(f"n={n_fixed}: median T2 = {median_T2:.2f} µs")

    return n_values, medians_by_n, all_T2s_by_n


def plot_medians(n_values, medians_by_n):
    plt.figure(figsize=(6, 4))
    plt.plot(n_values, medians_by_n, "o-", lw=2)
    plt.xlabel("Fixed Stretching Exponent n", fontsize=14)
    plt.ylabel("Median T₂ (µs)", fontsize=14)
    plt.title("Median T₂ vs. Fixed n", fontsize=15)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(n_values)
    plt.tight_layout()
    plt.show()


def T2_diff_xy():
    # fmt: off
    xy8_T2 = [37.26895793472622, 1140.8252920245307, 6743.746190548513, 664.5028904539821, 3598.680083737381, 2017.300341798606, 1728.5089467676432, 3285.9478691598215, 7424.210040726823, 362.89633046176385, 3180.7682611950745, 483.28108973828995, 971.0769759711598, 2083.6894182759843, 1377.7004622562004, 4756.182698853826, 4453.92332900852, 726.8021283279614, 2401.2456604559407, 438.1463745212289, 946.00477469283, 243.27590547510712, 651.372099431156, 516.3593364623061, 1894.859991659015, 804.2455211917787, 3650.2650457098835, 2659.6079663694477, 3462.180284292328, 1046.045826175307, 811.0439716745412, 3574.696871620105, 228.99936457655616, 2236.600481452815, 1042.594322183478, 2570.6667222367837, 3265.076821615608, 6258.685108710562, 7998.557574158693, 1890.640992115312]
    xy4_T2 = [1767.3770024556775, 2232.476715427384, 571.6173966760732, 1626.2272166066136, 5118.59372140522, 3659.8883342280083, 3009.822133001033, 7366.490729706452, 5432.824853647156, 3926.7878934839123, 1228.5950061785759, 1276.2302445691778, 976.2803415896387, 1418.3369475448187, 9999.961139108893, 3534.5835986188767, 363.59673072260387, 1517.49135754386, 384.5186394917788, 473.42469642826774, 318.844262968126, 1838.564839594962, 1130.2025592809055, 2126.6044345252067, 505.3908042589281, 2263.2792768418954, 3099.957455912386, 7656.351910514253, 1572.0253223314749, 370.5797440458685, 4550.031439191324, 147.9077928101106, 3051.1944371996465, 712.1338963694138, 2447.1940535522, 2081.7588565501887, 1931.2466937455129, 1125.6108407164722, 5047.799177132331]
    diff = np.array(xy8_T2) / np.array(xy4_T2)
    # diff = [diff[i] for i in range(len(diff)) if  diff[i] > 0]
    # fmt: on
    nv_indices = np.arange(len(diff))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(
        nv_indices,
        diff,
        color="blue",
        edgecolor="k",
        alpha=0.7,
        label="(T2_xy8/T2_xy4)",
    )
    ax.set_xlabel("NV Index", fontsize=14)
    ax.set_ylabel("T2_xy8/T2_xy42(us)", fontsize=14)
    ax.set_title("T2_xy8/T2_xy4 Value per NV", fontsize=15)
    ax.tick_params(labelsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_xy_all(nv_list, taus, norm_counts, norm_counts_ste, seq_xy):
    sns.set(style="whitegrid", palette="muted")
    num_nvs = len(nv_list)
    colors = sns.color_palette("deep", num_nvs)
    num_cols = 7
    num_rows = int(np.ceil(len(nv_list) / num_cols))

    # Full plot
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(num_cols * 3, num_rows * 3),
        sharex=True,
        sharey=False,
        constrained_layout=True,
        gridspec_kw={"wspace": 0.0, "hspace": 0.0},
    )
    axes = axes.flatten()
    print(f"axes: {axes}, len(axes): {len(axes)}")
    for nv_idx, ax in enumerate(axes):
        if nv_idx >= len(nv_list):
            ax.axis("off")
            continue

        nv_tau = taus  # Convert to µs
        nv_counts = norm_counts[nv_idx]
        # Plot data and fit on full plot
        sns.lineplot(
            x=nv_tau,
            y=nv_counts,
            ax=ax,
            color=colors[nv_idx % len(colors)],
            lw=0,
            marker="o",
            markersize=4,
            label=f"NV {nv_idx}",
        )
        ax.errorbar(
            nv_tau,
            norm_counts[nv_idx],
            yerr=abs(norm_counts_ste[nv_idx]),
            fmt="none",
            ecolor=colors[nv_idx % len(colors)],
            # elinewidth=0.5,
            alpha=0.9,
        )
        ax.legend(fontsize="xx-small")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax.tick_params(labelleft=False)

        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.tick_params(axis="y", labelsize=8, direction="in", pad=-10)
        for label in ax.get_yticklabels():
            label.set_horizontalalignment("right")
            label.set_x(0.01)
        # ax.set_xlabel("Time (μs)", fontsize=9, labelpad=1)

    for col in range(num_cols):
        bottom_row_idx = num_rows * num_cols - num_cols + col
        if bottom_row_idx < len(axes):
            ax = axes[bottom_row_idx]
            tick_positions = np.linspace(min(taus), max(taus), 5)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(
                [f"{tick:.2f}" for tick in tick_positions],
                rotation=45,
                fontsize=9,
            )
            ax.set_xlabel("Time (µs)")
        else:
            ax.set_xticklabels([])
    fig.text(
        0.000,
        0.5,
        "NV$^{-}$ Population",
        va="center",
        rotation="vertical",
        fontsize=12,
    )
    fig.suptitle(f"{seq_xy}", fontsize=16, y=1.0)
    # fig.set_constrained_layout(False)
    # fig.subplots_adjust(left=0.01, right=0.98, top=0.99, bottom=0.08, hspace=0.01, wspace=0.01)
    # fig.subplots_adjust(
    #     left=0.01, right=0.99, top=0.99, bottom=0.01, hspace=0.000, wspace=0.000
    # )
    fig.tight_layout(pad=0.2, rect=[0.02, 0.01, 0.99, 0.99])
    kpl.show()


if __name__ == "__main__":
    kpl.init_kplotlib()
    # 185MHz orientation
    # file_ids = [
    #     1818816006504,
    #     1818985247947,
    #     1819154094977,
    #     1819318427055,
    #     1819466247867,
    #     1819611450115,
    # ]
    # 68MHz orientation xy8
    # file_ids = [
    #     1820856154901,
    #     1820741644537,
    #     1820575030849,
    #     1820447821240,
    #     1820301119952,
    #     1820151354472,
    # ]
    ## 68MHz orientation/ manuual reference xy8
    # file_ids = [
    #     1822348912678,
    #     1822231688646,
    #     1822119685591,
    #     1821983732646,
    #     1821816914155,
    #     1821689973348,
    # ]

    ## 68MHz orientation XY4
    # file_ids = [1823061683325, 1823448847102]
    # 68MHz orientation XY16
    # file_ids = [1823813420689, 1824210908904]
    ## 68MHz orientation XY8 buffer/wait updated
    # file_ids = [1824501762414, 1824732255862]
    ## 68MHz orientation XY8 dense
    # file_ids = [1825683241321, 1825803148127]

    #### xy8 after making the both orientation degenerate
    # file_ids = [1838226467730, 1838534721391]  # 2us to 20us ) uniform spacing
    # file_ids = [1839161749987, 1839342727027, 1841082599417]  # (2us to 20us )/ 2*8
    # file_ids = [
    #     1841472330271,
    #     1841783840740,
    #     1842055914148,
    # ]  # (2us to 20us )/ 2*8 uniform spacing
    # file_stems = ["2025_04_30-22_07_41-rubin-nv0_2025_02_26"]  # xy8
    ### New data sets
    # file_stems = [
    #     # "2025_05_01-05_35_44-rubin-nv0_2025_02_26",
    #     # "2025_05_01-11_56_26-rubin-nv0_2025_02_26",
    #     "2025_05_02-10_43_10-rubin-nv0_2025_02_26",
    #     "2025_05_02-17_30_41-rubin-nv0_2025_02_26",
    #     "2025_05_02-23_57_21-rubin-nv0_2025_02_26",
    # ]  # xy4

    # file_stems = ["2025_05_06-20_59_05-rubin-nv0_2025_02_26"]  # xy4
    # file_stems = [
    #     "2025_05_09-11_56_26-rubin-nv0_2025_02_26",
    #     "2025_05_08-20_36_14-rubin-nv0_2025_02_26",
    #     "2025_05_08-04_29_55-rubin-nv0_2025_02_26",
    #     "2025_05_07-12_41_35-rubin-nv0_2025_02_26",
    # ]  # xy4 to obseve multiple peaks

    # file_stems = [
    #     "2025_05_12-00_18_03-rubin-nv0_2025_02_26",
    #     "2025_05_11-19_15_33-rubin-nv0_2025_02_26",
    #     "2025_05_12-17_17_40-rubin-nv0_2025_02_26",
    # ]  # xy4 log scale
    # file_stems = [
    #     "2025_05_14-01_03_29-rubin-nv0_2025_02_26",
    #     "2025_05_13-19_56_05-rubin-nv0_2025_02_26",
    #     "2025_05_13-14_50_04-rubin-nv0_2025_02_26",
    # ]  # xy8 log scale

    file_stems = [
        "2025_05_15-00_30_58-rubin-nv0_2025_02_26",
        "2025_05_14-19_25_27-rubin-nv0_2025_02_26",
        "2025_05_14-14_20_16-rubin-nv0_2025_02_26",
    ]  # xy16 log scale

    # cpdd
    # file_stems = [
    #     "2025_05_04-02_06_22-rubin-nv0_2025_02_26",
    #     "2025_05_04-07_37_32-rubin-nv0_2025_02_26",
    #     "2025_05_04-13_07_43-rubin-nv0_2025_02_26",
    #     "2025_05_04-19_00_22-rubin-nv0_2025_02_26",
    # ]  # xy8 cpdd
    ### Internal Test Plots

    file_stems = [
    "2026_01_09-06_06_55-johnson-nv0_2025_10_21",
    "2026_01_09-10_35_51-johnson-nv0_2025_10_21",
]  # xy14
    # plot_T2_on_T1()
    # contrast_plot()
    # T2_ratio_xy()
    # plt.show(block=True)
    # sys.exit()
    all_file_ids_str = widefield.combined_filenames_numbered(file_stems)
    print(f"File name: {all_file_ids_str}")
    raw_data = widefield.process_multiple_files(file_stems, load_npz=True)
    nv_list = raw_data["nv_list"]
    taus = np.array(raw_data["taus"]) / 1e3  # τ values (in us)
    # Get sequence type
    seq_xy = raw_data.get("xy_seq", "xy8").lower()
    base, N = widefield.parse_xy_sequence(seq_xy)
    print(f"Input Sequence Type: {seq_xy}")
    print(f"Total pi pulses: {N}")
    taus = 2 * N * taus
    # sys.exit()
    counts = np.array(raw_data["counts"])  # shape: (2, num_nvs, num_steps)
    sig_counts = counts[0]
    ref_counts = counts[1]
    # Normalize counts
    norm_counts, norm_counts_ste = widefield.process_counts(
        nv_list, sig_counts, ref_counts, threshold=True
    )
    # norm_counts, norm_counts_ste = widefield.process_counts(
    #     nv_list, ref_counts, threshold=True
    # )
    process_and_plot_xy8(nv_list, taus, norm_counts, norm_counts_ste, seq_xy)
    # n_values, medians_by_n, all_T2s_by_n = fit_for_fixed_n_range(
    #     nv_list, taus, norm_counts, norm_counts_ste, n_values=[1, 2, 3, 4, 5, 6]
    # )
    # plot_medians(n_values, medians_by_n)
    # full_xy8_fitting_over_n(
    #     nv_list, taus, norm_counts, norm_counts_ste, n_values=[1, 2, 3, 4, 5, 6]
    # )
    # fit_and_plot_all_T2_with_chi2(
    #     nv_list, taus, norm_counts, norm_counts_ste, n_values=[1, 2, 3, 4, 5, 6]
    # )
    # plot_xy_all(nv_list, taus, norm_counts, norm_counts_ste, seq_xy)
    # fit_params, T2_list, n_list, chi2_list, param_errors_list = process_and_fit_xy8(
    #     nv_list, taus, norm_counts, norm_counts_ste
    # )
    # plot_xy8(nv_list, taus, norm_counts, norm_counts_ste, T2_list, n_list, fit_params)
    plt.show(block=True)
