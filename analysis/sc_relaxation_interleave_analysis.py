# -*- coding: utf-8 -*-
"""
Created on Fall, 2024

@author: saroj chand
"""
import sys
from utils import data_manager as dm
from utils import kplotlib as kpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
from scipy.optimize import curve_fit, least_squares
from scipy.ndimage import gaussian_filter1d
from utils import widefield as widefield
from matplotlib.ticker import FormatStrFormatter


def exp_decay(tau, norm, rate, offset=0):
    """Exponential decay function."""
    return norm * np.exp(-rate * tau) + offset


def double_exp_decay(tau, norm1, rate1, norm2, rate2, offset=0):
    """Double exponential decay function for more complex fitting scenarios."""
    return norm1 * np.exp(-rate1 * tau) + norm2 * np.exp(-rate2 * tau) + offset


def T1_fit(tau, y, yerr=None, model="single"):
    """Performs fitting using least squares with different loss functions."""
    if not np.isfinite(y).all() or not np.isfinite(tau).all():
        raise ValueError("Input data contains non-finite values.")

    epsilon = 1e-10  # Small value to prevent division by zero

    if model == "single":

        def residuals(params):
            return (exp_decay(tau, *params) - y) / (
                yerr + epsilon if yerr is not None else 1
            )

        initial_guess = [y[0], 1 / (tau[-1] - tau[0]), 0]
    elif model == "double":

        def residuals(params):
            return (double_exp_decay(tau, *params) - y) / (
                yerr + epsilon if yerr is not None else 1
            )

        initial_guess = [
            y[0] / 2,
            1 / (tau[-1] - tau[0]),
            y[0] / 2,
            1 / (tau[-1] - tau[0]),
            0,
        ]
    else:
        raise ValueError("Invalid model specified.")

    # print("Initial guess:", initial_guess)  # Debugging print statement

    result = least_squares(residuals, initial_guess, loss="soft_l1")
    # Estimate parameter standard errors
    try:
        # Degrees of freedom = number of points - number of parameters
        dof = max(0, len(tau) - len(result.x))
        residual_var = np.sum(result.fun**2) / dof
        J = result.jac
        cov = np.linalg.inv(J.T @ J) * residual_var
        param_errors = np.sqrt(np.diag(cov))
    except:
        param_errors = np.full_like(result.x, np.nan)

    return result.x, result.success, param_errors


# def process_and_fit_data(data, use_double_fit=False, selected_indices=None):
#     """Processes and fits NV relaxation data with robust fitting."""
#     nv_list = data["nv_list"]
#     taus = np.array(data["taus"]) / 1e6  # Convert ns to ms
#     counts = np.array(data["counts"])
#     sig_counts, ref_counts = counts[0], counts[1]

#     # Process counts using widefield's process_counts function
#     sig_avg_counts, sig_avg_counts_ste = widefield.process_counts(
#         nv_list, sig_counts, threshold=False
#     )
#     ref_avg_counts, ref_avg_counts_ste = widefield.process_counts(
#         nv_list, ref_counts, threshold=False
#     )
#     # Compute the difference between the states
#     norm_counts = sig_avg_counts - ref_avg_counts
#     norm_counts_ste = np.sqrt(sig_avg_counts_ste**2 + ref_avg_counts_ste**2)

#     num_nvs = len(nv_list)
#     if selected_indices is not None:
#         nv_list = [nv_list[ind] for ind in selected_indices]
#         norm_counts = [norm_counts[ind] for ind in selected_indices]
#         norm_counts_ste = [norm_counts_ste[ind] for ind in selected_indices]

#     fit_params, fit_functions, residuals, param_errors, contrasts = [], [], [], [], []

#     for nv_idx in range(len(nv_list)):
#         nv_counts = gaussian_filter1d(
#             norm_counts[nv_idx], sigma=1
#         )  # Smoothing for stability
#         nv_counts_ste = norm_counts_ste[nv_idx]

#         # Try single exponential first
#         params, success, errors = T1_fit(
#             taus, nv_counts, yerr=nv_counts_ste, model="single"
#         )
#         if not success and use_double_fit:
#             params, success = T1_fit(
#                 taus, nv_counts, yerr=nv_counts_ste, model="double"
#             )

#         fit_curve = (
#             exp_decay(taus, *params)
#             if len(params) == 3
#             else double_exp_decay(taus, *params)
#         )
#         fit_params.append(params)
#         param_errors.append(errors)
#         fit_functions.append(
#             lambda t, p=params: (
#                 exp_decay(t, *p[:3]) if len(p) == 3 else double_exp_decay(t, *p)
#             )
#         )
#         residuals.append(nv_counts - fit_curve)

#     # print(f"rate_3Omega = {list(fit_params[:, 1])}")
#     # print(f"rate_3Omega_error = {list(param_errors[:, 1])}")
#     fit_params = np.array(fit_params)
#     param_errors = np.array(param_errors)

#     print(f"rate_3Omega = {list(fit_params[:, 1])}")
#     print(f"rate_3Omega_error = {list(param_errors[:, 1])}")
#     return (
#         fit_params,
#         fit_functions,
#         residuals,
#         taus,
#         norm_counts,
#         norm_counts_ste,
#         nv_list,
#         param_errors,
#     )


def process_and_fit_data(
    data,
    use_double_fit=False,
    selected_indices=None,
    *,
    error_method="cov",  # "cov" or "bootstrap"
    n_boot=300,  # used if error_method == "bootstrap"
    smooth_sigma=1.0,  # Gaussian smoothing sigma for stability
):
    """Processes and fits NV relaxation data with robust fitting."""
    nv_list = data["nv_list"]
    taus = np.array(data["taus"]) / 1e6  # Convert ns to ms
    counts = np.array(data["counts"])
    sig_counts, ref_counts = counts[0], counts[1]

    # Process counts using widefield's process_counts function
    sig_avg_counts, sig_avg_counts_ste = widefield.process_counts(
        nv_list, sig_counts, threshold=False
    )
    ref_avg_counts, ref_avg_counts_ste = widefield.process_counts(
        nv_list, ref_counts, threshold=False
    )

    # Compute the difference between the states
    norm_counts = sig_avg_counts - ref_avg_counts
    norm_counts_ste = np.sqrt(sig_avg_counts_ste**2 + ref_avg_counts_ste**2)

    if selected_indices is not None:
        nv_list = [nv_list[ind] for ind in selected_indices]
        norm_counts = [norm_counts[ind] for ind in selected_indices]
        norm_counts_ste = [norm_counts_ste[ind] for ind in selected_indices]

    fit_params, fit_functions, residuals, param_errors = [], [], [], []

    for nv_idx in range(len(nv_list)):
        # Smoothing for stability
        nv_counts = (
            gaussian_filter1d(norm_counts[nv_idx], sigma=smooth_sigma)
            if smooth_sigma and smooth_sigma > 0
            else norm_counts[nv_idx]
        )
        nv_counts_ste = norm_counts_ste[nv_idx]

        # Primary fit (single exp first; optionally double)
        params, success, errors_cov = T1_fit(
            taus, nv_counts, yerr=nv_counts_ste, model="single"
        )
        used_model = "single"
        if (not success) and use_double_fit:
            params, success2, errors_cov2 = T1_fit(
                taus, nv_counts, yerr=nv_counts_ste, model="double"
            )
            if success2:
                params, errors_cov, used_model = params, errors_cov2, "double"

        # Build fit curve & callable
        if used_model == "single" or len(params) == 3:
            fit_curve = exp_decay(taus, *params[:3])
            fit_fn = lambda t, p=params: exp_decay(t, *p[:3])
        else:
            fit_curve = double_exp_decay(taus, *params)
            fit_fn = lambda t, p=params: double_exp_decay(t, *p)

        # Choose error method
        if error_method == "cov":
            # Use covariance-based errors from T1_fit
            param_err = errors_cov
        elif error_method == "bootstrap":
            # Residual bootstrap around fitted curve
            resid = nv_counts - fit_curve
            boot_params = []
            for _ in range(n_boot):
                resampled = np.random.choice(resid, size=resid.size, replace=True)
                y_boot = fit_curve + resampled
                p_b, ok_b, _ = T1_fit(
                    taus, y_boot, yerr=nv_counts_ste, model=used_model
                )
                if ok_b and len(p_b) == len(params):
                    boot_params.append(p_b)
            if len(boot_params) >= 2:
                bp = np.array(boot_params, dtype=float)
                param_err = np.std(bp, axis=0, ddof=1)
            else:
                # Fallback to covariance if bootstrap failed to collect enough
                param_err = errors_cov
        else:
            raise ValueError("error_method must be 'cov' or 'bootstrap'")

        # Collect
        fit_params.append(params)
        param_errors.append(param_err)
        fit_functions.append(fit_fn)
        residuals.append(nv_counts - fit_curve)

    fit_params = np.array(fit_params, dtype=float)
    param_errors = np.array(param_errors, dtype=float)

    # For your logging; your "rate_3Omega" name suggested older scaling.
    # Here we just report the fitted rate (which you confirmed is Ω).
    print(f"omega_rate = {list(fit_params[:, 1])}")
    print(f"omega_rate_error = {list(param_errors[:, 1])}")

    return (
        fit_params,
        fit_functions,
        residuals,
        taus,
        norm_counts,
        norm_counts_ste,
        nv_list,
        param_errors,
    )


def plot_fitted_data(
    nv_list,
    taus,
    norm_counts,
    norm_counts_ste,
    fit_functions,
    fit_params,
    fit_errors,
    num_cols=8,
    selected_indices=None,
):
    """Plot for raw data with fitted curves using Seaborn style, including NV index labels."""
    fit_params = np.array(fit_params)
    param_errors = np.array(fit_errors)
    rates = fit_params[:, 1]
    rate_errors = param_errors[:, 1]
    T1 = 1 / rates
    T1_err = rate_errors / (rates**2)
    T1, T1_err = list(T1), list(T1_err)

    sns.set(style="whitegrid")
    num_nvs = len(nv_list)
    num_rows = int(np.ceil(num_nvs / num_cols))
    # Full plot
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(num_cols * 1.5, num_rows * 3),
        sharex=True,
        sharey=False,
        # constrained_layout=True,
        gridspec_kw={"wspace": 0.0, "hspace": 0.0},
    )
    axes = axes.flatten()
    # axes = axes[::-1]
    for nv_idx, ax in enumerate(axes):
        if nv_idx >= len(nv_list):
            ax.axis("off")
            continue
        if selected_indices is not None:
            nv_idx_label = selected_indices[nv_idx]
        else:
            nv_idx_label = nv_idx
        sns.scatterplot(
            x=taus,
            y=norm_counts[nv_idx],
            ax=ax,
            color="blue",
            label=f"NV {nv_idx_label}(T1 = {T1[nv_idx]:.2f} ± {T1_err[nv_idx]:.2f} ms)",
            s=10,
            alpha=0.7,
        )
        # Plot error bars separately for clarity
        ax.errorbar(
            taus,
            norm_counts[nv_idx],
            yerr=norm_counts_ste[nv_idx],
            fmt="o",
            alpha=0.9,
            ecolor="gray",
            markersize=0.1,
        )

        taus_fit = np.logspace(np.log10(taus[0]), np.log10(taus[-1]), 200)
        # Plot fitted curve if available
        if fit_functions[nv_idx]:
            fit_curve = fit_functions[nv_idx](taus_fit)
            sns.lineplot(
                x=taus_fit,
                y=fit_curve,
                ax=ax,
                # color="blue",
                # label='Fit',
                lw=1,
            )
        ax.legend(fontsize="xx-small")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        axes_grid = np.array(axes).reshape((num_rows, num_cols))

        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.tick_params(axis="y", labelsize=8, direction="in", pad=-10)
        for label in ax.get_yticklabels():
            label.set_horizontalalignment("right")
            label.set_x(0.02)  # Fine-tune this as needed
            label.set_zorder(100)
        # ax.tick_params(labelleft=False)
        # Loop over each column
        for col in range(num_cols):
            # Go from bottom row upwards
            for row in reversed(range(num_rows)):
                if row * num_cols + col < len(axes):  # Check if subplot exists
                    ax = axes_grid[row, col]

                    # Apply ticks
                    tick_positions = np.logspace(
                        np.log10(taus[0]), np.log10(taus[-1]), 6
                    )
                    ax.set_xticks(tick_positions)
                    ax.set_xticklabels(
                        [f"{tick:.2f}" for tick in tick_positions],
                        rotation=45,
                        fontsize=9,
                    )
                    ax.set_xscale("log")
                    ax.set_xlabel("Time (ms)")
                    break  # Done for this column

    fig.text(
        0.005,
        0.5,
        "NV$^{-}$ Population",
        va="center",
        rotation="vertical",
        fontsize=12,
    )
    # fig.suptitle(f"T1 Relaxation", fontsize=16)
    fig.tight_layout(pad=0.4, rect=[0.01, 0.01, 0.99, 0.99])
    plt.show(block=True)


def plot_fitted_data_separately(
    nv_list,
    taus,
    norm_counts,
    norm_counts_ste,
    fit_functions,
    fit_params,
    param_errors,
):
    """Plot separate figures for each NV with fitted curves using Seaborn style."""
    # sns.set(style="whitegrid")
    fit_params = np.array(fit_params)
    param_errors = np.array(param_errors)
    rates = fit_params[:, 1]
    rate_errors = param_errors[:, 1]
    T1 = 1 / rates
    T1_err = rate_errors / (rates**2)
    T1, T1_err = list(T1), list(T1_err)
    for nv_idx in range(len(nv_list)):
        fig, ax = plt.subplots(figsize=(5, 5))
        # Plot data with Seaborn's lineplot for a cleaner look
        sns.scatterplot(
            x=taus,
            y=norm_counts[nv_idx],
            ax=ax,
            color="blue",
            label=f"NV {nv_idx}",
            s=40,  # Size of markers
            alpha=0.7,
        )

        # Plot error bars separately for clarity
        ax.errorbar(
            taus,
            norm_counts[nv_idx],
            yerr=norm_counts_ste[nv_idx],
            fmt="o",
            alpha=0.5,
            ecolor="gray",
        )
        taus_fit = np.logspace(np.log10(taus[0]), np.log10(taus[-1]), 200)
        # Plot fitted curve if available
        if fit_functions[nv_idx]:
            fit_curve = fit_functions[nv_idx](taus_fit)
            sns.lineplot(
                x=taus_fit,
                y=fit_curve,
                ax=ax,
                # color="red",
                # label='Fit',
                lw=1.5,
            )

        # Add titles and labels
        ax.tick_params(axis="both", labelsize=15)

        ax.set_title(
            f"NV {nv_idx} (T1 = {T1[nv_idx]:.2f} ± {T1_err[nv_idx]:.2f} ms)",
            fontsize=15,
        )

        ax.set_xlabel("Relaxation Time (ms)", fontsize=15)
        ax.set_ylabel("Normalized Counts", fontsize=15)
        ax.set_xscale("log")
        # ax.set_yscale("log")
        ax.legend(fontsize="small")
        ax.grid(True, linestyle="--", linewidth=0.5)

        # Save each figure with a unique name
        now = datetime.now()
        date_time_str = now.strftime("%Y%m%d_%H%M%S")
        file_name = f"nv_{nv_idx}_fitted_plot_{date_time_str}.png"  # Example filename
        # fig.savefig(file_name)
        # plt.close(fig)  # Close the figure after saving to avoid display issues
        plt.show(block=True)


def plot_T1_with_errorbars(fit_params, param_errors, nv_list=None):
    fit_params = np.array(fit_params)
    param_errors = np.array(param_errors)
    nv_indices = np.arange(1, len(fit_params) + 1)
    nv_ticks = np.linspace(1, len(fit_params) + 1, 6)

    if fit_params.shape[1] == 3:
        rates = fit_params[:, 1]
        rate_errors = param_errors[:, 1]

        T1 = 1 / rates
        T1_err = rate_errors / (rates**2)

        plt.errorbar(
            nv_indices,
            T1,
            yerr=T1_err,
            fmt="o",
            capsize=2,
            label="T1 (Median: {:.2f} ms)".format(np.median(T1)),
            # color="blue",
            ecolor="gray",
        )

        # plt.xticks(nv_indices, nv_indices, rotation=45, ha="right")
        plt.xticks(nv_ticks, fontsize=15)
        plt.yticks(fontsize=15)
        # plt.ylim(0, 6)
        plt.xlabel("NV Index", fontsize=15)
        plt.ylabel("T1 (ms)", fontsize=15)
        plt.title("T1 Relaxation Times", fontsize=15)
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.legend()
        plt.show()
    else:
        print("Double exponential model not yet supported in this plot.")


# plot of rates gamma and omega
def plots_rates_omega_gamma():
    # fmt:off
    selected_indices = [1, 2, 3, 4, 5, 6, 10, 12, 13, 15, 16, 17, 19, 20, 21, 23, 29, 34, 36, 39, 40, 41, 42, 43, 50, 51, 52, 54, 56, 59, 61, 63, 65, 74, 7, 8, 9, 11, 14, 18, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 35, 38, 44, 45, 46, 47, 48, 49, 53, 55, 57, 58, 60, 62, 64, 66, 67, 68, 69, 70, 71, 72, 73]
    # orientation both
    rate_3Omega = [1.1976936889180865, 0.17635336350178202, 0.23207348681798132, 0.2379128860377774, 0.18377801719493864, 0.5750862179035522, 0.16161420979784533, 0.175357065112601, 0.35126974595121474, 0.2531256764795871, 0.22963997947989892, 0.2085497884267198, 0.1721189879220002, 0.3066930565073538, 1.5713073513830704, 0.2982985035299495, 0.22647517031149733, 0.2533841224924307, 0.18561857131220302, 0.17979467738319446, 0.24096829438433148, 0.1644795059342368, 0.3617068484791432, 0.21124976706108137, 0.2630984837400458, 0.29026653518687145, 0.19104752465027888, 0.42487877034203303, 0.17642317931885196, 0.42384402088692574, 0.1298782653877778, 0.21398279086855007, 0.23460825331997678, 0.274614373782283, 0.39659161274916177, 0.18114344213880662, 0.47734949645507657, 0.1969312452486425, 0.26306780262999946, 0.45419310004592023, 0.40496629180232663, 0.17440746598886298, 0.2202568949559647, 0.22605876232157365, 0.18346840995872907, 0.3823276903571677, 0.22704498547508387, 0.21584324810935793, 0.1704386109470164, 0.25330245144528163, 0.5936691580365423, 0.2645986856705085, 0.3315547082791894, 0.19096084427489537, 1.4992957231969237, 0.40998672407328735, 0.967889500351763, 0.2959707265775542, 0.2865808940643565, 0.4562341521942319, 0.17387044298258242, 0.16570351302136305, 0.271433386420418, 0.273040472869139, 0.12905843279530146, 0.250632942657402, 0.2018139016675621, 0.22235038245319685, 0.20869624962483996, 0.19313588778197704, 0.24069131433702193, 0.19963709512912292, 0.260765567624031]
    rate_3Omega_error = [0.09149802325571099, 0.006383904881166195, 0.02390240170749795, 0.025970539287805825, 0.02447279492420767, 0.07294193823720474, 0.01646084347913541, 0.03616632706471028, 0.03959039127285494, 0.017629956235876976, 0.017104976540962626, 0.01923503786157632, 0.011464338448519289, 0.046713489611564636, 0.22902712865861202, 0.026175097857274253, 0.02649487898534962, 0.06538722996859575, 0.02005145550374968, 0.013145870126826173, 0.01847416627488859, 0.01127012229233595, 0.07257081407479277, 0.03199317510374231, 0.01683083042335524, 0.03151361615836153, 0.01413384652993107, 0.021388637956370055, 0.01870569178983584, 0.07972574290406255, 0.011619044451812437, 0.0256728849543209, 0.027388798165514084, 0.018973959306608133, 0.04852698582635914, 0.033105436240177, 0.04583310464273806, 0.014607405905134934, 0.026480456723623085, 0.044406816280894476, 0.013401343969808995, 0.016933342927761233, 0.02092458426393885, 0.03370248934222166, 0.015787161664728943, 0.04898119926852212, 0.04232357762651195, 0.017227852577017134, 0.02048679208287819, 0.033302156673985485, 0.07116217123038969, 0.032644300358302784, 0.02185936713611569, 0.013562306007071825, 0.1956355876964972, 0.0735310292702796, 0.0633389291050173, 0.020624251514081572, 0.02012822524467345, 0.12233344641685023, 0.0113803014161342, 0.03224702849157224, 0.03694970889292201, 0.02624575218781167, 0.020122339454199698, 0.041182181162721314, 0.02176175841967699, 0.04898565252103019, 0.021698807966266605, 0.03657669695783045, 0.016333005261542886, 0.018510934314347735, 0.02212144370927943]
    # orientation_185MHz
    rate_180MHz = [3.9934763584260375, 0.35966731922061, 0.33207053639324463, 0.32604983383541275, 0.2897296106662776, 7.34840585431916, 0.3323119591071501, 0.6387132917777691, 0.2890933972779319, 0.4651780391498491, 0.36771094862341125, 0.2991913719455162, 0.36423211961265345, 1.4994476376282748, 1.1611413033535722, 0.37551389170644955, 0.6221808043147038, 0.22895204795842822, 0.2650488870024354, 0.36368139195808014, 0.47025496681684387, 0.3572888851078669, 0.82822527570202, 0.37494044610190524, 0.2917092949627433, 0.5254765343456955, 0.366113184205232, 0.7074924828880482, 0.23173451877488782, 2.8865425075854882, 0.2754841053929161, 0.2879539687770452, 0.3039791901417385, 0.35994633250992214]
    rate_180MHz_error  = [0.555434387231143, 0.05736811258897919, 0.037078731098652035, 0.03454341058137145, 0.03761814902076132, 1.7237276839569107, 0.040687593828151035, 0.20558755862977257, 0.05133986367170311, 0.04553923153770551, 0.025162210643080746, 0.0419226560220524, 0.05942011533787585, 0.2571156681223821, 0.1394211590700833, 0.02839118748991241, 0.04937695979208159, 0.08498667243398579, 0.04197123578853126, 0.056581048534034874, 0.07209297478389706, 0.03659425707438303, 0.042215853272456826, 0.029345892972961564, 0.03246628556601034, 0.031510760319036143, 0.017472451443373845, 0.06825576354510986, 0.02789237641040155, 0.3301138732676395, 0.01811855069002202, 0.03132604485826009, 0.027971616066746766, 0.05739818888107261]
    #rientation_68MHz
    rate_68MHz = [1.2355668481662014, 0.31820001829292105, 1.267117064267596, 0.37732536320859483, 0.5432519796164539, 0.7031028476543627, 0.45896311789369865, 0.3260576038616472, 0.4242335779894588, 0.348622095256152, 2.340500121279145, 0.6534111776612976, 2.1152392529257313, 0.8731240198927788, 0.33040974002334983, 0.28800293387772014, 0.9916799229167227, 0.25184749199773543, 2.445473468259843, 0.8004561580968046, 3.0059907394720526, 0.9331009047497419, 0.7825063820321756, 0.600861735638757, 3.5560148484363205, 0.2053517823632356, 0.36900787392217255, 0.4252402979571251, 0.9130284540975632, 3.2442174685450147, 0.14006130383939672, 6.360383049026441, 0.26899633314207877, 1.792299864149572, 0.403884109491424, 0.4046718871251261, 0.37823670183264285, 0.5980846168851024, 0.5548361880569093]
    rate_68MHz_error  = [0.09522774282138666, 0.09529288981172788, 0.15270678523225342, 0.026511954831080358, 0.017768546608314388, 0.06391182340838614, 0.05435798031105914, 0.02619673758880805, 0.03533137701449433, 0.053856058684344135, 0.1677613397426233, 0.04719259902612614, 0.12446836440524768, 0.09832054319117368, 0.025335356667678912, 0.03287655802765803, 0.10930664686541096, 0.04398512146462449, 0.1607509323702159, 0.04727445267873274, 0.36553754318733833, 0.12426476292725848, 0.08699474143863824, 0.04468854208830564, 0.3009525188944822, 0.054928893676152085, 0.02558669189527077, 0.028169722094463275, 0.1032504985899434, 0.2804536254667047, 0.046437665251114944, 0.5611583727526772, 0.04142000977004301, 0.16560002800329338, 0.023471389611513137, 0.03873443407917636, 0.032877730134903416, 0.05360765403399946, 0.045863757162936516]
    # ll rates
    rate_omega_2gamma = rate_180MHz + rate_68MHz
    rate_omega_2gamma_error = rate_180MHz_error  + rate_68MHz_error
    # fmt:on
    omega = [r / 3 for r in rate_3Omega]
    omega_error = [err / 3 for err in rate_3Omega_error]

    # Example: assuming you already have these
    R = np.array(rate_omega_2gamma)
    sigma_R = np.array(rate_omega_2gamma_error)

    omega = np.array(omega)  # from earlier (rate_3Omega / 3)
    sigma_omega = np.array(omega_error)  # from earlier (rate_3Omega_error / 3)
    # Compute gamma
    gamma = (R - omega) / 2

    # Error propagation
    gamma_error = 0.5 * np.sqrt(sigma_R**2 + sigma_omega**2)

    # Convert to arrays in case they're lists

    omega = np.array(omega)
    gamma = np.array(gamma)
    sigma_omega = np.array(omega_error)
    sigma_gamma = np.array(gamma_error)
    # print(f"Omega min/max: {min(omega):.3f}, {max(omega):.3f}")
    # Compute medians
    median_omega = np.median(omega)
    median_gamma = np.median(gamma)
    # Format with 3 decimal places
    formatted_omega = np.round(omega, 3)
    formatted_gamma = np.round(gamma, 3)
    print(f"omega = {list(formatted_omega)}")
    print(f"gamma = {list(formatted_gamma)}")
    plt.figure(figsize=(6.5, 5))
    # Use error bars in both x and y
    plt.errorbar(
        omega,
        gamma,
        xerr=sigma_omega,
        yerr=sigma_gamma,
        fmt="o",
        capsize=4,
        ecolor="gray",
        # alpha=0.5,
        elinewidth=1,
        label=r"$\gamma$ vs $\Omega$",
    )
    # Text box with median values
    textstr = "\n".join(
        (
            r"$\mathrm{Median\ }\Omega=%.3f\ \mathrm{kHz}$" % median_omega,
            r"$\mathrm{Median\ }\gamma=%.3f\ \mathrm{kHz}$" % median_gamma,
        )
    )

    # Placement of box: upper right inside axes
    plt.gca().text(
        0.97,
        0.97,
        textstr,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    plt.xlabel(r"$\Omega$ (KHz)", fontsize=15)
    plt.ylabel(r"$\gamma$ (KHz)", fontsize=15)
    plt.title(r"$\gamma$ vs $\Omega$", fontsize=15)
    plt.legend(fontsize=11)
    # plt.xlim(0, max(omega) + 0.1)
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.grid(True)
    # plt.tight_layout()
    plt.legend()
    plt.show()

    # Split data
    omega_185 = omega[:34]
    gamma_185 = gamma[:34]
    sigma_omega_185 = sigma_omega[:34]
    sigma_gamma_185 = sigma_gamma[:34]

    omega_68 = omega[34:]
    gamma_68 = gamma[34:]
    sigma_omega_68 = sigma_omega[34:]
    sigma_gamma_68 = sigma_gamma[34:]

    # Calculate medians
    median_omega_185 = np.median(omega_185)
    median_gamma_185 = np.median(gamma_185)

    median_omega_68 = np.median(omega_68)
    median_gamma_68 = np.median(gamma_68)

    # Plot
    plt.figure(figsize=(6.5, 5))

    # 185 MHz group (e.g., blue)
    plt.errorbar(
        omega_185,
        gamma_185,
        xerr=sigma_omega_185,
        yerr=sigma_gamma_185,
        fmt="o",
        capsize=4,
        ecolor="gray",
        elinewidth=1,
        color="tab:blue",
        label=r"$185\ \mathrm{MHz}$",
    )

    # 68 MHz group (e.g., orange)
    plt.errorbar(
        omega_68,
        gamma_68,
        xerr=sigma_omega_68,
        yerr=sigma_gamma_68,
        fmt="o",
        capsize=4,
        ecolor="gray",
        elinewidth=1,
        color="tab:orange",
        label=r"$68\ \mathrm{MHz}$",
    )

    # Median text box for 185 MHz (top right)
    textstr_185 = "\n".join(
        (
            r"$\mathrm{Median\ }\Omega=%.3f\ \mathrm{kHz}$" % median_omega_185,
            r"$\mathrm{Median\ }\gamma=%.3f\ \mathrm{kHz}$" % median_gamma_185,
        )
    )
    plt.gca().text(
        0.97,
        0.97,
        textstr_185,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="tab:blue", alpha=0.2),
    )

    # Median text box for 68 MHz (bottom left)
    textstr_68 = "\n".join(
        (
            r"$\mathrm{Median\ }\Omega=%.3f\ \mathrm{kHz}$" % median_omega_68,
            r"$\mathrm{Median\ }\gamma=%.3f\ \mathrm{kHz}$" % median_gamma_68,
        )
    )
    plt.gca().text(
        0.03,
        0.03,
        textstr_68,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="tab:orange", alpha=0.2),
    )

    # Labels and appearance
    plt.xlabel(r"$\Omega$ (kHz)", fontsize=15)
    plt.ylabel(r"$\gamma$ (kHz)", fontsize=15)
    plt.title(r"$\gamma$ vs $\Omega$", fontsize=15)
    plt.legend(fontsize=11)
    plt.xscale("log")
    plt.yscale("log")
    # plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot histograms
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    fig.suptitle(
        "Histograms of $\Omega$ and $\gamma$ for Different Orientations", fontsize=14
    )

    # Ω histogram for 185 MHz
    axs[0, 0].hist(omega_185, bins=10, color="tab:blue", alpha=0.7)
    axs[0, 0].axvline(median_omega_185, color="k", linestyle="dashed", linewidth=1)
    axs[0, 0].set_title(r"$\Omega$ (kHz) – 185 MHz", fontsize=13)
    axs[0, 0].set_xlabel(r"$\Omega$ (kHz)", fontsize=13)
    axs[0, 0].set_ylabel("Count", fontsize=13)

    # γ histogram for 185 MHz
    axs[1, 0].hist(gamma_185, bins=10, color="tab:blue", alpha=0.7)
    axs[1, 0].axvline(median_gamma_185, color="k", linestyle="dashed", linewidth=1)
    axs[1, 0].set_title(r"$\gamma$ (kHz) – 185 MHz", fontsize=13)
    axs[1, 0].set_xlabel(r"$\gamma$ (kHz)", fontsize=13)
    axs[1, 0].set_ylabel("Count", fontsize=13)

    # Ω histogram for 68 MHz
    axs[0, 1].hist(omega_68, bins=10, color="tab:orange", alpha=0.7)
    axs[0, 1].axvline(median_omega_68, color="k", linestyle="dashed", linewidth=1)
    axs[0, 1].set_title(r"$\Omega$ (kHz) – 68 MHz", fontsize=13)
    axs[0, 1].set_xlabel(r"$\Omega$ (kHz)", fontsize=13)
    axs[0, 1].set_ylabel("Count", fontsize=13)

    # γ histogram for 68 MHz
    axs[1, 1].hist(gamma_68, bins=10, color="tab:orange", alpha=0.7)
    axs[1, 1].axvline(median_gamma_68, color="k", linestyle="dashed", linewidth=1)
    axs[1, 1].set_title(r"$\gamma$ (kHz) – 68 MHz", fontsize=13)
    axs[1, 1].set_xlabel(r"$\gamma$ (kHz)", fontsize=13)
    axs[1, 1].set_ylabel("Count", fontsize=13)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Histogram for Ω (omega)
    plt.figure(figsize=(6, 5))
    plt.hist(omega_185, bins=11, alpha=0.6, label="185 MHz", color="tab:blue")
    plt.hist(omega_68, bins=11, alpha=0.6, label="68 MHz", color="tab:orange")
    plt.axvline(median_omega_185, color="tab:blue", linestyle="dashed", linewidth=1)
    plt.axvline(median_omega_68, color="tab:orange", linestyle="dashed", linewidth=1)
    plt.xlabel(r"$\Omega$ (kHz)", fontsize=13)
    plt.ylabel("Count", fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.title(r"Histogram of $\Omega$ for Two Orientations", fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Histogram for γ (gamma)
    plt.figure(figsize=(6, 5))
    plt.hist(gamma_185, bins=11, alpha=0.6, label="185 MHz", color="tab:blue")
    plt.hist(gamma_68, bins=11, alpha=0.6, label="68 MHz", color="tab:orange")
    plt.axvline(median_gamma_185, color="tab:blue", linestyle="dashed", linewidth=1)
    plt.axvline(median_gamma_68, color="tab:orange", linestyle="dashed", linewidth=1)
    plt.xlabel(r"$\gamma$ (kHz)", fontsize=13)
    plt.ylabel("Count", fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.title(r"Histogram of $\gamma$ for Two Orientations", fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.show()


# nv_indices = np.arange(len(omega))  # Or use your own x-axis

# plt.figure(figsize=(7, 5))

# # Plot omega
# plt.errorbar(
#     nv_indices,
#     omega,
#     yerr=sigma_omega,
#     fmt="o",
#     capsize=4,
#     alpha=0.5,
#     label=r"$\Omega$",
# )

# # Plot gamma
# plt.errorbar(
#     nv_indices,
#     gamma,
#     yerr=sigma_gamma,
#     fmt="s",
#     capsize=4,
#     color="orange",
#     alpha=0.5,
#     label=r"$\gamma$",
# )
# # Text box with median values
# textstr = "\n".join(
#     (
#         r"$\mathrm{Median\ }\Omega=%.3f\ \mathrm{kHz}$" % median_omega,
#         r"$\mathrm{Median\ }\gamma=%.3f\ \mathrm{kHz}$" % median_gamma,
#     )
# )

# # Placement of box: upper right inside axes
# plt.gca().text(
#     0.6,
#     0.97,
#     textstr,
#     transform=plt.gca().transAxes,
#     fontsize=10,
#     verticalalignment="top",
#     horizontalalignment="right",
#     bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
# )
# plt.xlabel("NV Index", fontsize=15)
# plt.ylabel("Rate (KHz)", fontsize=15)
# plt.title("$\Omega$ and $\gamma$", fontsize=15)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.legend(fontsize=11)
# plt.grid(True)
# plt.tight_layout()
# plt.show()


def plot_contrast(nv_list, fit_params):
    nv_indices = np.arange(len(nv_list))
    contrast_list = fit_params[:, 0]
    print(f"contrst_list = {list(contrast_list)}")
    fig, ax = plt.subplots(figsize=(6, 5))
    nv_indices = np.array(nv_indices)
    ax.bar(nv_indices, contrast_list, color="teal", edgecolor="k")
    ax.set_xlabel("NV Index", fontsize=14)
    ax.set_ylabel("Contrast", fontsize=14)
    ax.set_title("XY8 Fit Contrast per NV", fontsize=15)
    ax.tick_params(labelsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_fit_parameters_scatter(
    fit_params,
    fit_errors=None,
    nv_list=None,
    x="Omega",
    y="norm",
    annotate=False,
    logx=False,
    logy=False,
):
    """
    Scatter plot of chosen fit parameters (default Ω vs contrast).

    fit_params: array (N,3) with [norm, rate, offset]
    fit_errors: same shape, optional
    nv_list: optional list of NV indices for labeling
    x, y: which parameters to plot {"norm", "rate", "omega", "T1", "offset"}
    annotate: if True, annotate each point with NV index and params
    """

    fp = np.asarray(fit_params, float)
    fe = np.asarray(fit_errors, float) if fit_errors is not None else None

    def extract(field):
        if field == "norm":
            return fp[:, 0], fe[:, 0] if fe is not None else None, "Contrast (norm)"
        elif field == "gamma":
            return fp[:, 1], fe[:, 1] if fe is not None else None, r"$\gamma$ (1/ms)"
        elif field == "Omega":
            return (
                fp[:, 1] / 3,
                fe[:, 1] / 3 if fe is not None else None,
                r"$\Omega$ (1/ms)",
            )
            rates = fp[:, 1]
            vals = 1.0 / rates
            errs = fe[:, 1] / (rates**2) if fe is not None else None
            return vals, errs, r"$T_1$ (ms)"
        elif field == "offset":
            return fp[:, 2], fe[:, 2] if fe is not None else None, "Offset"
        else:
            raise ValueError("Unknown field")

    xvals, xerr, xlabel = extract(x)
    yvals, yerr, ylabel = extract(y)

    plt.figure(figsize=(6.5, 5.5))
    plt.errorbar(
        xvals,
        yvals,
        xerr=xerr,
        yerr=yerr,
        fmt="o",
        capsize=3,
        elinewidth=1,
        ecolor="gray",
        markersize=5,
        alpha=0.8,
    )

    if annotate and nv_list is not None:
        for i, (xv, yv) in enumerate(zip(xvals, yvals)):
            label = f"NV{nv_list[i]}: {x}={xv:.2f}, {y}={yv:.2f}"
            plt.annotate(
                label, (xv, yv), textcoords="offset points", xytext=(5, 5), fontsize=8
            )

    # --- Compute and add statistics for the y-axis values ---
    x_med = np.median(xvals)
    x_q1 = np.percentile(xvals, 25)
    x_q3 = np.percentile(xvals, 75)
    stats_label = f"$\{x}$: median={x_med:.3g}, Q1={x_q1:.3g}, Q3={x_q3:.3g}"
    plt.plot([], [], " ", label=stats_label)  # invisible handle for legend

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs {xlabel}")
    if logx:
        plt.xscale("log")
    if logy:
        plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="best", fontsize=14, frameon=True)
    plt.tight_layout()


#     plt.show()


import numpy as np
import matplotlib.pyplot as plt


def _nv_key(nv, fallback_index):
    """Stable NV key for matching Ω and γ across datasets."""
    for attr in ("index", "nv_id", "id"):
        v = getattr(nv, attr, None)
        if v is not None:
            return int(v)
    return int(fallback_index)


def plot_combined_omega_gamma(res_omega, res_g113, res_g217, show_medians=True):
    """
    No outlier filtering here.
    Uses:
      Ω = rate/3  from the Ω dataset
      γ = (rate - Ω)/2  for each γ point, using matched Ω per NV if available; else median Ω.
    Plots contrast (norm) vs Ω and vs γ (combined 113+217).
    """

    # ---------- Ω dataset ----------
    fp_o = np.asarray(res_omega["fit_params"], float)  # [norm, rate, offset]
    fe_o = (
        np.asarray(res_omega["fit_errors"], float)
        if res_omega.get("fit_errors") is not None
        else None
    )
    nv_o = res_omega.get("nv_list") or []

    omega_vals = fp_o[:, 1] / 3.0
    omega_errs = (fe_o[:, 1] / 3.0) if fe_o is not None else None
    norm_o = fp_o[:, 0]
    norm_o_errs = (fe_o[:, 0]) if fe_o is not None else None

    # Map Ω (and its error) by NV key for exact pairing
    omega_map = {}
    sigma_omega_map = {}
    for i, nv in enumerate(nv_o):
        key = _nv_key(nv, i)
        omega_map[key] = omega_vals[i]
        if fe_o is not None:
            sigma_omega_map[key] = fe_o[i, 1] / 3.0

    omega_med = float(np.nanmedian(omega_vals)) if omega_vals.size else np.nan

    # ---------- γ datasets (combine 113 + 217) ----------
    # Stack fit_params / fit_errors / nv lists safely
    fp_g = np.vstack([res_g113["fit_params"], res_g217["fit_params"]])
    nv_g = (res_g113.get("nv_list") or []) + (res_g217.get("nv_list") or [])

    fe_g = None
    if (res_g113.get("fit_errors") is not None) and (
        res_g217.get("fit_errors") is not None
    ):
        fe_g = np.vstack([res_g113["fit_errors"], res_g217["fit_errors"]])

    rate_g = fp_g[:, 1]
    norm_g = fp_g[:, 0]
    rate_g_e = fe_g[:, 1] if fe_g is not None else None
    norm_g_e = fe_g[:, 0] if fe_g is not None else None

    # Compute γ = (rate_g - Ω_match)/2, prefer matched Ω; fallback to median Ω
    gamma_vals = np.empty_like(rate_g, dtype=float)
    for i in range(rate_g.shape[0]):
        key = _nv_key(nv_g[i], i) if i < len(nv_g) else i
        omega_here = omega_map.get(key, omega_med)
        gamma_vals[i] = 0.5 * (rate_g[i] - omega_here)

    # Error propagation for γ if both sides have errors
    gamma_errs = None
    if (rate_g_e is not None) or (sigma_omega_map):
        gamma_errs = np.zeros_like(rate_g, dtype=float)
        for i in range(rate_g.shape[0]):
            key = _nv_key(nv_g[i], i) if i < len(nv_g) else i
            sigma_rate = rate_g_e[i] if rate_g_e is not None else 0.0
            sigma_omega = sigma_omega_map.get(key, 0.0)
            gamma_errs[i] = 0.5 * np.sqrt(sigma_rate**2 + sigma_omega**2)

    # --- build individual masks ---
    k = 15
    mask_omega_val = iqr_outlier_mask(omega_vals, k)
    mask_gamma_val = iqr_outlier_mask(gamma_vals, k)
    mask_norm_o = iqr_outlier_mask(norm_o, k)
    mask_norm_g = iqr_outlier_mask(norm_g, k)

    mask_omega_err = np.ones_like(omega_vals, dtype=bool)
    if omega_errs is not None:
        rel_err_omega = omega_errs / np.maximum(np.abs(omega_vals), 1e-20)
        mask_omega_err = rel_err_omega < 100.0  # example: <50% relative error

    mask_gamma_err = np.ones_like(gamma_vals, dtype=bool)
    if gamma_errs is not None:
        rel_err_gamma = gamma_errs / np.maximum(np.abs(gamma_vals), 1e-20)
        mask_gamma_err = rel_err_gamma < 100.0

    # remove negative values ---
    # mask_nonneg = (omega_vals > 0) & (gamma_vals > 0)
    # --- combine into a single mask ---
    mask_all = (
        mask_omega_val
        & mask_gamma_val
        & mask_norm_o
        & mask_norm_g
        & mask_omega_err
        & mask_gamma_err
    )

    # --- apply to all arrays ---
    omega_vals = omega_vals[mask_all]
    omega_errs = omega_errs[mask_all] if omega_errs is not None else None
    norm_o = norm_o[mask_all]
    norm_o_errs = norm_o_errs[mask_all] if norm_o_errs is not None else None

    gamma_vals = gamma_vals[mask_all]
    gamma_errs = gamma_errs[mask_all] if gamma_errs is not None else None
    norm_g = norm_g[mask_all]
    norm_g_errs = norm_g_e[mask_all] if norm_g_e is not None else None

    gamma_med = float(np.nanmedian(gamma_vals)) if gamma_vals.size else np.nan

    print(len(omega_vals), len(gamma_vals))
    # ---------- Plot (no filtering) ----------
    plt.figure(figsize=(7, 6))
    # Ω points
    plt.errorbar(
        omega_vals,
        norm_o,
        xerr=omega_errs,
        yerr=norm_o_errs,
        fmt="o",
        markersize=5,
        capsize=3,
        elinewidth=1,
        alpha=0.85,
        label="Ω ",
    )

    # γ points (combined)
    plt.errorbar(
        gamma_vals,
        norm_g,
        xerr=gamma_errs,
        yerr=norm_g_errs,
        fmt="s",
        markersize=5,
        capsize=3,
        elinewidth=1,
        alpha=0.85,
        label="γ",
    )

    if show_medians:
        if np.isfinite(omega_med):
            plt.axvline(
                omega_med,
                linestyle="--",
                alpha=0.6,
                label=f"Ω median = {omega_med:.3g}",
            )
        if np.isfinite(gamma_med):
            plt.axvline(
                gamma_med, linestyle=":", alpha=0.6, label=f"γ median = {gamma_med:.3g}"
            )

    plt.xlabel(r"Parameter value ($\Omega$ or $\gamma$)")
    # plt.xlabel(r"Parameter value ($\gamma$)")
    plt.ylabel("Contrast (norm)")
    plt.title("Contrast vs Ω and γ")
    # plt.title("Contrast vs γ")
    plt.grid(True, linestyle="--", alpha=0.5)
    # plt.xscale("log")
    # plt.yscale("log")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.show()


def plot_omega_vs_gamma(res_omega, res_g113, res_g217, show_medians=True):
    """
    No outlier filtering here.
    Uses:
      Ω = rate/3  from the Ω dataset
      γ = (rate - Ω)/2  for each γ point, using matched Ω per NV if available; else median Ω.
    Plots contrast (norm) vs Ω and vs γ (combined 113+217).
    """

    # ---------- Ω dataset ----------
    fp_o = np.asarray(res_omega["fit_params"], float)  # [norm, rate, offset]
    fe_o = (
        np.asarray(res_omega["fit_errors"], float)
        if res_omega.get("fit_errors") is not None
        else None
    )
    nv_o = res_omega.get("nv_list") or []

    omega_vals = fp_o[:, 1] / 3.0
    omega_errs = (fe_o[:, 1] / 3.0) if fe_o is not None else None
    norm_o = fp_o[:, 0]
    norm_o_errs = (fe_o[:, 0]) if fe_o is not None else None

    # Map Ω (and its error) by NV key for exact pairing
    omega_map = {}
    sigma_omega_map = {}
    for i, nv in enumerate(nv_o):
        key = _nv_key(nv, i)
        omega_map[key] = omega_vals[i]
        if fe_o is not None:
            sigma_omega_map[key] = fe_o[i, 1] / 3.0

    omega_med = float(np.nanmedian(omega_vals)) if omega_vals.size else np.nan

    # ---------- γ datasets (combine 113 + 217) ----------
    # Stack fit_params / fit_errors / nv lists safely
    fp_g = np.vstack([res_g113["fit_params"], res_g217["fit_params"]])
    nv_g = (res_g113.get("nv_list") or []) + (res_g217.get("nv_list") or [])

    fe_g = None
    if (res_g113.get("fit_errors") is not None) and (
        res_g217.get("fit_errors") is not None
    ):
        fe_g = np.vstack([res_g113["fit_errors"], res_g217["fit_errors"]])

    rate_g = fp_g[:, 1]
    norm_g = fp_g[:, 0]
    rate_g_e = fe_g[:, 1] if fe_g is not None else None
    norm_g_e = fe_g[:, 0] if fe_g is not None else None

    # --- Compute γ with per-point Ω and remember the Ω used ---
    gamma_vals = np.empty_like(rate_g, dtype=float)
    omega_for_gamma = np.empty_like(rate_g, dtype=float)  # NEW

    matched_flag = np.zeros_like(
        rate_g, dtype=bool
    )  # mark true matches (not median fallback)

    for i in range(rate_g.shape[0]):
        key = _nv_key(nv_g[i], i) if i < len(nv_g) else i
        if key in omega_map:
            omega_here = omega_map[key]
            matched_flag[i] = True
        else:
            omega_here = omega_med  # fallback
        omega_for_gamma[i] = omega_here
        gamma_vals[i] = 0.5 * (rate_g[i] - omega_here)

    # --- Error propagation for γ unchanged, but also compute omega errors for γ points ---
    gamma_errs = None
    omega_for_gamma_errs = None  # NEW
    if (rate_g_e is not None) or (sigma_omega_map):
        gamma_errs = np.zeros_like(rate_g, dtype=float)
        omega_for_gamma_errs = np.zeros_like(rate_g, dtype=float)
        for i in range(rate_g.shape[0]):
            key = _nv_key(nv_g[i], i) if i < len(nv_g) else i
            sigma_rate = rate_g_e[i] if rate_g_e is not None else 0.0
            sigma_omega = sigma_omega_map.get(key, 0.0)
            gamma_errs[i] = 0.5 * np.sqrt(sigma_rate**2 + sigma_omega**2)
            omega_for_gamma_errs[i] = sigma_omega

    # ----------------------
    # MASKS: build them per-plot (do NOT combine Ω- and γ-length masks)
    # ----------------------

    # 1) Ω vs norm (use only Ω arrays)
    k = 15
    mask_o_val = iqr_outlier_mask(omega_vals, k)
    mask_o_norm = iqr_outlier_mask(norm_o, k)
    mask_o_err = np.ones_like(omega_vals, dtype=bool)
    if omega_errs is not None:
        rel = omega_errs / np.maximum(np.abs(omega_vals), 1e-20)
        mask_o_err = rel < 10.0  # e.g., <100% rel error
    mask_o = mask_o_val & mask_o_norm & mask_o_err

    # 2) γ vs norm (use only γ arrays)
    mask_g_val = iqr_outlier_mask(gamma_vals, k)
    mask_g_norm = iqr_outlier_mask(norm_g, k)
    mask_g_err = np.ones_like(gamma_vals, dtype=bool)
    if gamma_errs is not None:
        rel = gamma_errs / np.maximum(np.abs(gamma_vals), 1e-20)
        mask_g_err = rel < 10.0
    mask_g = mask_g_val & mask_g_norm & mask_g_err

    # 3) γ vs Ω correlation (use the Ω that was used for each γ)
    mask_c_val_x = iqr_outlier_mask(omega_for_gamma, k)
    mask_c_val_y = iqr_outlier_mask(gamma_vals, k)
    mask_c_err = np.ones_like(gamma_vals, dtype=bool)
    if (gamma_errs is not None) and (omega_for_gamma_errs is not None):
        relx = omega_for_gamma_errs / np.maximum(np.abs(omega_for_gamma), 1e-20)
        rely = gamma_errs / np.maximum(np.abs(gamma_vals), 1e-20)
        mask_c_err = (relx < 10.0) & (rely < 10.0)

    # OPTIONAL: only keep points with true Ω–γ matches (no median fallback)
    strict_pairs_only = True
    mask_matched = (
        matched_flag if strict_pairs_only else np.ones_like(matched_flag, dtype=bool)
    )

    mask_c = mask_c_val_x & mask_c_val_y & mask_c_err & mask_matched

    print(len(mask_c))
    # ------------- plotting -------------
    # Ω vs norm
    plt.figure(figsize=(7, 6))
    plt.errorbar(
        omega_vals[mask_o],
        norm_o[mask_o],
        xerr=(omega_errs[mask_o] if omega_errs is not None else None),
        yerr=(norm_o_errs[mask_o] if norm_o_errs is not None else None),
        fmt="o",
        markersize=5,
        capsize=3,
        elinewidth=1,
        alpha=0.85,
        label="Ω",
    )
    if show_medians and np.isfinite(omega_med):
        plt.axvline(omega_med, ls="--", alpha=0.6, label=f"Ω median = {omega_med:.3g}")
    plt.xlabel(r"$\Omega$ (1/ms)")
    plt.ylabel("Contrast (norm)")
    plt.title("Contrast vs Ω")
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()

    # γ vs norm
    plt.figure(figsize=(7, 6))
    plt.errorbar(
        gamma_vals[mask_g],
        norm_g[mask_g],
        xerr=(gamma_errs[mask_g] if gamma_errs is not None else None),
        yerr=(norm_g_e[mask_g] if norm_g_e is not None else None),
        fmt="s",
        markersize=5,
        capsize=3,
        elinewidth=1,
        alpha=0.85,
        label="γ",
    )
    gamma_med = float(np.nanmedian(gamma_vals[mask_g])) if np.any(mask_g) else np.nan
    if show_medians and np.isfinite(gamma_med):
        plt.axhline(gamma_med, ls=":", alpha=0.6, label=f"γ median = {gamma_med:.3g}")

    plt.xlabel(r"$\gamma$ (1/ms)")
    plt.ylabel("Contrast (norm)")
    plt.title("Contrast vs γ")
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()

    # γ vs Ω (per-point Ω used for that γ)
    plt.figure(figsize=(7, 6))
    plt.errorbar(
        omega_for_gamma[mask_c],
        gamma_vals[mask_c],
        xerr=(
            omega_for_gamma_errs[mask_c] if omega_for_gamma_errs is not None else None
        ),
        yerr=(gamma_errs[mask_c] if gamma_errs is not None else None),
        fmt="o",
        markersize=5,
        capsize=3,
        elinewidth=1,
        ecolor="gray",
        alpha=0.8,
    )
    if show_medians and np.isfinite(gamma_med):
        plt.axhline(gamma_med, ls=":", alpha=0.6, label=f"γ median = {gamma_med:.3g}")
    if show_medians and np.isfinite(omega_med):
        plt.axvline(omega_med, ls="--", alpha=0.6, label=f"Ω median = {omega_med:.3g}")
    plt.xlabel(r"$\Omega$ (1/ms)")
    plt.ylabel(r"$\gamma$ (1/ms)")
    plt.title(rf"$\gamma$ vs $\Omega$ ({len(omega_for_gamma[mask_c])}NVs)")
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()

    # Compute ratio (per γ point, using the Ω used for that γ)
    ratio_vals = gamma_vals / np.maximum(omega_for_gamma, 1e-20)  # avoid divide by 0

    # Error propagation: σ(ratio) ≈ ratio * sqrt[(σγ/γ)^2 + (σΩ/Ω)^2]
    ratio_errs = None
    if (gamma_errs is not None) and (omega_for_gamma_errs is not None):
        rel_γ = gamma_errs / np.maximum(np.abs(gamma_vals), 1e-20)
        rel_Ω = omega_for_gamma_errs / np.maximum(np.abs(omega_for_gamma), 1e-20)
        ratio_errs = np.abs(ratio_vals) * np.sqrt(rel_γ**2 + rel_Ω**2)

    # Mask outliers for ratio
    mask_ratio_val = iqr_outlier_mask(ratio_vals, k=15)
    mask_ratio_err = np.ones_like(ratio_vals, dtype=bool)
    if ratio_errs is not None:
        rel_err = ratio_errs / np.maximum(np.abs(ratio_vals), 1e-20)
        mask_ratio_err = rel_err < 1.0  # keep <100% rel error

    mask_ratio = mask_ratio_val & mask_ratio_err & mask_matched

    # -------- Plot --------
    ratio_med = np.nanmedian(ratio_vals[mask_ratio])

    plt.figure(figsize=(7, 6))
    plt.errorbar(
        np.arange(np.sum(mask_ratio)),
        ratio_vals[mask_ratio],
        yerr=(ratio_errs[mask_ratio] if ratio_errs is not None else None),
        fmt="o",
        markersize=5,
        capsize=3,
        elinewidth=1,
        alpha=0.85,
    )
    plt.axhline(2.0, ls="--", color="red", alpha=0.6, label="Reference = 2")
    plt.axhline(1.0, ls="--", color="gray", alpha=0.6, label="Reference = 1")
    plt.xlabel("NV index (filtered)")
    plt.ylabel(r"$\gamma / \Omega$")
    plt.title(r"Ratio $\gamma / \Omega$ for each NV")
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.show()


# import numpy as np
# import matplotlib.pyplot as plt

# def iqr_outlier_mask(values, k=1.5):
#     v = np.asarray(values, float)
#     finite = np.isfinite(v)
#     if not np.any(finite):
#         return np.zeros_like(v, dtype=bool)
#     q1, q3 = np.percentile(v[finite], [25, 75])
#     iqr = q3 - q1
#     lower, upper = q1 - k*iqr, q3 + k*iqr
#     mask = (v >= lower) & (v <= upper) & finite
#     return mask

# def plot_fit_parameters_scatter(
#     fit_params,
#     fit_errors=None,
#     nv_list=None,
#     x="omega",              # {"norm","rate","omega","T1","offset"}
#     y="norm",
#     annotate=False,
#     logx=False,
#     logy=False,
#     drop_x_outliers=False,  # IQR filter on x
#     iqr_k=1.5,
# ):
#     """
#     Scatter of chosen fit parameters with optional error bars & IQR outlier removal.
#     Expects single-exp params: fit_params[:,0]=norm, [:,1]=rate(=Ω), [:,2]=offset
#     """

#     # ---- Validate/shape ----
#     fp = np.asarray(fit_params, dtype=float)
#     if fp.ndim != 2 or fp.shape[1] < 3:
#         print(f"Warning: fit_params has shape {fp.shape}. Expected (N,3) float. "
#               "This often happens if some NVs used double-exp -> ragged array.")
#         # Try to coerce by keeping only first 3 cols if possible
#         if fp.ndim == 2 and fp.shape[1] >= 3:
#             fp = fp[:, :3].astype(float)
#         else:
#             return

#     fe = None
#     if fit_errors is not None:
#         try:
#             fe = np.asarray(fit_errors, dtype=float)
#             if fe.shape != fp.shape:
#                 print(f"Warning: fit_errors shape {fe.shape} != fit_params shape {fp.shape}. Disabling error bars.")
#                 fe = None
#         except Exception:
#             print("Warning: could not convert fit_errors to float. Disabling error bars.")
#             fe = None

#     # ---- Extract fields ----
#     def extract(field):
#         if field == "norm":
#             vals = fp[:, 0]; errs = (fe[:, 0] if fe is not None else None); label = "Contrast (norm)"
#         elif field == "rate":
#             vals = fp[:, 1]; errs = (fe[:, 1] if fe is not None else None); label = r"Rate $\Gamma$ (1/ms)"
#         elif field == "omega":
#             # you said the fitted rate already equals Ω
#             vals = fp[:, 1]; errs = (fe[:, 1] if fe is not None else None); label = r"$\Omega$ (1/ms)"
#         elif field == "T1":
#             rates = fp[:, 1]
#             vals = 1.0 / rates
#             errs = (fe[:, 1] / (rates**2)) if fe is not None else None
#             label = r"$T_1$ (ms)"
#         elif field == "offset":
#             vals = fp[:, 2]; errs = (fe[:, 2] if fe is not None else None); label = "Offset"
#         else:
#             raise ValueError("Unknown field (use one of {'norm','rate','omega','T1','offset'})")
#         return np.asarray(vals, float), (np.asarray(errs, float) if errs is not None else None), label

#     xvals, xerr, xlabel = extract(x)
#     yvals, yerr, ylabel = extract(y)

#     # ---- Finite mask ----
#     finite = np.isfinite(xvals) & np.isfinite(yvals)
#     if xerr is not None: finite &= np.isfinite(xerr)
#     if yerr is not None: finite &= np.isfinite(yerr)
#     if not np.any(finite):
#         print("Nothing to plot: all points are non-finite (NaN/Inf) after extraction.")
#         return

#     # ---- Optional IQR outlier removal on x ----
#     if drop_x_outliers:
#         keep_iqr = iqr_outlier_mask(xvals[finite], k=iqr_k)
#         full_mask = finite.copy()
#         full_mask[np.where(finite)[0][~keep_iqr]] = False
#         finite = full_mask

#     # Apply mask
#     xvals, yvals = xvals[finite], yvals[finite]
#     xerr = (xerr[finite] if xerr is not None else None)
#     yerr = (yerr[finite] if yerr is not None else None)
#     kept_idx = np.where(finite)[0]

#     if xvals.size == 0:
#         print("Nothing to plot: zero points after masking/outlier removal.")
#         return

#     # ---- Plot ----
#     plt.figure(figsize=(6, 5))
#     plt.errorbar(
#         xvals, yvals,
#         xerr=xerr, yerr=yerr,
#         fmt="o", capsize=3, elinewidth=1, ecolor="gray",
#         markersize=5, alpha=0.85
#     )

#     # Annotations (respect masked indices)
#     if annotate and nv_list is not None:
#         nv_arr = np.asarray(nv_list)
#         for i, (xv, yv, idx) in enumerate(zip(xvals, yvals, kept_idx)):
#             nv_label = nv_arr[idx] if idx < len(nv_arr) else idx
#             plt.annotate(f"NV{nv_label}", (xv, yv),
#                          textcoords="offset points", xytext=(5, 5), fontsize=8)

#     # Medians/correlation for quick sanity (on kept points)
#     medx, medy = np.median(xvals), np.median(yvals)
#     plt.gca().text(0.97, 0.97, f"Median {x}: {medx:.3g}\nMedian {y}: {medy:.3g}",
#                    transform=plt.gca().transAxes, fontsize=10,
#                    va="top", ha="right", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
#     if xvals.size > 1:
#         r = np.corrcoef(xvals, yvals)[0, 1]
#         plt.gca().text(0.03, 0.97, f"Pearson r = {r:.3f}",
#                        transform=plt.gca().transAxes, fontsize=10,
#                        va="top", ha="left", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

#     plt.xlabel(xlabel, fontsize=14)
#     plt.ylabel(ylabel, fontsize=14)
#     if logx: plt.xscale("log")
#     if logy: plt.yscale("log")
#     plt.title(f"{ylabel} vs {xlabel}", fontsize=15)
#     plt.grid(True, linestyle="--", alpha=0.5)
#     plt.tight_layout()
#     plt.show()


def iqr_outlier_mask(values, k=1.5):
    """
    Returns a boolean mask of INLIERS (True = keep) using IQR rule.
    values: array-like
    k: multiplier (1.5 for mild outliers, 3 for extreme)
    """
    v = np.asarray(values, dtype=float)
    finite = np.isfinite(v)
    q1, q3 = np.percentile(v[finite], [25, 75])
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    mask = (v >= lower) & (v <= upper) & finite
    return mask


# ----- WHICH COLUMNS HOLD WHICH PARAMS -----


def run_fit_batch(file_ids, selected_indices, use_double_fit=False):
    data = widefield.process_multiple_files(file_ids, load_npz=True)
    (
        fit_params,
        fit_functions,
        residuals,
        taus,
        norm_counts,
        norm_counts_ste,
        nv_list,
        fit_errors,
    ) = process_and_fit_data(
        data, use_double_fit=use_double_fit, selected_indices=selected_indices
    )
    return {
        "fit_params": fit_params,
        "fit_errors": fit_errors,
        "nv_list": nv_list,
    }


# Usage Example
if __name__ == "__main__":
    kpl.init_kplotlib()
    # sig0 is Prepare--> wait t --> readout S(0,0)
    # sig1 repare --> wait t --> ms=-1 pi pulse --> readout S(0,-1)
    file_ids = [1811090565552, 1811157726815, 1811214987613, 1811261374947]

    # Orientation 68 MHz
    # sig0 is Prepare -->  ms=-1 pi pulse --wait t -->  ms=-1 pi pulse --> readout S(-1,-1)
    # sig0 is Prepare -->  ms=-1 pi pulse --wait t -->  ms=+1 pi pulse --> readout S(-1,+1)
    # file_ids = [1812751434024, 1812874254999, 1812983279270, 1813093367715]

    # Orientation 185 MHz
    # sig0 is Prepare -->  ms=-1 pi pulse --wait t -->  ms=-1 pi pulse --> readout S(-1,-1)
    # sig0 is Prepare -->  ms=-1 pi pulse --wait t -->  ms=+1 pi pulse --> readout S(-1,+1)
    # file_ids = [1813232235609, 1813330917197, 1813433255605, 1813532080202]
    # fmt: off
    # # orientation_68MHz_indice
    selected_indices_68MHz = [7, 8, 9, 11, 14, 18, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 35, 38, 44, 45, 46, 47, 48, 49, 53, 55, 57, 58, 60, 62, 64, 66, 67, 68, 69, 70, 71, 72, 73]
    # orientation_185MHz_indices
    selected_indices_185MHz  =[1, 2, 3, 4, 5, 6, 10, 12, 13, 15, 16, 17, 19, 20, 21, 23, 29, 34, 36, 39, 40, 41, 42, 43, 50, 51, 52, 54, 56, 59, 61, 63, 65, 74]
    # combined_indices
    selected_indices= selected_indices_185MHz + selected_indices_68MHz
    # selected_indices= selected_indices_185MHz
    # selected_indices= selected_indices_68MHz
    # print(selected_indices)
    #fmt: off
    # file_path, all_file_ids_str = widefield.combined_filename(file_ids)
    # print(f"File path: {file_path}")\

    ### omega all nv
    # file_ids = ["2025_10_14-21_08_23-rubin-nv0_2025_09_08", "2025_10_15-01_27_23-rubin-nv0_2025_09_08"]
    ### gamma 113MHz orientaion
    # file_ids = ["2025_10_18-07_23_48-rubin-nv0_2025_09_08", "2025_10_18-03_24_11-rubin-nv0_2025_09_08"]
    # indices_113_MHz = [0, 1, 3, 6, 10, 14, 16, 17, 19, 23, 24, 25, 26, 27, 32, 33, 34, 35, 37, 38, 41, 49, 50, 51, 53, 54, 55, 60, 62, 63, 64, 66, 67, 68, 70, 72, 73, 74, 75, 76, 78, 80, 81, 82, 83, 84, 86, 88, 90, 92, 93, 95, 96, 99, 100, 101, 102, 103, 105, 108, 109, 111, 113, 114]
    # gamma 217Mhz orientation
    # file_ids = ["2025_10_18-21_17_09-rubin-nv0_2025_09_08", "2025_10_18-17_10_51-rubin-nv0_2025_09_08"]
    # indices_217_MHz = [2, 4, 5, 7, 8, 9, 11, 12, 13, 15, 18, 20, 21, 22, 28, 29, 30, 31, 36, 39, 40, 42, 43, 44, 45, 46, 47, 48, 52, 56, 57, 58, 59, 61, 65, 69, 71, 77, 79, 85, 87, 89, 91, 94, 97, 98, 104, 106, 107, 110, 112, 115, 116, 117]

    file_ids = ["2025_11_03-10_32_16-johnson-nv0_2025_10_21"] #omaga
    # file_ids = ["2025_11_03-19_34_20-johnson-nv0_2025_10_21",
    #             "2025_11_14-08_23_39-johnson-nv0_2025_10_21"] ## m111 orientation
    # file_ids = ["2025_11_14-08_23_39-johnson-nv0_2025_10_21"] ## m111 orientation
    data = widefield.process_multiple_files(file_ids, load_npz=True)
    # data = dm.get_raw_data(file_id=1550610460299)  # Example file ID
    (
        fit_params,
        fit_functions,
        residuals,
        taus,
        norm_counts,
        norm_counts_ste,
        nv_list,
        fit_errors,
    ) = process_and_fit_data(data, use_double_fit=False, selected_indices=None)

    # print(f"contrst_list = {list(offset_list)}")
    # plot_contrast(nv_list, fit_params)
    # plot_fitted_data(
    #     nv_list,
    #     taus,
    #     norm_counts,
    #     norm_counts_ste,
    #     fit_functions,
    #     fit_params,
    #     fit_errors,
    #     num_cols=10,
    #     selected_indices=None,
    # )
    # scatter_fitted_parameters(fit_params, nv_list)
    # omega = fit_params[:, 1] / 3.0   # rate → Ω
    # mask = iqr_outlier_mask(omega, k=1.5)  # or k=3.0 for stricter
    # fit_params = fit_params[mask]
    # fit_errors = fit_errors[mask] if fit_errors is not None else None
    # nv_list = [nv for i, nv in enumerate(nv_list) if mask[i]]

    # scatter_fitted_parameters(fit_params, nv_list)

    # 1. Filter by omega outliers
    omega = fit_params[:, 1] / 3.0   # rate → Ω
    mask1 = iqr_outlier_mask(omega, k=10)  # keep "normal" Ω values

    # 2. Filter by error threshold
    if fit_errors is not None:
        # e.g. reject if relative error > 50% or absolute error too large
        error_threshold = 10.0   # 50% relative error
        rel_err = fit_errors[:, 1] / fit_params[:, 1]
        mask2 = rel_err < error_threshold
    else:
        mask2 = np.ones_like(mask1, dtype=bool)

    # 3. Combine both masks
    mask = mask1 & mask2

    # 4. Apply mask consistently
    fit_params = fit_params[mask]
    fit_errors = fit_errors[mask] if fit_errors is not None else None
    nv_list   = [nv for i, nv in enumerate(nv_list) if mask[i]]


    # Optional: quick log of how many were removed
    print(f"IQR filter kept {mask.sum()}/{len(mask)} NVs.")
    plot_fit_parameters_scatter(fit_params, fit_errors=fit_errors, nv_list=nv_list, x= "Omega")
    # plot_T1_with_errorbars(fit_params, fit_errors, nv_list)
    plot_fitted_data_separately(
        nv_list,
        taus,
        norm_counts,
        norm_counts_ste,
        fit_functions,
        fit_params,
        fit_errors,
    )
    plt.show(block = True)
    sys.exit()
    # plots_rates_omega_gamma()

    ############
    # Ω: all NVs
    omega_files = [
        "2025_11_03-10_32_16-johnson-nv0_2025_10_21",
    ]

    # γ: 113 MHz orientation
    gamma_113_files = [
        "2025_10_18-07_23_48-rubin-nv0_2025_09_08",
        "2025_10_18-03_24_11-rubin-nv0_2025_09_08",
    ]
    indices_113_MHz = [1, 3, 6, 10, 14, 16, 17, 19, 23, 24, 25, 26, 27, 32, 33, 34, 35,
                    37, 38, 41, 49, 50, 51, 53, 54, 55, 60, 62, 63, 64, 66, 67, 68, 70,
                    72, 73, 74, 75, 76, 78, 80, 81, 82, 83, 84, 86, 88, 90, 92, 93, 95,
                    96, 99, 100, 101, 102, 103, 105, 108, 109, 111, 113, 114]

    # γ: 217 MHz orientation
    gamma_217_files = [
        "2025_10_18-21_17_09-rubin-nv0_2025_09_08",
        "2025_10_18-17_10_51-rubin-nv0_2025_09_08",
        "2025_10_19-01_34_07-rubin-nv0_2025_09_08"
    ]
    indices_217_MHz = [2, 4, 5, 7, 8, 9, 11, 12, 13, 15, 18, 20, 21, 22, 28, 29, 30, 31, 36,
                    39, 40, 42, 43, 44, 45, 46, 47, 48, 52, 56, 57, 58, 59, 61, 65, 69,
                    71, 77, 79, 85, 87, 89, 91, 94, 97, 98, 104, 106, 107, 110, 112, 115,
                    116, 117]


    ############
    # Johnson data
    ############
    # Ω: all NVs
    omega_files = [
        "2025_10_14-21_08_23-rubin-nv0_2025_09_08",
        "2025_10_15-01_27_23-rubin-nv0_2025_09_08",
    ]

    # γ: 113 MHz orientation
    gamma_113_files = [
        "2025_10_18-07_23_48-rubin-nv0_2025_09_08",
        "2025_10_18-03_24_11-rubin-nv0_2025_09_08",
    ]
    indices_113_MHz = [1, 3, 6, 10, 14, 16, 17, 19, 23, 24, 25, 26, 27, 32, 33, 34, 35,
                    37, 38, 41, 49, 50, 51, 53, 54, 55, 60, 62, 63, 64, 66, 67, 68, 70,
                    72, 73, 74, 75, 76, 78, 80, 81, 82, 83, 84, 86, 88, 90, 92, 93, 95,
                    96, 99, 100, 101, 102, 103, 105, 108, 109, 111, 113, 114]

    # γ: 217 MHz orientation
    gamma_217_files = [
        "2025_10_18-21_17_09-rubin-nv0_2025_09_08",
        "2025_10_18-17_10_51-rubin-nv0_2025_09_08",
        "2025_10_19-01_34_07-rubin-nv0_2025_09_08"
    ]
    indices_217_MHz = [2, 4, 5, 7, 8, 9, 11, 12, 13, 15, 18, 20, 21, 22, 28, 29, 30, 31, 36,
                    39, 40, 42, 43, 44, 45, 46, 47, 48, 52, 56, 57, 58, 59, 61, 65, 69,
                    71, 77, 79, 85, 87, 89, 91, 94, 97, 98, 104, 106, 107, 110, 112, 115,
                    116, 117]

    # ======== RUN ALL THREE ========

    # Ω (looser IQR like your example; error cutoff optional)
    res_omega = run_fit_batch(omega_files, None)
    # γ @ 113 MHz
    res_g113 = run_fit_batch(gamma_113_files, indices_113_MHz)
    # γ @ 217 MHz
    res_g217 = run_fit_batch(gamma_217_files, indices_217_MHz)

    # plot_combined_omega_gamma(res_omega, res_g113, res_g217)
    plot_omega_vs_gamma(res_omega, res_g113, res_g217)
    kpl.show(block=True)
