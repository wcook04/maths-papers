# scripts/main_analysis_v2.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit
from scipy.fft import rfft, rfftfreq
from scipy.signal import detrend, windows
import mpmath
from multiprocessing import Pool, cpu_count
from pathlib import Path
from tqdm import tqdm
import time
import warnings

# --- Configuration & Constants ---
# File and Directory Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = BASE_DIR / "results"

ZEROS_FILEPATH = RAW_DATA_DIR / "zeros6_clean.csv"

# P_total(t) and C2_m1(t) Parameters
P_PRIMES = np.array([2, 3, 5, 7, 11], dtype=float)
P_M_TERMS = 5
LN_P_PRIMES = np.log(P_PRIMES)
LN_2 = np.log(2.0)

# Numerical Parameters
LOG_FLOOR_VALUE = -10.0
LOG_FLOOR_THRESHOLD = 1e-10
RATIO_C2M1_ZERO_THRESHOLD = 1e-10
MPMATH_DPS = 30

# Derivative Subset Parameters
DERIVATIVE_SUBSET_FIRST_N = 10000
DERIVATIVE_SUBSET_RANDOM_N = 10000
RANDOM_SEED = 42

# Phase 4: Spectral Analysis Parameters
CONTINUOUS_T_MAX_ZERO_INDEX = 10000
CONTINUOUS_T_DELTA = 0.05
FFT_WINDOW_TYPE = 'hann'

# Plotting
PLT_STYLE = 'seaborn-v0_8-whitegrid'
DEFAULT_DPI = 300
PLOT_RANDOM_SUBSET_SIZE_PHASE1 = 200000

# DataFrame column names (to centralize and avoid typos)
COL_T = 't'
COL_P_TOTAL = 'P_total'
COL_C2_M1 = 'C2_m1'
COL_LOG_ABS_P_TOTAL = 'log_abs_P_total'
COL_LOG_ABS_C2_M1 = 'log_abs_C2_m1'
COL_PHASE_C2_M1 = 'Phase_C2_m1'
COL_ZETA_PRIME_REAL = 'zeta_prime_real'
COL_ZETA_PRIME_IMAG = 'zeta_prime_imag'
COL_ARG_ZETA_PRIME = 'arg_zeta_prime'
COL_ABS_ZETA_PRIME = 'abs_zeta_prime'
COL_RATIO_P_C2M1 = 'Ratio_P_C2m1'
COL_LOG_ABS_C2_M1_BIN = 'log_abs_C2_m1_bin'
COL_PHASE_DIST_0_PI = 'Phase_Dist_0_pi'
COL_MOD4_INDEX_C2M1 = 'Mod4_Index_C2m1'
COL_ABS_ARG_ZETA_PRIME_MOD_PI = 'abs_arg_zeta_prime_mod_pi'
COL_DELTA_MOD4_INDEX = 'delta_Mod4_Index'


# --- Utility Functions --- (Identical to previous, so omitted for brevity, will be included in final file)
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def save_plot(fig, phase_name, plot_name):
    plot_dir = RESULTS_DIR / phase_name / "plots"
    ensure_dir(plot_dir)
    fig.savefig(plot_dir / f"{plot_name}.png", dpi=DEFAULT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot: {plot_dir / f'{plot_name}.png'}")

def save_summary(text, phase_name, summary_name):
    summary_dir = RESULTS_DIR / phase_name / "summaries"
    ensure_dir(summary_dir)
    with open(summary_dir / f"{summary_name}.txt", "w") as f:
        f.write(text)
    print(f"Saved summary: {summary_dir / f'{summary_name}.txt'}")

def save_dataframe(df, dir_path, filename):
    ensure_dir(dir_path)
    filepath = dir_path / f"{filename}.parquet"
    df.to_parquet(filepath)
    print(f"Saved DataFrame to {filepath}")

def load_dataframe(dir_path, filename):
    filepath = dir_path / f"{filename}.parquet"
    if filepath.exists():
        print(f"Loading DataFrame from {filepath}")
        return pd.read_parquet(filepath)
    return None

# --- Computation Functions (from Phase 0, now more modular) ---

def compute_P_total_vec(t_values, primes, m_terms, ln_primes_arr):
    P_total = np.zeros_like(t_values, dtype=float)
    t_values_col = t_values[:, np.newaxis, np.newaxis] 

    for i, p in enumerate(primes):
        ln_p = ln_primes_arr[i]
        m_range = np.arange(1, m_terms + 1)[np.newaxis, :, np.newaxis] 
        term_sum = np.sum(
            (1.0 / (m_range * (p**(m_range / 2.0)))) * np.cos(m_range * t_values_col * ln_p),
            axis=1
        )
        P_total += term_sum.flatten()
    return -P_total

def compute_C2_m1_vec(t_values, ln_2_val):
    return -(1.0 / np.sqrt(2.0)) * np.cos(t_values * ln_2_val)

def compute_log_abs(values, threshold, floor_val):
    abs_values = np.abs(values)
    log_abs_values = np.full_like(values, floor_val, dtype=float)
    mask = abs_values >= threshold
    if np.any(mask): # Check if there are any True values in mask
      log_abs_values[mask] = np.log10(abs_values[mask])
    return log_abs_values


def compute_phase_C2_m1_vec(t_values, ln_2_val):
    return (t_values * ln_2_val) % (2 * np.pi)

def _compute_zeta_derivative_worker(s_complex_tuple): # Expects (index, s_complex)
    original_index, s_complex = s_complex_tuple
    mpmath.mp.dps = MPMATH_DPS
    try:
        val = mpmath.diff(mpmath.zeta, s_complex, 1)
        return original_index, val
    except Exception:
        return original_index, mpmath.nan


# --- Phase 0: Foundational Data Preparation ---
def run_phase0(zeros_filepath, force_recompute=False):
    print("\n--- Running Phase 0: Foundational Data Preparation ---")
    start_time = time.time()
    phase0_output_filename = "df_zeros_augmented_phase0"
    
    df_zeros = load_dataframe(PROCESSED_DATA_DIR, phase0_output_filename)

    if df_zeros is not None and not force_recompute:
        print("Loaded pre-computed Phase 0 DataFrame. Verifying columns...")
        required_cols = [COL_T, COL_P_TOTAL, COL_C2_M1, COL_LOG_ABS_P_TOTAL, 
                         COL_LOG_ABS_C2_M1, COL_PHASE_C2_M1]
        # Zeta prime columns are optional as they are on a subset
        optional_prime_cols = [COL_ZETA_PRIME_REAL, COL_ZETA_PRIME_IMAG, 
                               COL_ARG_ZETA_PRIME, COL_ABS_ZETA_PRIME]
        
        missing_required = [col for col in required_cols if col not in df_zeros.columns]
        
        # Check if derivative columns need recomputing (e.g. if not present or mostly NaN)
        needs_derivative_recompute = False
        if any(col not in df_zeros.columns for col in optional_prime_cols) or \
           (COL_ZETA_PRIME_REAL in df_zeros.columns and df_zeros[COL_ZETA_PRIME_REAL].isna().sum() > len(df_zeros) * 0.95): # if >95% NaN
             needs_derivative_recompute = True


        if not missing_required and not needs_derivative_recompute:
            print("All required Phase 0 columns present. Skipping re-computation.")
            summary_text = f"Phase 0: Loaded pre-computed data from {PROCESSED_DATA_DIR / (phase0_output_filename + '.parquet')}.\n"
            summary_text += f"Processed {len(df_zeros)} zeros.\n"
            save_summary(summary_text, "phase0", "phase0_summary_loaded")
            end_time = time.time()
            print(f"Phase 0 (loaded) completed in {end_time - start_time:.2f} seconds.")
            return df_zeros
        else:
            print(f"Missing required columns: {missing_required} or derivative recompute needed. Re-running relevant parts of Phase 0.")
            # Fall through to re-computation, but start with the loaded df if it exists
            if df_zeros is None: # If file didn't exist at all
                 if not zeros_filepath.exists():
                    print(f"ERROR: Zeros file not found at {zeros_filepath}")
                    return None
                 df_zeros = pd.read_csv(zeros_filepath)
                 df_zeros.rename(columns={'ImaginaryPart': COL_T}, inplace=True)
                 df_zeros.sort_values(COL_T, inplace=True)
                 df_zeros.reset_index(drop=True, inplace=True)
                 print(f"Loaded {len(df_zeros)} zeros from raw file.")
    else: # df_zeros is None or force_recompute
        print(f"Loading zeros from {zeros_filepath}...")
        if not zeros_filepath.exists():
            print(f"ERROR: Zeros file not found at {zeros_filepath}")
            return None
        df_zeros = pd.read_csv(zeros_filepath)
        df_zeros.rename(columns={'ImaginaryPart': COL_T}, inplace=True)
        df_zeros.sort_values(COL_T, inplace=True)
        df_zeros.reset_index(drop=True, inplace=True)
        print(f"Loaded {len(df_zeros)} zeros from raw file.")
        needs_derivative_recompute = True # If loading raw, derivatives definitely need computing

    # Compute P_total and C2_m1 related columns if missing
    if COL_P_TOTAL not in df_zeros.columns:
        print("Computing P_total(t)...")
        df_zeros[COL_P_TOTAL] = compute_P_total_vec(df_zeros[COL_T].values, P_PRIMES, P_M_TERMS, LN_P_PRIMES)
    if COL_C2_M1 not in df_zeros.columns:
        print("Computing C2_m1(t)...")
        df_zeros[COL_C2_M1] = compute_C2_m1_vec(df_zeros[COL_T].values, LN_2)
    if COL_LOG_ABS_P_TOTAL not in df_zeros.columns:
        print("Computing log_abs_P_total...")
        df_zeros[COL_LOG_ABS_P_TOTAL] = compute_log_abs(df_zeros[COL_P_TOTAL].values, LOG_FLOOR_THRESHOLD, LOG_FLOOR_VALUE)
    if COL_LOG_ABS_C2_M1 not in df_zeros.columns:
        print("Computing log_abs_C2_m1...")
        df_zeros[COL_LOG_ABS_C2_M1] = compute_log_abs(df_zeros[COL_C2_M1].values, LOG_FLOOR_THRESHOLD, LOG_FLOOR_VALUE)
    if COL_PHASE_C2_M1 not in df_zeros.columns:
        print("Computing Phase_C2_m1...")
        df_zeros[COL_PHASE_C2_M1] = compute_phase_C2_m1_vec(df_zeros[COL_T].values, LN_2)

    # Derivative Computation
    if needs_derivative_recompute:
        print(f"Preparing subset for derivative computation (first {DERIVATIVE_SUBSET_FIRST_N}, random {DERIVATIVE_SUBSET_RANDOM_N})...")
        if len(df_zeros) < DERIVATIVE_SUBSET_FIRST_N: # Handle small datasets
            subset_indices = df_zeros.index
        elif len(df_zeros) < DERIVATIVE_SUBSET_FIRST_N + DERIVATIVE_SUBSET_RANDOM_N:
            first_n_indices = df_zeros.index[:DERIVATIVE_SUBSET_FIRST_N]
            remaining_indices = df_zeros.index[DERIVATIVE_SUBSET_FIRST_N:]
            if len(remaining_indices) > 0:
                 random_indices = np.random.choice(remaining_indices, size=min(DERIVATIVE_SUBSET_RANDOM_N, len(remaining_indices)), replace=False)
                 subset_indices = np.concatenate([first_n_indices, random_indices])
            else: # if only enough for first_n
                subset_indices = first_n_indices
        else: # Standard case
            first_n_indices = df_zeros.index[:DERIVATIVE_SUBSET_FIRST_N]
            remaining_indices = df_zeros.index[DERIVATIVE_SUBSET_FIRST_N:]
            np.random.seed(RANDOM_SEED)
            random_indices = np.random.choice(remaining_indices, size=DERIVATIVE_SUBSET_RANDOM_N, replace=False)
            subset_indices = np.concatenate([first_n_indices, random_indices])
        
        subset_indices = np.unique(subset_indices)
        
        # Initialize derivative columns if they don't exist
        for col in [COL_ZETA_PRIME_REAL, COL_ZETA_PRIME_IMAG, COL_ARG_ZETA_PRIME, COL_ABS_ZETA_PRIME]:
            if col not in df_zeros.columns:
                df_zeros[col] = np.nan
            else: # If they exist, NaN out the subset to be recomputed
                df_zeros.loc[subset_indices, col] = np.nan


        t_subset_values = df_zeros.loc[subset_indices, COL_T].values
        s_complex_tuples = [(original_idx, mpmath.mpc(0.5, t_val)) 
                            for original_idx, t_val in zip(subset_indices, t_subset_values)]

        print(f"Computing zeta derivatives for {len(s_complex_tuples)} points using multiprocessing...")
        num_processes = max(1, cpu_count() - 1)
        
        results_zeta_prime_indexed = []
        if s_complex_tuples: # only run if list is not empty
            with Pool(processes=num_processes) as pool:
                with tqdm(total=len(s_complex_tuples), desc="Zeta Prime Calc") as pbar:
                    for result in pool.imap_unordered(_compute_zeta_derivative_worker, s_complex_tuples):
                        results_zeta_prime_indexed.append(result)
                        pbar.update()
            
            for original_idx, zp_val in results_zeta_prime_indexed:
                if not mpmath.isnan(zp_val):
                    df_zeros.loc[original_idx, COL_ZETA_PRIME_REAL] = float(zp_val.real)
                    df_zeros.loc[original_idx, COL_ZETA_PRIME_IMAG] = float(zp_val.imag)
                    df_zeros.loc[original_idx, COL_ARG_ZETA_PRIME] = float(mpmath.arg(zp_val))
                    df_zeros.loc[original_idx, COL_ABS_ZETA_PRIME] = float(mpmath.fabs(zp_val))
        else:
            print("No points selected for derivative computation.")

    save_dataframe(df_zeros, PROCESSED_DATA_DIR, phase0_output_filename)

    summary_text = f"Phase 0: Foundational Data Preparation Complete.\n"
    summary_text += f"Processed {len(df_zeros)} zeros.\n"
    summary_text += f"P_total computed with {len(P_PRIMES)} primes (up to {P_PRIMES[-1]}) and M={P_M_TERMS} terms.\n"
    summary_text += f"Log floor for magnitudes: {LOG_FLOOR_VALUE} if abs(val) < {LOG_FLOOR_THRESHOLD}.\n"
    if 'subset_indices' in locals(): # check if derivative subset was processed
        summary_text += f"Zeta prime computed for {len(subset_indices)} zeros (DPS={MPMATH_DPS}).\n"
        summary_text += f"  - Non-NaN zeta_prime_real: {df_zeros[COL_ZETA_PRIME_REAL].notna().sum()}\n"
    else:
        summary_text += f"Zeta prime computation for subset potentially skipped or used existing data.\n"
        summary_text += f"  - Current Non-NaN zeta_prime_real: {df_zeros[COL_ZETA_PRIME_REAL].notna().sum() if COL_ZETA_PRIME_REAL in df_zeros.columns else 'N/A'}\n"

    summary_text += f"Augmented DataFrame saved to {PROCESSED_DATA_DIR / (phase0_output_filename + '.parquet')}.\n"
    save_summary(summary_text, "phase0", "phase0_summary")

    end_time = time.time()
    print(f"Phase 0 completed in {end_time - start_time:.2f} seconds.")
    return df_zeros


# --- Phase 1: Exploratory Analysis of P_total(t) and C2_m1(t) Relationships ---
def run_phase1(df_zeros_input, force_recompute_plots=False): # Added force_recompute_plots
    if df_zeros_input is None:
        print("Phase 0 data not loaded. Skipping Phase 1.")
        return None # Return None if input is None
    
    # Work on a copy to ensure original df passed from main is not modified by this phase
    # if it's passed to other phases directly from main.
    df_zeros = df_zeros_input.copy() 
    
    print("\n--- Running Phase 1: Exploratory Analysis of P_total(t) and C2_m1(t) Relationships ---")
    start_time = time.time()
    summary_phase1 = "Phase 1 Summary:\n\n"
    
    phase_name = "phase1"
    plot_dir = RESULTS_DIR / phase_name / "plots"
    ensure_dir(plot_dir) # Ensure plot directory exists

    # Columns needed for this phase (ensure they are present from Phase 0)
    phase1_required_cols = [COL_LOG_ABS_C2_M1, COL_LOG_ABS_P_TOTAL, COL_T, COL_P_TOTAL, COL_C2_M1]
    if not all(col in df_zeros.columns for col in phase1_required_cols):
        summary_phase1 += "Error: Not all required columns from Phase 0 are present. Aborting Phase 1.\n"
        print("Error in Phase 1: Missing required columns. Run Phase 0 again.")
        save_summary(summary_phase1, phase_name, "phase1_summary_error")
        return df_zeros # Return the (potentially incomplete) df

    # Create Ratio_P_C2m1 if it doesn't exist or needs recomputing for df_zeros copy
    if COL_RATIO_P_C2M1 not in df_zeros.columns:
        df_zeros[COL_RATIO_P_C2M1] = df_zeros[COL_P_TOTAL] / df_zeros[COL_C2_M1]
        df_zeros.loc[np.abs(df_zeros[COL_C2_M1]) < RATIO_C2M1_ZERO_THRESHOLD, COL_RATIO_P_C2M1] = np.nan

    # Create a random subset for scatter plots *after* all necessary columns are on df_zeros
    if len(df_zeros) > PLOT_RANDOM_SUBSET_SIZE_PHASE1:
        df_plot_subset = df_zeros.sample(n=PLOT_RANDOM_SUBSET_SIZE_PHASE1, random_state=RANDOM_SEED)
    else:
        df_plot_subset = df_zeros

    # 1. Primary Scatter Plot
    plot_name_scatter = "scatter_P_vs_C2"
    if force_recompute_plots or not (plot_dir / f"{plot_name_scatter}.png").exists():
        fig, ax = plt.subplots(figsize=(10, 8))
        sc = ax.scatter(df_plot_subset[COL_LOG_ABS_C2_M1], df_plot_subset[COL_LOG_ABS_P_TOTAL],
                        alpha=0.1, s=5, cmap='viridis', c=df_plot_subset[COL_T])
        ax.set_xlabel(f'{COL_LOG_ABS_C2_M1}')
        ax.set_ylabel(f'{COL_LOG_ABS_P_TOTAL}')
        ax.set_title(f'{COL_LOG_ABS_P_TOTAL} vs. {COL_LOG_ABS_C2_M1} at Zeros (Subset)')
        plt.colorbar(sc, label=COL_T)
        ax.grid(True)
        save_plot(fig, phase_name, plot_name_scatter)
    summary_phase1 += f"1. Generated primary scatter plot: {COL_LOG_ABS_P_TOTAL} vs. {COL_LOG_ABS_C2_M1}.\n"
    summary_phase1 += f"   Visual Analysis: Does a fish-tail like structure emerge? (Manual inspection needed)\n"

    # 2. Density/Contour Plot
    plot_name_density = "density_P_vs_C2"
    if force_recompute_plots or not (plot_dir / f"{plot_name_density}.png").exists():
        fig, ax = plt.subplots(figsize=(10, 8))
        hb = ax.hexbin(df_zeros[COL_LOG_ABS_C2_M1], df_zeros[COL_LOG_ABS_P_TOTAL],
                       gridsize=100, cmap='inferno', mincnt=1)
        ax.set_xlabel(f'{COL_LOG_ABS_C2_M1}')
        ax.set_ylabel(f'{COL_LOG_ABS_P_TOTAL}')
        ax.set_title(f'Density of {COL_LOG_ABS_P_TOTAL} vs. {COL_LOG_ABS_C2_M1} at Zeros')
        plt.colorbar(hb, label='Density (Counts)')
        ax.grid(True)
        save_plot(fig, phase_name, plot_name_density)
    summary_phase1 += f"2. Generated density plot (hexbin) of {COL_LOG_ABS_P_TOTAL} vs. {COL_LOG_ABS_C2_M1}.\n"

    # 3. Conditional Distributions
    plot_name_conditional = "conditional_dist_P_given_C2_bins"
    if force_recompute_plots or not (plot_dir / f"{plot_name_conditional}.png").exists():
        num_bins_conditional = 10
        try:
            # Ensure the binning column is created on the main df_zeros for consistency
            if COL_LOG_ABS_C2_M1_BIN not in df_zeros.columns:
                 df_zeros[COL_LOG_ABS_C2_M1_BIN] = pd.cut(df_zeros[COL_LOG_ABS_C2_M1], bins=num_bins_conditional)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            for bin_val, group in df_zeros.groupby(COL_LOG_ABS_C2_M1_BIN, observed=True):
                group[COL_LOG_ABS_P_TOTAL].plot(kind='hist', ax=ax, alpha=0.5, label=f'C2_m1 bin: {bin_val}', bins=50, density=True)
            ax.set_xlabel(f'{COL_LOG_ABS_P_TOTAL}')
            ax.set_ylabel('Density')
            ax.set_title(f'Conditional Distributions of {COL_LOG_ABS_P_TOTAL} given {COL_LOG_ABS_C2_M1} bins')
            ax.legend(title=f'{COL_LOG_ABS_C2_M1} Bins', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            save_plot(fig, phase_name, plot_name_conditional)
            summary_phase1 += f"3. Generated conditional histograms of {COL_LOG_ABS_P_TOTAL} for {num_bins_conditional} bins of {COL_LOG_ABS_C2_M1}.\n"
            summary_phase1 += f"   Visual Analysis: Do bimodal/multimodal distributions appear, suggesting branches? (Manual inspection needed)\n"
        except Exception as e:
            summary_phase1 += f"3. Error generating conditional distributions: {e}\n"
            print(f"Error in conditional distributions: {e}")
    else: # if plot exists
        summary_phase1 += f"3. Conditional histograms plot already exists. Skipped regeneration.\n"


    # 4. Ratio P_total(t) / C2_m1(t) Exploration (Ratio already computed on df_zeros)
    plot_name_hist_ratio = "hist_ratio_P_C2m1"
    if force_recompute_plots or not (plot_dir / f"{plot_name_hist_ratio}.png").exists():
        fig, ax = plt.subplots(figsize=(10, 6))
        df_zeros[COL_RATIO_P_C2M1].dropna().plot(kind='hist', bins=200, ax=ax, density=True,
                                              range=(-5, 5))
        ax.set_xlabel(f'Ratio {COL_P_TOTAL} / {COL_C2_M1}')
        ax.set_ylabel('Density')
        ax.set_title(f'Histogram of Ratio {COL_P_TOTAL} / {COL_C2_M1}')
        ax.grid(True)
        save_plot(fig, phase_name, plot_name_hist_ratio)
    summary_phase1 += f"4a. Generated histogram of {COL_RATIO_P_C2M1} ratio.\n"
    summary_phase1 += f"    Mean ratio: {df_zeros[COL_RATIO_P_C2M1].mean():.4f}, Median: {df_zeros[COL_RATIO_P_C2M1].median():.4f}\n"
    summary_phase1 += f"    Old test values: -0.29, 0.92. (Manual inspection needed for clustering)\n"

    # Scatter plot colored by ratio (uses df_plot_subset, which now should have Ratio_P_C2m1)
    plot_name_scatter_ratio = "scatter_P_vs_C2_color_by_ratio"
    if force_recompute_plots or not (plot_dir / f"{plot_name_scatter_ratio}.png").exists():
        fig, ax = plt.subplots(figsize=(10, 8))
        # df_plot_subset was created from df_zeros. If Ratio_P_C2m1 was added to df_zeros,
        # df_plot_subset needs to be recreated or have the column added.
        # To be safe, ensure df_plot_subset has the ratio column.
        if COL_RATIO_P_C2M1 not in df_plot_subset.columns and COL_RATIO_P_C2M1 in df_zeros.columns:
            df_plot_subset = df_zeros.sample(n=min(PLOT_RANDOM_SUBSET_SIZE_PHASE1, len(df_zeros)), random_state=RANDOM_SEED)


        if COL_RATIO_P_C2M1 in df_plot_subset.columns:
            ratio_clipped = np.clip(df_plot_subset[COL_RATIO_P_C2M1].fillna(0), -2, 2)
            sc_ratio = ax.scatter(df_plot_subset[COL_LOG_ABS_C2_M1], df_plot_subset[COL_LOG_ABS_P_TOTAL],
                                  alpha=0.1, s=5, c=ratio_clipped, cmap='coolwarm')
            ax.set_xlabel(f'{COL_LOG_ABS_C2_M1}')
            ax.set_ylabel(f'{COL_LOG_ABS_P_TOTAL}')
            ax.set_title(f'{COL_LOG_ABS_P_TOTAL} vs. {COL_LOG_ABS_C2_M1} (Colored by Ratio {COL_RATIO_P_C2M1})')
            plt.colorbar(sc_ratio, label=f'{COL_RATIO_P_C2M1} (clipped to [-2, 2])')
            ax.grid(True)
            save_plot(fig, phase_name, plot_name_scatter_ratio)
            summary_phase1 += f"4b. Generated scatter plot colored by {COL_RATIO_P_C2M1} ratio.\n"
        else:
            summary_phase1 += f"4b. Skipped scatter plot colored by ratio as '{COL_RATIO_P_C2M1}' column missing from subset.\n"
            print(f"Warning: '{COL_RATIO_P_C2M1}' column missing from df_plot_subset for ratio-colored scatter plot.")


    # 5. Value of P_total(t) at Zeros
    plot_name_hist_P_total = "hist_P_total_at_zeros"
    if force_recompute_plots or not (plot_dir / f"{plot_name_hist_P_total}.png").exists():
        fig, ax = plt.subplots(figsize=(10, 6))
        df_zeros[COL_P_TOTAL].plot(kind='hist', bins=200, ax=ax, density=True,
                                 range=(-2,2))
        ax.set_xlabel(f'{COL_P_TOTAL} at Zeros')
        ax.set_ylabel('Density')
        ax.set_title(f'Histogram of {COL_P_TOTAL} at Zeros')
        ax.grid(True)
        save_plot(fig, phase_name, plot_name_hist_P_total)
    summary_phase1 += f"5. Generated histogram of {COL_P_TOTAL} at zeros.\n"
    summary_phase1 += f"   Mean P_total: {df_zeros[COL_P_TOTAL].mean():.4f}, Median: {df_zeros[COL_P_TOTAL].median():.4f}\n"
    summary_phase1 += f"   Old test value for |P(t)|: ~0.458 (log10 value ~-0.338775). (Manual inspection)\n"

    save_summary(summary_phase1, phase_name, "phase1_summary")
    end_time = time.time()
    print(f"Phase 1 completed in {end_time - start_time:.2f} seconds.")
    return df_zeros # Return the df with potentially new 'Ratio_P_C2m1' and bin column

# --- Phase 2: Deriving Structure - If Branches Emerge --- (Still mostly placeholder)
def run_phase2(df_zeros_input, force_recompute_plots=False):
    if df_zeros_input is None: return None
    df_zeros = df_zeros_input.copy()
    print("\n--- Running Phase 2: Deriving Structure - If Branches Emerge ---")
    start_time = time.time()
    summary_phase2 = "Phase 2 Summary (Exploratory - Manual Inspection Focus):\n\n"
    phase_name = "phase2"
    
    summary_phase2 += "1. Branch Separation: Manual inspection of Phase 1 plots is required.\n"
    summary_phase2 += "2. Curve Fitting: Skipped as branches are not yet algorithmically derived.\n"
    summary_phase2 += "3. Analyze Intersection of Derived Branches: Skipped.\n"
    
    save_summary(summary_phase2, phase_name, "phase2_summary")
    end_time = time.time()
    print(f"Phase 2 completed (manual inspection focus) in {end_time - start_time:.2f} seconds.")
    return df_zeros 

# --- Phase 3: Phase Property Exploration ---
def min_angular_dist_to_set_vec(angles, target_angles_arr): # Vectorized
    angles_col = angles[:, np.newaxis]
    diffs = np.abs(angles_col - target_angles_arr)
    diffs_plus_2pi = np.abs(angles_col - (target_angles_arr + 2*np.pi))
    diffs_minus_2pi = np.abs(angles_col - (target_angles_arr - 2*np.pi))
    all_diffs = np.concatenate([diffs, diffs_plus_2pi, diffs_minus_2pi], axis=1)
    return np.min(all_diffs, axis=1)


def run_phase3(df_zeros_input, force_recompute_plots=False):
    if df_zeros_input is None: return None
    df_zeros = df_zeros_input.copy()
    print("\n--- Running Phase 3: Phase Property Exploration ---")
    start_time = time.time()
    summary_phase3 = "Phase 3 Summary:\n\n"
    phase_name = "phase3"
    plot_dir = RESULTS_DIR / phase_name / "plots"
    ensure_dir(plot_dir)

    phase3_required_cols = [COL_PHASE_C2_M1]
    if not all(col in df_zeros.columns for col in phase3_required_cols):
        summary_phase3 += "Error: Not all required columns from Phase 0 are present for Phase 3. Aborting.\n"
        print("Error in Phase 3: Missing required columns.")
        save_summary(summary_phase3, phase_name, "phase3_summary_error")
        return df_zeros

    # 1. Distribution of Phase_C2_m1
    plot_name_hist_phase = "hist_Phase_C2_m1"
    if force_recompute_plots or not (plot_dir / f"{plot_name_hist_phase}.png").exists():
        fig, ax = plt.subplots(figsize=(10, 6))
        df_zeros[COL_PHASE_C2_M1].plot(kind='hist', bins=100, ax=ax, density=True)
        ax.set_xlabel(f'{COL_PHASE_C2_M1} (radians, [0, 2π))')
        ax.set_ylabel('Density')
        ax.set_title(f'Histogram of {COL_PHASE_C2_M1} at Zeros')
        ax.set_xticks(np.arange(0, 2*np.pi + 0.01, np.pi/2))
        ax.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
        ax.grid(True)
        save_plot(fig, phase_name, plot_name_hist_phase)
    summary_phase3 += f"1. Generated histogram of {COL_PHASE_C2_M1} at zeros.\n"
    summary_phase3 += f"   Visual Analysis: Strong clustering near 0 and π? (Manual inspection)\n"

    # 2. Define Phase_Dist_0_pi
    if COL_PHASE_DIST_0_PI not in df_zeros.columns or force_recompute_plots: # force_recompute_plots implies recalculate
        target_phases_for_dist = np.array([0, np.pi])
        df_zeros[COL_PHASE_DIST_0_PI] = min_angular_dist_to_set_vec(df_zeros[COL_PHASE_C2_M1].values, target_phases_for_dist)
    
    plot_name_hist_phase_dist = "hist_Phase_Dist_0_pi"
    if force_recompute_plots or not (plot_dir / f"{plot_name_hist_phase_dist}.png").exists():
        fig, ax = plt.subplots(figsize=(10, 6))
        df_zeros[COL_PHASE_DIST_0_PI].plot(kind='hist', bins=100, ax=ax, density=True)
        ax.set_xlabel(f'{COL_PHASE_DIST_0_PI} (Min angular distance to 0 or π)')
        ax.set_ylabel('Density')
        ax.set_title(f'Histogram of {COL_PHASE_DIST_0_PI}')
        ax.grid(True)
        save_plot(fig, phase_name, plot_name_hist_phase_dist)
    summary_phase3 += f"2. Calculated and plotted histogram of {COL_PHASE_DIST_0_PI}.\n"

    # 3. "Mod-N" Phase State Derivation (Testing old Mod-4)
    if COL_MOD4_INDEX_C2M1 not in df_zeros.columns or force_recompute_plots:
        df_zeros[COL_MOD4_INDEX_C2M1] = np.floor(df_zeros[COL_PHASE_C2_M1] / (np.pi / 2.0)).astype(int)
        df_zeros.loc[df_zeros[COL_MOD4_INDEX_C2M1] == 4, COL_MOD4_INDEX_C2M1] = 0
    
    plot_name_bar_mod4 = "bar_Mod4_Index_C2m1_dist"
    if force_recompute_plots or not (plot_dir / f"{plot_name_bar_mod4}.png").exists():
        fig, ax = plt.subplots(figsize=(8, 5))
        df_zeros[COL_MOD4_INDEX_C2M1].value_counts(normalize=True).sort_index().plot(kind='bar', ax=ax)
        ax.set_xlabel(f'{COL_MOD4_INDEX_C2M1} (0, 1, 2, 3)')
        ax.set_ylabel('Proportion')
        ax.set_title(f'Distribution of {COL_MOD4_INDEX_C2M1}')
        ax.grid(axis='y')
        save_plot(fig, phase_name, plot_name_bar_mod4)
    summary_phase3 += f"3. Calculated and plotted distribution of {COL_MOD4_INDEX_C2M1} (old definition).\n"

    # 4. Correlate Phase with Derived Branches (Still placeholder)
    summary_phase3 += "4. Correlation of Phase with Derived Branches: Skipped as no 'derived_branch' column exists.\n"

    # 5. Correlation of Phase_Dist_0_pi with arg(zeta_prime)
    df_deriv_subset = df_zeros[df_zeros[COL_ARG_ZETA_PRIME].notna()].copy() # Ensure it's a copy for new col
    if not df_deriv_subset.empty:
        # Calculate abs_arg_zeta_prime_mod_pi for the subset if not already present or forced
        if COL_ABS_ARG_ZETA_PRIME_MOD_PI not in df_deriv_subset.columns or force_recompute_plots:
             df_deriv_subset[COL_ABS_ARG_ZETA_PRIME_MOD_PI] = np.abs(df_deriv_subset[COL_ARG_ZETA_PRIME] % np.pi)
        
        plot_name_scatter_phase_arg = "scatter_phase_dist_vs_arg_zeta_prime"
        if force_recompute_plots or not (plot_dir / f"{plot_name_scatter_phase_arg}.png").exists():
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(df_deriv_subset[COL_ABS_ARG_ZETA_PRIME_MOD_PI], df_deriv_subset[COL_PHASE_DIST_0_PI], alpha=0.2, s=10)
            ax.set_xlabel(f'|{COL_ARG_ZETA_PRIME} % π|')
            ax.set_ylabel(f'{COL_PHASE_DIST_0_PI}')
            ax.set_title(f'{COL_PHASE_DIST_0_PI} vs. |{COL_ARG_ZETA_PRIME} % π|')
            ax.grid(True)
            save_plot(fig, phase_name, plot_name_scatter_phase_arg)
        summary_phase3 += f"5a. Generated scatter plot: {COL_PHASE_DIST_0_PI} vs. |{COL_ARG_ZETA_PRIME} % π| (for derivative subset).\n"

        try:
            from scipy.stats import linregress
            # Ensure no NaNs in data used for regression
            clean_arg_data = df_deriv_subset[COL_ABS_ARG_ZETA_PRIME_MOD_PI].dropna()
            clean_phase_dist_data = df_deriv_subset[COL_PHASE_DIST_0_PI].loc[clean_arg_data.index] # Align indices

            if len(clean_arg_data) > 1 and len(clean_phase_dist_data) > 1 :
                slope, intercept, r_value, p_value, std_err = linregress(clean_arg_data, clean_phase_dist_data)
                summary_phase3 += f"5b. Linear fit for {COL_PHASE_DIST_0_PI} vs. |{COL_ARG_ZETA_PRIME} % π|:\n"
                summary_phase3 += f"     Slope: {slope:.4f}, Intercept: {intercept:.4f}, R-squared: {r_value**2:.4f}\n"
            else:
                summary_phase3 += f"5b. Not enough non-NaN data points for linear regression of phase_dist vs arg_zeta_prime.\n"
        except Exception as e:
            summary_phase3 += f"5b. Error during linear fit for phase_dist vs arg_zeta_prime: {e}\n"
            print(f"Error in phase_dist vs arg_zeta_prime linear fit: {e}")
    else:
        summary_phase3 += "5. Correlation with arg(zeta_prime): Skipped as no derivative data is available.\n"

    save_summary(summary_phase3, phase_name, "phase3_summary")
    end_time = time.time()
    print(f"Phase 3 completed in {end_time - start_time:.2f} seconds.")
    return df_zeros

# --- Phase 4: Spectral Analysis of Multi-Prime P_total(t) --- (Mostly unchanged logic)
def run_phase4(df_zeros_input, force_recompute_plots=False):
    if df_zeros_input is None: return None
    df_zeros = df_zeros_input.copy()
    print("\n--- Running Phase 4: Spectral Analysis of Multi-Prime P_total(t) ---")
    start_time = time.time()
    summary_phase4 = "Phase 4 Summary:\n\n"
    phase_name = "phase4"
    plot_dir = RESULTS_DIR / phase_name / "plots"
    ensure_dir(plot_dir)
    
    plot_name_fft = "fft_P_total_continuous"
    plot_name_fft_peaks = "fft_P_total_continuous_with_peaks"

    if force_recompute_plots or not (plot_dir / f"{plot_name_fft_peaks}.png").exists():
        if len(df_zeros) >= CONTINUOUS_T_MAX_ZERO_INDEX:
            t_max_for_signal = df_zeros.iloc[CONTINUOUS_T_MAX_ZERO_INDEX -1][COL_T]
        elif not df_zeros.empty:
            t_max_for_signal = df_zeros[COL_T].max()
        else:
            print("No zeros data for Phase 4 continuous signal. Skipping FFT.")
            summary_phase4 += "No zeros data available to determine t_max for continuous signal. FFT skipped.\n"
            save_summary(summary_phase4, phase_name, "phase4_summary")
            return df_zeros

        t_continuous = np.arange(0, t_max_for_signal + CONTINUOUS_T_DELTA, CONTINUOUS_T_DELTA)
        print(f"Generating continuous P_total(t) signal for t in [0, {t_max_for_signal:.2f}] with dt={CONTINUOUS_T_DELTA} ({len(t_continuous)} points)...")
        P_total_continuous = compute_P_total_vec(t_continuous, P_PRIMES, P_M_TERMS, LN_P_PRIMES)

        P_signal_mean_subtracted = P_total_continuous - np.mean(P_total_continuous)
        P_signal_for_fft = P_signal_mean_subtracted

        N = len(P_signal_for_fft)
        if N <=1:
            summary_phase4 += "Not enough points for FFT. Skipping.\n"
            print("Not enough points for FFT in Phase 4. Skipping.")
            save_summary(summary_phase4, phase_name, "phase4_summary")
            return df_zeros

        if FFT_WINDOW_TYPE:
            window_func = windows.get_window(FFT_WINDOW_TYPE, N)
            P_signal_for_fft = P_signal_for_fft * window_func
            print(f"Applied {FFT_WINDOW_TYPE} window to signal for FFT.")

        yf = rfft(P_signal_for_fft)
        xf = rfftfreq(N, d=CONTINUOUS_T_DELTA)
        power_spectrum = np.abs(yf)**2
        
        valid_indices = xf > 1e-9 # Avoid true zero frequency for period calculation
        frequencies_hz = xf[valid_indices]
        power = power_spectrum[valid_indices]
        
        periods = np.zeros_like(frequencies_hz)
        non_zero_freq_mask = frequencies_hz > 1e-9 # Stricter check for period calculation
        if np.any(non_zero_freq_mask):
            periods[non_zero_freq_mask] = 1.0 / frequencies_hz[non_zero_freq_mask]
        
        power_for_periods = power # Same length as periods now

        fig, ax = plt.subplots(2, 1, figsize=(12, 10))
        ax[0].plot(frequencies_hz, power)
        ax[0].set_xlabel('Frequency (Hz)')
        ax[0].set_ylabel('Power')
        ax[0].set_title(f'Power Spectrum of Continuous P_total(t) (Window: {FFT_WINDOW_TYPE})')
        ax[0].grid(True)
        ax[0].set_xlim(0, max(frequencies_hz)/10 if len(frequencies_hz)>0 and max(frequencies_hz)>0 else 1)

        period_plot_mask = (periods <= 30) & (periods > 0) # Ensure periods are positive
        ax[1].plot(periods[period_plot_mask], power_for_periods[period_plot_mask])
        ax[1].set_xlabel('Period (units of t)')
        ax[1].set_ylabel('Power')
        ax[1].set_title('Power Spectrum vs. Period (Periods <= 30)')
        ax[1].grid(True)

        plt.tight_layout()
        # Save initial FFT plot without peaks first, then add peaks and save again
        save_plot(fig, phase_name, plot_name_fft) 
        summary_phase4 += f"1 & 2. Generated FFT of continuous P_total(t) (t up to {t_max_for_signal:.2f}, dt={CONTINUOUS_T_DELTA}).\n"
        summary_phase4 += f"   Window function used: {FFT_WINDOW_TYPE}\n"

        if len(periods[period_plot_mask]) > 0:
            from scipy.signal import find_peaks
            # Use power_for_periods that corresponds to the plotted periods
            plotted_power = power_for_periods[period_plot_mask]
            plotted_periods = periods[period_plot_mask]

            if len(plotted_power) > 0: # Check if there's anything to find peaks in
                peaks_indices_in_plotted, _ = find_peaks(plotted_power, prominence=np.max(plotted_power)/20 if np.max(plotted_power) > 0 else None, distance=10)
                
                dominant_periods_found = plotted_periods[peaks_indices_in_plotted]
                dominant_powers_found = plotted_power[peaks_indices_in_plotted]
                
                sorted_peak_indices = np.argsort(dominant_powers_found)[::-1]
                summary_phase4 += "   Dominant Periods (sorted by power, for periods <= 30):\n"
                for i in range(min(5, len(sorted_peak_indices))):
                    idx_in_plotted = sorted_peak_indices[i]
                    original_peak_idx = peaks_indices_in_plotted[idx_in_plotted]
                    
                    # To get frequency, need to map back to original frequencies_hz
                    # This mapping is tricky if period_plot_mask is not contiguous.
                    # For simplicity, we'll just report period and power.
                    summary_phase4 += f"     - Period: {plotted_periods[original_peak_idx]:.4f} (Power: {plotted_power[original_peak_idx]:.2e})\n"
                    ax[1].axvline(plotted_periods[original_peak_idx], color='r', linestyle='--', alpha=0.7)
            else:
                summary_phase4 += "   No power data in the plotted period range (<=30) to find peaks.\n"


            theoretical_periods_p = (2 * np.pi) / LN_P_PRIMES
            summary_phase4 += "   Theoretical periods (2π/ln p) for P_PRIMES:\n"
            for p_val, th_period in zip(P_PRIMES, theoretical_periods_p):
                summary_phase4 += f"     - p={int(p_val)}: {th_period:.4f}\n"
                if th_period <= 30: # Only plot if in range
                    ax[1].axvline(th_period, color='g', linestyle=':', alpha=0.7)
            
            save_plot(fig, phase_name, plot_name_fft_peaks) # Save plot with peaks
        else:
            summary_phase4 += "   No valid periods found for peak analysis or plotting FFT vs period.\n"
            plt.close(fig) # Close the figure if not saved with peaks

    else: # if plot exists
        summary_phase4 += f"FFT plot {plot_name_fft_peaks}.png already exists. Skipped regeneration.\n"


    summary_phase4 += "   Visual Analysis: How do dominant periods align with 2π/ln p? Are other anomalous periods present?\n"
    save_summary(summary_phase4, phase_name, "phase4_summary")
    end_time = time.time()
    print(f"Phase 4 completed in {end_time - start_time:.2f} seconds.")
    return df_zeros


# --- Phase 5: Sequential Analysis (Dynamical Properties) ---
def run_phase5(df_zeros_input, force_recompute_plots=False):
    if df_zeros_input is None: return None
    df_zeros = df_zeros_input.copy()
    print("\n--- Running Phase 5: Sequential Analysis (Dynamical Properties) ---")
    start_time = time.time()
    summary_phase5 = "Phase 5 Summary (Exploratory):\n\n"
    phase_name = "phase5"
    plot_dir = RESULTS_DIR / phase_name / "plots"
    ensure_dir(plot_dir)

    # 1. Track Changes
    if COL_MOD4_INDEX_C2M1 in df_zeros.columns:
        if COL_DELTA_MOD4_INDEX not in df_zeros.columns or force_recompute_plots:
            df_zeros[COL_DELTA_MOD4_INDEX] = df_zeros[COL_MOD4_INDEX_C2M1].diff().fillna(0)
        summary_phase5 += f"1. Calculated {COL_DELTA_MOD4_INDEX} for consecutive zeros.\n"
    else:
        summary_phase5 += f"1. Tracking changes: Skipped as {COL_MOD4_INDEX_C2M1} missing.\n"

    # 2. Visualize Trajectories
    plot_name_traj = "3D_trajectory_example"
    required_traj_cols = [COL_LOG_ABS_C2_M1, COL_LOG_ABS_P_TOTAL, COL_PHASE_C2_M1, COL_T]
    if all(col in df_zeros.columns for col in required_traj_cols):
        if force_recompute_plots or not (plot_dir / f"{plot_name_traj}.png").exists():
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            df_traj_subset = df_zeros.head(1000)

            sc = ax.scatter(df_traj_subset[COL_LOG_ABS_C2_M1], 
                            df_traj_subset[COL_LOG_ABS_P_TOTAL], 
                            df_traj_subset[COL_PHASE_C2_M1], 
                            c=df_traj_subset[COL_T], cmap='viridis', s=10)
            
            ax.plot(df_traj_subset[COL_LOG_ABS_C2_M1], 
                    df_traj_subset[COL_LOG_ABS_P_TOTAL], 
                    df_traj_subset[COL_PHASE_C2_M1], 
                    color='gray', alpha=0.3, linestyle='-')

            ax.set_xlabel(f'{COL_LOG_ABS_C2_M1}')
            ax.set_ylabel(f'{COL_LOG_ABS_P_TOTAL}')
            ax.set_zlabel(f'{COL_PHASE_C2_M1} (radians)')
            ax.set_title('3D Trajectory of Zeros (First 1000)')
            plt.colorbar(sc, label=COL_T)
            ax.view_init(elev=20, azim=-60)
            save_plot(fig, phase_name, plot_name_traj)
        summary_phase5 += "2. Generated example 3D trajectory plot for first 1000 zeros.\n"
    else:
        summary_phase5 += "2. Visualize Trajectories: Skipped as prerequisite columns missing.\n"
        
    save_summary(summary_phase5, phase_name, "phase5_summary")
    end_time = time.time()
    print(f"Phase 5 completed in {end_time - start_time:.2f} seconds.")
    return df_zeros


# --- Main Execution ---
if __name__ == "__main__":
    plt.style.use(PLT_STYLE)
    # Create base directories
    ensure_dir(PROCESSED_DATA_DIR)
    ensure_dir(RESULTS_DIR)
    
    # Allow forcing recomputation of all plots for a specific phase or all phases
    # For simplicity here, we'll add a general force_recompute_all_plots flag.
    # More granular control can be added per phase if needed.
    FORCE_RECOMPUTE_ALL_PLOTS = False # Set to True to regenerate all plots
    FORCE_RECOMPUTE_PHASE0_DATA = False # Set to True to re-run all Phase 0 data calcs

    df_augmented_zeros = run_phase0(ZEROS_FILEPATH, force_recompute=FORCE_RECOMPUTE_PHASE0_DATA)

    if df_augmented_zeros is not None:
        df_p1 = run_phase1(df_augmented_zeros, force_recompute_plots=FORCE_RECOMPUTE_ALL_PLOTS)
        df_p2 = run_phase2(df_p1 if df_p1 is not None else df_augmented_zeros, 
                           force_recompute_plots=FORCE_RECOMPUTE_ALL_PLOTS) # Phase 2 is mostly placeholder
        df_p3 = run_phase3(df_p2 if df_p2 is not None else df_augmented_zeros, 
                           force_recompute_plots=FORCE_RECOMPUTE_ALL_PLOTS)
        df_p4 = run_phase4(df_augmented_zeros, # Phase 4 typically doesn't depend on p1-p3 column changes
                           force_recompute_plots=FORCE_RECOMPUTE_ALL_PLOTS)
        df_p5 = run_phase5(df_p3 if df_p3 is not None else df_augmented_zeros, # Phase 5 might use Mod4 from p3
                           force_recompute_plots=FORCE_RECOMPUTE_ALL_PLOTS)
        
        print("\n--- All Phases Attempted ---")
        print(f"Results saved in: {RESULTS_DIR}")
        print(f"Processed data (e.g., augmented DataFrame) saved in: {PROCESSED_DATA_DIR}")
    else:
        print("Critical error in Phase 0 or data loading. Halting further analysis.")