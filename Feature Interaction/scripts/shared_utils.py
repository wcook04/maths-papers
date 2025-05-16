# Paper_12/scripts/shared_utils.py

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # Keep for save_plot, though not directly used in main
import re
import time
import mpmath # For Zeta Prime
from multiprocessing import Pool, cpu_count, freeze_support # For Zeta Prime, freeze_support for PyInstaller
from tqdm import tqdm # For progress bars
import argparse # For command-line arguments

# --- Setup Paths for Sibling Imports (constants.py) ---
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# --- Import from Local constants.py ---
try:
    from constants import (
        BASE_DIR, DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, # Dirs
        BRANCH_PI_HALF, BRANCH_3PI_HALF, BRANCH_TRANSITION_OTHER, ALL_BRANCHES, # Branches
        COL_T, COL_PHASE_BRANCH, COL_P_TOTAL, COL_LOG_ABS_P_TOTAL, # Core Cols
        COL_RATIO_P_C2M1, RATIO_C2M1_ZERO_THRESHOLD, # Ratio Cols
        COL_ZETA_PRIME_REAL, COL_ZETA_PRIME_IMAG, COL_ARG_ZETA_PRIME, # Zeta Cols
        COL_ABS_ZETA_PRIME, COL_ABS_ARG_ZETA_PRIME_MOD_PI,
        COL_C2_BREATH_SIGN_NUMERICAL, C2_BREATH_SIGN_ZERO_THRESHOLD, # C2 Breath
        PRIMES_TO_PROCESS, P_PRIMES_FOR_P_TOTAL, P_M_TERMS_FOR_P_TOTAL, # Primes
        LN_P_PRIMES_FOR_P_TOTAL, LN_2,
        PRIME_FEATURE_NAMES, INTERACTION_PAIRS_TO_PROCESS, PRIME_INTERACTION_FEATURE_NAMES, # Dynamic Names
        LOG_FLOOR_VALUE, LOG_FLOOR_THRESHOLD, DOT_LOG_LOOKBACK_GENERAL, # Calc Params
        ALPHA_DENOM_EPSILON_GENERAL, IMPULSE_ZERO_THRESHOLD_GENERAL,
        MPMATH_DPS, DERIVATIVE_SUBSET_FIRST_N, DERIVATIVE_SUBSET_RANDOM_N, RANDOM_SEED, # More Calc Params
        PHASE_PI_HALF_LOWER_GENERAL, PHASE_PI_HALF_UPPER_GENERAL, # Phase Bounds
        PHASE_3PI_HALF_LOWER_GENERAL, PHASE_3PI_HALF_UPPER_GENERAL,
        DELTA_PHI_CORR_PI_HALF_LOWER_GENERAL, DELTA_PHI_CORR_PI_HALF_UPPER_GENERAL,
        DELTA_PHI_CORR_3PI_HALF_LOWER_GENERAL, DELTA_PHI_CORR_3PI_HALF_UPPER_GENERAL,
        RAW_ZEROS_FILENAME_STEM, MASTER_DF_FILENAME_STEM, # File Stems
        DEFAULT_DPI, PLT_STYLE # Plotting
        # Make sure all other previously listed constants are also present here
    )
    # print("SUCCESS: shared_utils.py successfully imported from constants.py") # Verified
except ImportError as e:
    print(f"FATAL ERROR: shared_utils.py could not import from constants.py. Error: {e}")
    sys.exit(1)

# Set mpmath precision globally, although workers reset it
mpmath.mp.dps = MPMATH_DPS

# --- I. Utility Functions ---
def ensure_dir(path: Path):
    """Ensures the directory exists, creating it if necessary."""
    path.mkdir(parents=True, exist_ok=True)

def sanitize_filename(filename_stem: str) -> str:
    """Sanitizes a string for use as a filename stem."""
    if not isinstance(filename_stem, str): filename_stem = str(filename_stem)
    name = filename_stem.strip().replace(' ', '_')
    name = re.sub(r'(?u)[^-\w.]', '_', name) # Keep alphanumeric, underscore, hyphen, dot
    name = name.strip('_-.') # Remove leading/trailing _-.
    return name if name else "sanitized_empty_name"

def save_plot(fig, plot_name_stem: str, output_dir: Path, tight_layout: bool = True, dpi: int = DEFAULT_DPI):
    """Saves a matplotlib figure to a file and closes it."""
    ensure_dir(output_dir)
    sanitized_name = sanitize_filename(plot_name_stem)
    filepath = output_dir / f"{sanitized_name}.png"
    if tight_layout:
        try: fig.tight_layout()
        except Exception: pass # Ignore layout errors
    try:
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        # print(f"Saved plot: {filepath}") # Can be verbose
    except Exception as e:
        print(f"ERROR: Could not save plot {filepath}. Error: {e}")
        plt.close(fig)

def save_summary_report(report_lines: list, summary_name_stem: str, output_dir: Path):
    """Saves a list of strings as a text report."""
    ensure_dir(output_dir)
    sanitized_name = sanitize_filename(summary_name_stem)
    filepath = output_dir / f"{sanitized_name}.txt"
    try:
        with open(filepath, "w", encoding='utf-8') as f:
            for line in report_lines: f.write(line + "\n")
        # print(f"Saved summary report: {filepath}") # Can be verbose
    except Exception as e:
        print(f"ERROR: Could not save summary report {filepath}. Error: {e}")

def load_dataframe_robust(filepath: Path, report_list: list = None) -> pd.DataFrame | None:
    """Loads a DataFrame from a Parquet file with error handling."""
    if report_list is None: report_list = []
    if filepath.exists() and filepath.suffix == '.parquet':
        try:
            df = pd.read_parquet(filepath)
            msg = f"SUCCESS: Loaded DataFrame from {filepath} (Shape: {df.shape})."
            report_list.append(msg); print(msg)
            return df
        except Exception as e:
            msg = f"ERROR: Failed to load Parquet file {filepath}. Error: {e}"
            report_list.append(msg); print(msg); return None
    else:
        # Changed to INFO level as it's expected not to exist on first run
        msg = f"INFO: File not found or not a Parquet file: {filepath}"
        report_list.append(msg); print(msg); return None

def save_dataframe_robust(df: pd.DataFrame, filepath: Path, report_list: list = None):
    """Saves a DataFrame to a Parquet file with error and duplicate column handling."""
    if report_list is None: report_list = []
    ensure_dir(filepath.parent)
    try:
        df_to_save = df.copy() # Work on a copy to avoid modifying original DF unexpectedly
        if df_to_save.columns.has_duplicates:
            dup_cols = df_to_save.columns[df_to_save.columns.duplicated(keep=False)].tolist()
            msg_warn = (f"WARNING: DataFrame for {filepath} has duplicate columns: {dup_cols}. De-duplicating (keep='first').")
            report_list.append(msg_warn); print(msg_warn)
            df_to_save = df_to_save.loc[:, ~df_to_save.columns.duplicated(keep='first')]
            # Re-check just in case something went wrong (unlikely with loc)
            if df_to_save.columns.has_duplicates:
                 msg_err = (f"ERROR: De-duplication failed for {filepath}. DataFrame NOT saved.")
                 report_list.append(msg_err); print(msg_err); return # Prevent saving bad DF
        df_to_save.to_parquet(filepath, index=False)
        msg_succ = f"SUCCESS: Saved DataFrame to {filepath} (Shape: {df_to_save.shape})."
        report_list.append(msg_succ); print(msg_succ)
    except Exception as e:
        msg_err_save = f"ERROR: Failed to save DataFrame to {filepath}. Error: {e}"
        report_list.append(msg_err_save); print(msg_err_save)


def format_table_to_string(header: list, data_rows: list, title: str = "") -> str:
    """Formats data into a simple string table."""
    if not data_rows and not header: return f"{title}\n(Table is empty)\n"
    if not header and data_rows:
        if data_rows and data_rows[0]: header = [f"Col_{i+1}" for i in range(len(data_rows[0]))]
        else: return f"{title}\n(No data in rows for header inference)\n"
    elif not header: return f"{title}\n(Header is empty)\n"
    num_cols = len(header)
    if not data_rows:
        col_widths = [len(str(h)) for h in header]
        header_fmt = " | ".join([f"{{:<{w}}}" for w in col_widths])
        separator = "-+-".join(["-" * w for w in col_widths])
        table_str_list = [f"{title}"] if title else []
        table_str_list.append(header_fmt.format(*header))
        table_str_list.append(separator); table_str_list.append("(No data rows)")
        return "\n".join(table_str_list) + "\n"
    col_widths = [len(str(h)) for h in header]
    for row in data_rows:
        for i, cell in enumerate(row):
            # Ensure cell can be stringified and handle rows shorter/longer than header
            if i < num_cols: col_widths[i] = max(col_widths[i], len(str(cell)))
            else: break # Row has more cells than header, ignore extra
    header_fmt = " | ".join([f"{{:<{w}}}" for w in col_widths])
    row_fmt = " | ".join([f"{{:<{w}}}" for w in col_widths])
    separator = "-+-".join(["-" * w for w in col_widths])
    table_str_list = [f"{title}"] if title else []
    table_str_list.append(header_fmt.format(*header))
    table_str_list.append(separator)
    for row in data_rows:
        # Pad row with empty strings if it's shorter than header
        padded_row = list(row[:num_cols]) + [''] * (num_cols - len(row)) if len(row) < num_cols else row[:num_cols]
        table_str_list.append(row_fmt.format(*padded_row))
    return "\n".join(table_str_list) + "\n"

# --- II. Calculation Helper Functions ---
def compute_log_abs_with_floor(values: pd.Series, threshold: float, floor_val: float) -> pd.Series:
    """Computes log10(abs(values)), flooring results below a threshold to a specified value."""
    if not isinstance(values, pd.Series): raise TypeError("Input 'values' must be a pandas Series.")
    abs_values = np.abs(values.to_numpy(dtype=float))
    log_abs_values = np.full_like(abs_values, np.nan, dtype=float) # Initialize with NaNs

    # Mask for values >= threshold (where we actually compute log10)
    calc_mask = ~np.isnan(abs_values) & (abs_values >= threshold)
    # Mask for values < threshold (where we assign floor_val)
    floor_mask = ~np.isnan(abs_values) & (abs_values < threshold)

    if np.any(calc_mask):
        # Handle values == 0 explicitly within the calc_mask range if needed, assign floor_val
        # This might happen if threshold is 0, but the logic below handles abs_values == 0 fine.
        # Ensure log10 is only called on positive values.
        safe_log_mask = calc_mask & (abs_values > 0)
        if np.any(safe_log_mask):
             log_abs_values[safe_log_mask] = np.log10(abs_values[safe_log_mask])
        # If threshold is > 0, and abs_values[calc_mask] == 0, this case shouldn't happen,
        # but if threshold is 0, abs_values == 0 is in calc_mask. log10(0) is -inf.
        # We map 0 to floor_val *before* log10 if needed, or after.
        # Current logic: if abs_values[safe_log_mask] is 0, np.log10(0) is -inf, which is <= floor_val.
        # So the floor_mask logic catches it. Let's just ensure 0 is handled.
        zero_at_threshold_mask = calc_mask & (abs_values == 0)
        if np.any(zero_at_threshold_mask):
            log_abs_values[zero_at_threshold_mask] = floor_val # Map exact zero to floor

    if np.any(floor_mask):
        log_abs_values[floor_mask] = floor_val # Apply floor value for small positive values

    # Any remaining NaNs from the original input will stay NaN
    return pd.Series(log_abs_values, index=values.index, name=values.name if values.name else None)


def _compute_zeta_derivative_worker_shared(s_complex_tuple: tuple) -> tuple:
    """Worker function for multiprocessing Zeta derivative calculation."""
    original_index, s_complex = s_complex_tuple
    # Import mpmath inside the worker function to potentially help with multiprocessing start methods
    import mpmath as mp_worker_zeta
    mp_worker_zeta.mp.dps = MPMATH_DPS # Ensure precision is set in the worker process
    try:
        # Compute the first derivative of zeta(s) at s
        val = mp_worker_zeta.diff(mp_worker_zeta.zeta, s_complex, 1)
        return original_index, val
    except Exception:
        # Return NaN or a specific error indicator if calculation fails
        return original_index, mp_worker_zeta.nan # Or complex(np.nan, np.nan)


def _compute_P_total_vec_internal(t_values_np: np.ndarray, primes_for_p_total: list, m_terms: int, ln_primes_for_p_total: list) -> np.ndarray:
    """
    Vectorized internal computation for the P_total sum.
    Based on the formula sum_{p in P_subset} sum_{m=1}^M cos(m * t * ln(p)) / (m * p^(m/2)).
    Returns the negative of this sum as per convention.
    """
    p_total_val = np.zeros_like(t_values_np, dtype=float)

    # Reverting to the original prime loop structure as it was designed for broadcasting over m and t correctly.
    # t_values_col: (N, 1, 1)
    # m_range:      (1, M, 1)
    # ln_p_iter:    scalar (effectively (1, 1, 1))

    t_values_col_reshaped = t_values_np[:, np.newaxis, np.newaxis] # Shape (N, 1, 1)
    m_range_reshaped = np.arange(1, m_terms + 1, dtype=float)[np.newaxis, :, np.newaxis] # Shape (1, M, 1)

    for i, p_val_float in enumerate(primes_for_p_total):
        ln_p_iter = ln_primes_for_p_total[i] # Scalar
        # Denominator: (1, M, 1) * (scalar ** ((1, M, 1) / 2.0))
        # Correct is (1, M, 1) * (p_val_float ** (m_range_reshaped / 2.0)) - this results in (1, M, 1)
        denominator = m_range_reshaped * (p_val_float**(m_range_reshaped / 2.0)) # Shape (1, M, 1)

        # Cosine term: cos((1, M, 1) * (N, 1, 1) * scalar)
        # cos(m_range_reshaped * t_values_col_reshaped * ln_p_iter) # Shape (N, M, 1)
        cosine_term = np.cos(m_range_reshaped * t_values_col_reshaped * ln_p_iter) # Shape (N, M, 1)

        # Term sum for prime: Sum over M dimension
        # (1.0 / denominator) shape is (1, M, 1)
        # (1.0 / denominator) * cosine_term shape is (N, M, 1) - correct broadcasting
        term_sum_for_prime = np.sum((1.0 / denominator) * cosine_term, axis=1) # Shape (N, 1)

        # Add to total. Flatten to (N,) before adding.
        p_total_val += term_sum_for_prime.flatten() # Shape (N,)

    return -p_total_val # Return negative sum

# --- III. Core Feature Calculation Functions ---
def _ensure_or_calculate_prime_features(
    df: pd.DataFrame, prime_p: int, t_col: str, feature_names: dict,
    log_floor_threshold: float = LOG_FLOOR_THRESHOLD, log_floor_value: float = LOG_FLOOR_VALUE,
    dot_log_lookback: int = DOT_LOG_LOOKBACK_GENERAL, report_list: list = None, force_recalculate: bool = False
) -> pd.DataFrame:
    """Ensures presence or calculates features for a single prime p."""
    if report_list is None: report_list = []
    df_out = df.copy()
    ln_p = np.log(prime_p); sqrt_p = np.sqrt(prime_p)

    # Check if time column exists
    if t_col not in df_out.columns:
        msg = f"ERROR (P{prime_p}): Time column '{t_col}' not found. Skipping features."
        report_list.append(msg); print(msg); return df # Return original DF if time col missing

    t_col_data = df_out[t_col].astype(float) # Ensure float dtype for calculations

    # Define feature column names
    m1_val_col = feature_names.get("m1_val")
    abs_col = feature_names.get("abs")
    log_abs_col = feature_names.get("log_abs")
    phase_col = feature_names.get("phase")
    dot_log_abs_col = feature_names.get("dot_log_abs")

    # Calculate m1_val (always needed as base)
    if m1_val_col and (force_recalculate or m1_val_col not in df_out.columns):
        df_out[m1_val_col] = np.cos(t_col_data * ln_p) / sqrt_p
        report_list.append(f"  Calculated '{m1_val_col}' for P{prime_p}.")
    elif m1_val_col:
        report_list.append(f"  '{m1_val_col}' for P{prime_p} exists.")
    else:
        report_list.append(f"  Warning: m1_val feature name not defined for P{prime_p}. Skipping related calculations.")
        return df_out # Cannot proceed without m1_val name

    # Calculate abs
    if abs_col and (force_recalculate or abs_col not in df_out.columns):
        if m1_val_col in df_out.columns:
            df_out[abs_col] = np.abs(df_out[m1_val_col])
            report_list.append(f"  Calculated '{abs_col}' for P{prime_p}.")
        else:
             report_list.append(f"  Skipped '{abs_col}' for P{prime_p}, '{m1_val_col}' missing.")
    elif abs_col:
         report_list.append(f"  '{abs_col}' for P{prime_p} exists.")


    # Calculate log_abs
    if log_abs_col and (force_recalculate or log_abs_col not in df_out.columns):
        if abs_col in df_out.columns:
            df_out[log_abs_col] = compute_log_abs_with_floor(df_out[abs_col], log_floor_threshold, log_floor_value)
            report_list.append(f"  Calculated '{log_abs_col}' for P{prime_p}.")
        else:
            report_list.append(f"  Skipped '{log_abs_col}' for P{prime_p}, '{abs_col}' missing.")
    elif log_abs_col:
        report_list.append(f"  '{log_abs_col}' for P{prime_p} exists.")


    # Calculate phase
    if phase_col and (force_recalculate or phase_col not in df_out.columns):
        df_out[phase_col] = (t_col_data * ln_p) % (2 * np.pi)
        report_list.append(f"  Calculated '{phase_col}' for P{prime_p}.")
    elif phase_col:
        report_list.append(f"  '{phase_col}' for P{prime_p} exists.")

    # Calculate dot_log_abs (requires log_abs)
    if dot_log_abs_col and (force_recalculate or dot_log_abs_col not in df_out.columns):
        if log_abs_col in df_out.columns:
            if dot_log_lookback > 0 and len(df_out) >= dot_log_lookback:
                 # Series.diff() is generally robust to index type if Series is sorted (which the orchestration ensures)
                df_out[dot_log_abs_col] = df_out[log_abs_col].diff(periods=dot_log_lookback) / float(dot_log_lookback)
                # Fill initial NaNs from diff with a reasonable value (e.g., 0 or forward/backward fill)
                # For a derivative approximation, 0 or the first non-NaN value might be considered.
                # Let's stick to leaving the NaNs, the final dropna step handles them.
            else: # lookback is 0 or df too short
                df_out[dot_log_abs_col] = 0.0 # Or np.nan if the concept isn't applicable
                report_list.append(f"  Note: DataFrame too short ({len(df_out)} rows) or lookback 0 for '{dot_log_abs_col}'. Filled with 0.0.")
            report_list.append(f"  Calculated '{dot_log_abs_col}' for P{prime_p}.")
        else:
            report_list.append(f"  Skipped '{dot_log_abs_col}' for P{prime_p}, '{log_abs_col}' missing.")
    elif dot_log_abs_col:
        report_list.append(f"  '{dot_log_abs_col}' for P{prime_p} exists.")

    return df_out


def _calculate_pairwise_interaction_features(
    df_input: pd.DataFrame, ref_prime_p: int, correcting_prime_x: int,
    feature_names_ref: dict, feature_names_corr: dict, interaction_feature_names: dict,
    alpha_denom_epsilon: float = ALPHA_DENOM_EPSILON_GENERAL,
    impulse_zero_threshold: float = IMPULSE_ZERO_THRESHOLD_GENERAL,
    delta_phi_corr_pi_half_lower: float = DELTA_PHI_CORR_PI_HALF_LOWER_GENERAL,
    delta_phi_corr_pi_half_upper: float = DELTA_PHI_CORR_PI_HALF_UPPER_GENERAL,
    delta_phi_corr_3pi_half_lower: float = DELTA_PHI_CORR_3PI_HALF_LOWER_GENERAL,
    delta_phi_corr_3pi_half_upper: float = DELTA_PHI_CORR_3PI_HALF_UPPER_GENERAL,
    report_list: list = None, force_recalculate: bool = False
) -> pd.DataFrame:
    """Calculates pairwise interaction features between two primes."""
    if report_list is None: report_list = []
    df_out = df_input.copy()
    pair_str = f"C{correcting_prime_x}-C{ref_prime_p}" # Correcting (X) vs Reference (P)

    # Get required input column names from feature_names dicts
    col_abs_ref = feature_names_ref.get("abs")
    col_phase_ref_m1 = feature_names_ref.get("phase")
    col_dot_log_abs_ref = feature_names_ref.get("dot_log_abs")
    col_abs_corr = feature_names_corr.get("abs")
    col_phase_corr_m1 = feature_names_corr.get("phase")
    col_dot_log_abs_corr = feature_names_corr.get("dot_log_abs")

    # Check if all required input columns exist
    required_inputs = [col_abs_ref, col_phase_ref_m1, col_dot_log_abs_ref,
                       col_abs_corr, col_phase_corr_m1, col_dot_log_abs_corr]
    missing_inputs = [col for col in required_inputs if col is None or (col is not None and col not in df_out.columns)] # Check for None or missing
    if missing_inputs:
        msg = f"ERROR ({pair_str} Interaction): Missing prerequisites: {missing_inputs}. Skipping."
        report_list.append(msg); print(msg); return df_input # Return original DF if inputs missing

    # Get interaction feature column names
    alpha_col = interaction_feature_names.get("alpha_scaling")
    delta_phi_col = interaction_feature_names.get("delta_phi")
    net_impulse_col = interaction_feature_names.get("net_impulse")
    net_impulse_sign_col = interaction_feature_names.get("net_impulse_sign")
    delta_phi_pred_arm_col = interaction_feature_names.get("delta_phi_pred_arm")

    # Calculate alpha_scaling
    if alpha_col and (force_recalculate or alpha_col not in df_out.columns):
        # Handle potential division by zero in the reference amplitude by adding epsilon
        df_out[alpha_col] = df_out[col_abs_corr] / (df_out[col_abs_ref] + alpha_denom_epsilon)
        report_list.append(f"  Calculated '{alpha_col}'.")
    elif alpha_col:
        report_list.append(f"  '{alpha_col}' exists.")

    # Calculate delta_phi
    if delta_phi_col and (force_recalculate or delta_phi_col not in df_out.columns):
        # Ensure phase columns are treated as float for arithmetic
        df_out[delta_phi_col] = (df_out[col_phase_corr_m1].astype(float) - df_out[col_phase_ref_m1].astype(float)) % (2 * np.pi)
        report_list.append(f"  Calculated '{delta_phi_col}'.")
    elif delta_phi_col:
        report_list.append(f"  '{delta_phi_col}' exists.")

    # Calculate net_impulse (requires dot_log_abs for ref and corr, and delta_phi)
    if net_impulse_col and (force_recalculate or net_impulse_col not in df_out.columns):
         if delta_phi_col in df_out.columns and alpha_col in df_out.columns and col_dot_log_abs_ref in df_out.columns and col_dot_log_abs_corr in df_out.columns: # Added check for dot_log_abs columns here
            # Fill NaNs from diff() before calculating impulse, as per plan. Use 0 for NaNs.
            dlr_filled = df_out[col_dot_log_abs_ref].fillna(0)
            dlc_filled = df_out[col_dot_log_abs_corr].fillna(0)

            # Calculate interaction term: alpha * dot_log_abs_corr * cos(delta_phi)
            # Ensure columns are numeric types
            alpha_data = df_out[alpha_col].astype(float)
            cos_delta_phi = np.cos(df_out[delta_phi_col].astype(float))

            interaction_term = alpha_data * dlc_filled * cos_delta_phi

            # Net impulse = dot_log_abs_ref + interaction_term
            df_out[net_impulse_col] = dlr_filled + interaction_term
            report_list.append(f"  Calculated '{net_impulse_col}'.")
         else:
            missing_impulse_prereqs = []
            if delta_phi_col not in df_out.columns: missing_impulse_prereqs.append(f"'{delta_phi_col}'")
            if alpha_col not in df_out.columns: missing_impulse_prereqs.append(f"'{alpha_col}'")
            if col_dot_log_abs_ref not in df_out.columns: missing_impulse_prereqs.append(f"'{col_dot_log_abs_ref}'")
            if col_dot_log_abs_corr not in df_out.columns: missing_impulse_prereqs.append(f"'{col_dot_log_abs_corr}'")
            report_list.append(f"  Skipped '{net_impulse_col}': Missing prerequisites ({', '.join(missing_impulse_prereqs)}).")
    elif net_impulse_col:
        report_list.append(f"  '{net_impulse_col}' exists.")

    # Calculate net_impulse_sign (requires net_impulse)
    if net_impulse_sign_col and (force_recalculate or net_impulse_sign_col not in df_out.columns):
        if net_impulse_col in df_out.columns:
            # Use thresholds to define zero
            conditions_sign = [
                df_out[net_impulse_col] > impulse_zero_threshold,
                df_out[net_impulse_col] < -impulse_zero_threshold
            ]
            choices_sign = [1, -1]
            df_out[net_impulse_sign_col] = np.select(conditions_sign, choices_sign, default=0) # Default to 0 for values within threshold
            report_list.append(f"  Calculated '{net_impulse_sign_col}'.")
        else:
            report_list.append(f"  Skipped '{net_impulse_sign_col}': '{net_impulse_col}' missing.")
    elif net_impulse_sign_col:
        report_list.append(f"  '{net_impulse_sign_col}' exists.")

    # Calculate delta_phi_pred_arm (requires delta_phi)
    if delta_phi_pred_arm_col and (force_recalculate or delta_phi_pred_arm_col not in df_out.columns):
         if delta_phi_col in df_out.columns:
            # Define arms based on delta_phi thresholds
            conditions_arm = [
                (df_out[delta_phi_col] >= delta_phi_corr_pi_half_lower) & (df_out[delta_phi_col] < delta_phi_corr_pi_half_upper),
                (df_out[delta_phi_col] >= delta_phi_corr_3pi_half_lower) & (df_out[delta_phi_col] < delta_phi_corr_3pi_half_upper)
            ]
            choices_arm = [BRANCH_PI_HALF, BRANCH_3PI_HALF]
            df_out[delta_phi_pred_arm_col] = np.select(conditions_arm, choices_arm, default=BRANCH_TRANSITION_OTHER) # Default to OTHER
            report_list.append(f"  Calculated '{delta_phi_pred_arm_col}'.")
         else:
             report_list.append(f"  Skipped '{delta_phi_pred_arm_col}': '{delta_phi_col}' missing.")
    elif delta_phi_pred_arm_col:
        report_list.append(f"  '{delta_phi_pred_arm_col}' exists.")


    return df_out


# --- IV. Main Orchestration Block ---
if __name__ == "__main__":
    # Required for multiprocessing when packaging with PyInstaller
    freeze_support()

    # Apply plot style if matplotlib is used for saving plots later
    plt.style.use(PLT_STYLE)

    parser = argparse.ArgumentParser(
        description="Orchestrate master feature DataFrame creation for Riemann Zeros analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show default values
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recomputation of the master DataFrame even if a pre-existing Parquet file is found."
    )
    parser.add_argument(
        "--dps",
        type=int,
        default=MPMATH_DPS,
        help="Set mpmath decimal precision for zeta prime calculations."
    )
    # Optional: Add arguments for subset sizes if you want to override constants from command line
    # parser.add_argument("--first-n", type=int, default=DERIVATIVE_SUBSET_FIRST_N, help="Number of initial points for zeta prime calculation.")
    # parser.add_argument("--random-n", type=int, default=DERIVATIVE_SUBSET_RANDOM_N, help="Number of random points for zeta prime calculation.")

    args = parser.parse_args()

    # Override MPMATH_DPS if specified via command line
    MPMATH_DPS = args.dps
    mpmath.mp.dps = MPMATH_DPS # Set for main process, workers will set it too

    report_list_main = []
    orchestration_start_time = time.time()
    report_list_main.append(f"--- Master Feature DataFrame Orchestration Started: {time.ctime(orchestration_start_time)} ---")
    report_list_main.append(f"Command-line Args: {vars(args)}")
    report_list_main.append(f"Using MPMATH_DPS: {MPMATH_DPS}")


    raw_data_filepath = RAW_DATA_DIR / f"{RAW_ZEROS_FILENAME_STEM}.csv"
    master_df_filepath = PROCESSED_DATA_DIR / f"{MASTER_DF_FILENAME_STEM}.parquet"
    summary_output_dir = PROCESSED_DATA_DIR # Or a specific log directory

    ensure_dir(PROCESSED_DATA_DIR) # Ensure processed directory exists early

    df_master = None # Initialize DataFrame variable

    # --- Load Existing or Start New ---
    if not args.force_recompute and master_df_filepath.exists():
        report_list_main.append(f"\nAttempting to load existing master DataFrame: {master_df_filepath}")
        df_master = load_dataframe_robust(master_df_filepath, report_list=report_list_main)

        if df_master is not None:
            report_list_main.append("\nSuccessfully loaded existing master DataFrame. Skipping recalculation based on --force-recompute=False.")
            # If successfully loaded and not forcing recompute, we are done with this script's main purpose.
            orchestration_end_time = time.time()
            report_list_main.append(f"\n--- Master Feature DataFrame Orchestration Ended (Loaded Existing): {time.ctime(orchestration_end_time)} ---")
            report_list_main.append(f"Total Orchestration Time: {orchestration_end_time - orchestration_start_time:.2f} seconds.")
            save_summary_report(report_list_main, "master_df_orchestration_log", summary_output_dir)
            print(f"\nMaster DataFrame orchestration complete (loaded existing). Log: {summary_output_dir / 'master_df_orchestration_log.txt'}")
            sys.exit(0) # Exit successfully after loading


    # --- If df_master is None (file not found or load failed or force_recompute is True) ---
    if args.force_recompute:
        report_list_main.append(f"\nFORCE_RECOMPUTE is True. Rebuilding master DataFrame from raw: {raw_data_filepath}")
    else: # File didn't exist or load failed
        report_list_main.append(f"\nMaster DataFrame not found at {master_df_filepath} or previous load failed. Building from raw.")

    # 1. Load Raw Data
    report_list_main.append(f"\nStep 1: Loading raw data from {raw_data_filepath}")
    try:
        # Specify dtype=float for ImaginaryPart if possible to avoid mixed types/warnings
        df_master = pd.read_csv(raw_data_filepath)
        report_list_main.append(f"  Raw CSV loaded successfully. Initial shape: {df_master.shape}")
    except FileNotFoundError:
        report_list_main.append(f"  FATAL ERROR: Raw data file NOT FOUND: {raw_data_filepath}. Please ensure 'zeros6_clean.csv' is in the {RAW_DATA_DIR} directory. Halting.");
        save_summary_report(report_list_main, "master_df_orchestration_log", summary_output_dir)
        sys.exit(1)
    except Exception as e:
        report_list_main.append(f"  FATAL ERROR: Error reading {raw_data_filepath}: {e}. Halting.");
        save_summary_report(report_list_main, "master_df_orchestration_log", summary_output_dir)
        sys.exit(1)

    # 2. Basic Preprocessing (Rename, Type Conversion, Sort, Handle initial NaNs)
    report_list_main.append("\nStep 2: Basic Preprocessing (Rename, Type, Sort, Dropna)")
    if 'ImaginaryPart' in df_master.columns:
        df_master.rename(columns={'ImaginaryPart': COL_T}, inplace=True)
        report_list_main.append(f"  Renamed 'ImaginaryPart' to '{COL_T}'.")
    elif COL_T not in df_master.columns:
        report_list_main.append(f"  FATAL ERROR: Time column '{COL_T}' (or 'ImaginaryPart') not in raw data columns: {df_master.columns.tolist()}. Halting.");
        save_summary_report(report_list_main, "master_df_orchestration_log", summary_output_dir)
        sys.exit(1)

    # Convert time column to numeric, coercing errors to NaN
    df_master[COL_T] = pd.to_numeric(df_master[COL_T], errors='coerce')

    # Drop rows where the time column became NaN (could not be converted)
    rows_before_t_dropna = len(df_master)
    df_master.dropna(subset=[COL_T], inplace=True)
    rows_after_t_dropna = len(df_master)
    report_list_main.append(f"  Dropped rows where '{COL_T}' was not numeric. Rows: {rows_before_t_dropna} -> {rows_after_t_dropna} ({rows_before_t_dropna - rows_after_t_dropna} dropped).")

    # Sort by time
    df_master.sort_values(COL_T, inplace=True, ignore_index=True) # ignore_index=True resets index
    report_list_main.append(f"  Sorted DataFrame by '{COL_T}'. Shape after preprocessing: {df_master.shape}")
    initial_raw_rows = len(df_master) # Record initial count after basic cleaning

    # 3. P_total features
    report_list_main.append(f"\nStep 3: Calculating P_total features (Primes: {P_PRIMES_FOR_P_TOTAL}, M={P_M_TERMS_FOR_P_TOTAL})")
    try:
        # Pass numpy array of t values for potential speedup
        df_master[COL_P_TOTAL] = _compute_P_total_vec_internal(
            df_master[COL_T].to_numpy(), P_PRIMES_FOR_P_TOTAL, P_M_TERMS_FOR_P_TOTAL, LN_P_PRIMES_FOR_P_TOTAL)
        df_master[COL_LOG_ABS_P_TOTAL] = compute_log_abs_with_floor(
            df_master[COL_P_TOTAL], LOG_FLOOR_THRESHOLD, LOG_FLOOR_VALUE)
        report_list_main.append(f"  Calculated '{COL_P_TOTAL}' and '{COL_LOG_ABS_P_TOTAL}'.")
    except Exception as e:
        report_list_main.append(f"  ERROR calculating P_total features: {e}. Proceeding with other features.")
        # Ensure columns exist even if calculation failed, to prevent later errors
        if COL_P_TOTAL not in df_master.columns: df_master[COL_P_TOTAL] = np.nan
        if COL_LOG_ABS_P_TOTAL not in df_master.columns: df_master[COL_LOG_ABS_P_TOTAL] = np.nan


    # 4. Individual Prime Features
    report_list_main.append(f"\nStep 4: Calculating Individual Prime Features for {PRIMES_TO_PROCESS}")
    for prime_p_loop in tqdm(PRIMES_TO_PROCESS, desc="Prime Features"):
        report_list_main.append(f"  Processing Prime: {prime_p_loop}")
        # Note: _ensure_or_calculate_prime_features makes a copy internally,
        # so reassigning df_master is necessary to keep changes.
        df_master = _ensure_or_calculate_prime_features(
            df_master, prime_p_loop, COL_T, PRIME_FEATURE_NAMES.get(prime_p_loop, {}), # Use .get for safety
            log_floor_threshold=LOG_FLOOR_THRESHOLD, log_floor_value=LOG_FLOOR_VALUE,
            dot_log_lookback=DOT_LOG_LOOKBACK_GENERAL, report_list=report_list_main,
            force_recalculate=True # Always recalculate during a full build
        )
        # Check if expected columns were actually added for this prime
        # A basic check, e.g., for the m1_val column
        expected_m1_col = PRIME_FEATURE_NAMES.get(prime_p_loop, {}).get("m1_val")
        if expected_m1_col and expected_m1_col not in df_master.columns:
             report_list_main.append(f"  Warning: Prime {prime_p_loop} features may not have been calculated correctly.")

    # --- FIX APPLIED HERE: Step 5 is moved OUTSIDE the Step 4 loop ---

    # 5. Ratio P_C2m1
    report_list_main.append("\nStep 5: Calculating Ratio P_total / C2_m1")
    c2_m1_col = PRIME_FEATURE_NAMES.get(2, {}).get("m1_val")

    if COL_P_TOTAL in df_master.columns and c2_m1_col and c2_m1_col in df_master.columns:
        try:
            # Ensure c2_m1_col is float for division and abs() check
            df_master[c2_m1_col] = pd.to_numeric(df_master[c2_m1_col], errors='coerce')

            # Calculate the ratio
            # Handle potential division by zero C2_m1 values by setting ratio to NaN
            # Do this *before* calculating the ratio to avoid RuntimeWarning and inf
            mask_zero_denom = np.abs(df_master[c2_m1_col].fillna(0)) < RATIO_C2M1_ZERO_THRESHOLD # Fill NaN in denom check
            
            df_master[COL_RATIO_P_C2M1] = np.nan # Initialize column
            
            # Perform division only where denominator is non-zero
            df_master.loc[~mask_zero_denom, COL_RATIO_P_C2M1] = df_master.loc[~mask_zero_denom, COL_P_TOTAL] / df_master.loc[~mask_zero_denom, c2_m1_col]

            # The previous replace inf/ -inf might still be useful if NaN in numerator propagates oddly, but less likely with this approach.
            # df_master[COL_RATIO_P_C2M1] = df_master[COL_RATIO_P_C2M1].replace([np.inf, -np.inf], np.nan) # Redundant with mask logic, but harmless.

            report_list_main.append(f"  Calculated '{COL_RATIO_P_C2M1}'. NaNs introduced by near-zero denominator: {mask_zero_denom.sum()}")
            # Also check for NaNs from other sources (e.g., NaN in numerator)
            report_list_main.append(f"  Total NaNs in '{COL_RATIO_P_C2M1}': {df_master[COL_RATIO_P_C2M1].isnull().sum()}")

        except Exception as e:
            report_list_main.append(f"  ERROR calculating '{COL_RATIO_P_C2M1}': {e}. Column may be incomplete/missing.")
            if COL_RATIO_P_C2M1 not in df_master.columns:
                 df_master[COL_RATIO_P_C2M1] = np.nan # Ensure column exists even on error
    else:
        missing_prereqs = []
        if COL_P_TOTAL not in df_master.columns: missing_prereqs.append(f"'{COL_P_TOTAL}'")
        if not c2_m1_col: missing_prereqs.append("C2_m1 column name from PRIME_FEATURE_NAMES[2]")
        elif c2_m1_col not in df_master.columns: missing_prereqs.append(f"'{c2_m1_col}'")
        report_list_main.append(f"  Skipped '{COL_RATIO_P_C2M1}': Prerequisites missing ({', '.join(missing_prereqs)}).")
        df_master[COL_RATIO_P_C2M1] = np.nan # Ensure column exists even if skipped


    # 6. Zeta Prime Features
    report_list_main.append("\nStep 6: Calculating Zeta Prime Features (on subset)")
    zeta_prime_cols_main = [COL_ZETA_PRIME_REAL, COL_ZETA_PRIME_IMAG, COL_ARG_ZETA_PRIME, COL_ABS_ZETA_PRIME, COL_ABS_ARG_ZETA_PRIME_MOD_PI]
    # Initialize only if recomputing OR if column doesn't exist for any core ZP column
    if args.force_recompute or any(col not in df_master.columns for col in zeta_prime_cols_main):
         for zp_col in zeta_prime_cols_main:
             if zp_col not in df_master.columns:
                 df_master[zp_col] = np.nan
                 report_list_main.append(f"  Initializing missing Zeta Prime column '{zp_col}' with NaNs.")
             # If force_recompute is True, we don't strictly *need* to initialize with NaN if it exists,
             # as the subset logic will overwrite the relevant rows. But explicit NaN init is safe.

         current_df_len = len(df_master)
         first_n_actual = min(DERIVATIVE_SUBSET_FIRST_N, current_df_len)
         first_n_indices_zp = df_master.index[:first_n_actual]

         subset_indices_zp_list = list(first_n_indices_zp) # Start with first_n

         remaining_indices_zp = df_master.index[first_n_actual:]
         if len(remaining_indices_zp) > 0:
             random_n_actual_zp = min(DERIVATIVE_SUBSET_RANDOM_N, len(remaining_indices_zp))
             if random_n_actual_zp > 0:
                 np.random.seed(RANDOM_SEED)
                 random_indices_selected_zp = np.random.choice(remaining_indices_zp, size=random_n_actual_zp, replace=False)
                 subset_indices_zp_list.extend(list(random_indices_selected_zp))

         # Use a set for uniqueness then convert back to sorted list/array if order matters (it doesn't strictly here, but helps reporting)
         subset_indices_zp = sorted(list(set(subset_indices_zp_list)))

         report_list_main.append(f"  Selected {len(subset_indices_zp)} unique indices for Zeta Prime computation (First {first_n_actual}, Random {random_n_actual_zp}).")

         if len(subset_indices_zp) > 0:
             s_complex_tuples_zp = []
             valid_indices_count = 0
             for idx_zp in subset_indices_zp:
                 if idx_zp in df_master.index and pd.notna(df_master.loc[idx_zp, COL_T]): # Ensure index is still valid and t is not NaN
                     s_complex_tuples_zp.append((idx_zp, mpmath.mpc(0.5, df_master.loc[idx_zp, COL_T])))
                     valid_indices_count += 1
                 else:
                     report_list_main.append(f"  Skipping ZP calc for index {idx_zp}: Index or T value invalid.")

             report_list_main.append(f"  Preparing {valid_indices_count} s_complex values for workers (MPMATH_DPS={MPMATH_DPS}).")

             if s_complex_tuples_zp:
                 results_zp = []
                 num_processes = cpu_count() # Use all available cores
                 report_list_main.append(f"  Using {num_processes} processes for Zeta Prime.")

                 # Dynamic chunksize calculation
                 base_chunk_multiplier = 16 # Start with a larger multiplier
                 if len(s_complex_tuples_zp) > num_processes * base_chunk_multiplier:
                     dynamic_chunksize = max(1, len(s_complex_tuples_zp) // (num_processes * base_chunk_multiplier))
                 elif len(s_complex_tuples_zp) > num_processes * 4:
                      dynamic_chunksize = max(1, len(s_complex_tuples_zp) // (num_processes * 4))
                 else:
                     dynamic_chunksize = 1
                 dynamic_chunksize = min(dynamic_chunksize, 512) # Cap chunksize to avoid extremely large chunks
                 report_list_main.append(f"  Using dynamic chunksize: {dynamic_chunksize} for imap_unordered.")

                 try:
                     with Pool(processes=num_processes) as pool:
                         # Use imap_unordered with chunksize for progress bar and potentially better memory usage
                         for result_item in tqdm(pool.imap_unordered(_compute_zeta_derivative_worker_shared, s_complex_tuples_zp, chunksize=dynamic_chunksize),
                                                total=len(s_complex_tuples_zp), desc="Zeta Prime Calc"):
                             results_zp.append(result_item)

                     successfully_calculated_count = 0
                     for original_idx, zp_val in results_zp:
                         # Use mpmath.isnan for mpf and mpc types
                         if not mpmath.isnan(zp_val) and not (isinstance(zp_val, mpmath.mpc) and (mpmath.isnan(zp_val.real) or mpmath.isnan(zp_val.imag))):
                             # Ensure results are converted to standard Python floats/complex for pandas
                             df_master.loc[original_idx, COL_ZETA_PRIME_REAL] = float(mpmath.re(zp_val))
                             df_master.loc[original_idx, COL_ZETA_PRIME_IMAG] = float(mpmath.im(zp_val))
                             df_master.loc[original_idx, COL_ARG_ZETA_PRIME] = float(mpmath.arg(zp_val))
                             df_master.loc[original_idx, COL_ABS_ZETA_PRIME] = float(mpmath.fabs(zp_val))
                             successfully_calculated_count += 1
                     report_list_main.append(f"  Populated Zeta Prime features for {successfully_calculated_count} results.")
                     if successfully_calculated_count < valid_indices_count:
                         report_list_main.append(f"  Note: {valid_indices_count - successfully_calculated_count} Zeta Prime calculations failed for subset indices.")

                 except Exception as e:
                     report_list_main.append(f"  ERROR during multiprocessing pool execution for Zeta Prime: {e}")
                     report_list_main.append("  Zeta Prime features will be incomplete.")

             else:
                 report_list_main.append("  No valid s_complex values to process for Zeta Prime computation after checks.")
         else:
             report_list_main.append("  No indices selected for Zeta Prime computation.")

         # Calculate COL_ABS_ARG_ZETA_PRIME_MOD_PI after ZP calculations, but only for rows where COL_ARG_ZETA_PRIME is not NaN
         if COL_ARG_ZETA_PRIME in df_master.columns and df_master[COL_ARG_ZETA_PRIME].notna().any():
             # Calculate only for non-NaN rows
             notna_arg_mask = df_master[COL_ARG_ZETA_PRIME].notna()
             if COL_ABS_ARG_ZETA_PRIME_MOD_PI not in df_master.columns:
                  df_master[COL_ABS_ARG_ZETA_PRIME_MOD_PI] = np.nan # Initialize if missing
             df_master.loc[notna_arg_mask, COL_ABS_ARG_ZETA_PRIME_MOD_PI] = np.abs(df_master.loc[notna_arg_mask, COL_ARG_ZETA_PRIME] % np.pi)
             report_list_main.append(f"  Calculated '{COL_ABS_ARG_ZETA_PRIME_MOD_PI}'. Populated for {df_master[COL_ABS_ARG_ZETA_PRIME_MOD_PI].notna().sum()} rows.")
         else:
             report_list_main.append(f"  Skipped '{COL_ABS_ARG_ZETA_PRIME_MOD_PI}' as '{COL_ARG_ZETA_PRIME}' is missing or all NaN after ZP calc.")

    else:
        report_list_main.append("  Skipping Zeta Prime calculation: Columns already exist and force_recompute is False.")
        if any(col not in df_master.columns for col in zeta_prime_cols_main):
             report_list_main.append("  Warning: Some core Zeta Prime columns are missing despite skipping recalculation.")

    # 7. C2 Breath Sign Numerical
    report_list_main.append("\nStep 7: Calculating C2 Breath Sign Numerical")
    dot_log_abs_c2_col = PRIME_FEATURE_NAMES.get(2, {}).get('dot_log_abs')
    if dot_log_abs_c2_col and dot_log_abs_c2_col in df_master.columns:
        try:
            # Fill NaNs in the source column temporarily for the comparison (or handle with mask)
            # Using a mask is generally safer than fillna(0) if 0 is a valid value near the threshold
            source_series = df_master[dot_log_abs_c2_col]
            conditions_c2b = [
                source_series > C2_BREATH_SIGN_ZERO_THRESHOLD,
                source_series < -C2_BREATH_SIGN_ZERO_THRESHOLD
            ]
            # Use np.select to assign values based on conditions, leaving NaN if source is NaN
            # np.select handles NaNs in the source conditions by default, resulting in default=0
            df_master[COL_C2_BREATH_SIGN_NUMERICAL] = np.select(conditions_c2b, [1, -1], default=0) # Default to 0 for values within threshold or NaN source
            # Explicitly set to NaN where source was NaN if 0 is not desired for NaN
            # If NaN source should map to NaN result: df_master.loc[source_series.isna(), COL_C2_BREATH_SIGN_NUMERICAL] = np.nan
            # Current logic maps NaN source to 0, which might be acceptable depending on interpretation. Sticking with default 0 for now.

            report_list_main.append(f"  Calculated '{COL_C2_BREATH_SIGN_NUMERICAL}'.")
            report_list_main.append(f"  '{COL_C2_BREATH_SIGN_NUMERICAL}' Counts (incl. 0s): {df_master[COL_C2_BREATH_SIGN_NUMERICAL].value_counts().to_string()}")

        except Exception as e:
            report_list_main.append(f"  ERROR calculating '{COL_C2_BREATH_SIGN_NUMERICAL}': {e}. Column may be incomplete/missing.")
            if COL_C2_BREATH_SIGN_NUMERICAL not in df_master.columns:
                 df_master[COL_C2_BREATH_SIGN_NUMERICAL] = np.nan # Ensure column exists even on error
    else:
        missing_c2b_prereq = f"'{dot_log_abs_c2_col}'" if dot_log_abs_c2_col else "dot_log_abs column name for P2"
        report_list_main.append(f"  Skipped '{COL_C2_BREATH_SIGN_NUMERICAL}': {missing_c2b_prereq} missing.")
        if COL_C2_BREATH_SIGN_NUMERICAL not in df_master.columns:
             df_master[COL_C2_BREATH_SIGN_NUMERICAL] = np.nan # Ensure column exists even if skipped


    # 8. Initial Phase Branch
    report_list_main.append("\nStep 8: Calculating Initial Phase Branch (based on C2 phase)")
    phase_c2_col = PRIME_FEATURE_NAMES.get(2, {}).get("phase")
    if phase_c2_col and phase_c2_col in df_master.columns:
        try:
            phase_c2_data = df_master[phase_c2_col].astype(float) # Ensure float
            # Define the conditions for each branch based on C2 phase thresholds
            conditions_pb = [
                (phase_c2_data >= PHASE_PI_HALF_LOWER_GENERAL) & (phase_c2_data < PHASE_PI_HALF_UPPER_GENERAL),
                (phase_c2_data >= PHASE_3PI_HALF_LOWER_GENERAL) & (phase_c2_data < PHASE_3PI_HALF_UPPER_GENERAL)
            ]
            # Define the corresponding branch values
            choices_pb = [BRANCH_PI_HALF, BRANCH_3PI_HALF]
            # Use np.select to assign branch based on conditions, default to OTHER.
            # np.select handles NaNs in phase_c2_data correctly, assigning the default value (BRANCH_TRANSITION_OTHER)
            df_master[COL_PHASE_BRANCH] = np.select(conditions_pb, choices_pb, default=BRANCH_TRANSITION_OTHER)
            report_list_main.append(f"  Calculated '{COL_PHASE_BRANCH}'.")
            # Report value counts for verification
            report_list_main.append(f"  '{COL_PHASE_BRANCH}' Counts:\n{df_master[COL_PHASE_BRANCH].value_counts(normalize=True).apply(lambda x: f'{x*100:.2f}%').to_string()}")
        except Exception as e:
            report_list_main.append(f"  ERROR calculating '{COL_PHASE_BRANCH}': {e}. Column may be incomplete/missing.")
            if COL_PHASE_BRANCH not in df_master.columns:
                 df_master[COL_PHASE_BRANCH] = np.nan # Ensure column exists even on error
    else:
        missing_pb_prereq = f"'{phase_c2_col}'" if phase_c2_col else "phase column name for P2"
        report_list_main.append(f"  Skipped '{COL_PHASE_BRANCH}': {missing_pb_prereq} missing.")
        if COL_PHASE_BRANCH not in df_master.columns:
             df_master[COL_PHASE_BRANCH] = np.nan # Ensure column exists even if skipped


    # 9. Pairwise Interaction Features
    report_list_main.append(f"\nStep 9: Calculating Pairwise Interaction Features for {INTERACTION_PAIRS_TO_PROCESS}")
    for ref_p_loop, corr_p_loop in tqdm(INTERACTION_PAIRS_TO_PROCESS, desc="Interaction Features"):
        pair = (ref_p_loop, corr_p_loop)
        report_list_main.append(f"  Processing Interaction Pair: {pair}")
        # Note: _calculate_pairwise_interaction_features makes a copy internally,
        # so reassigning df_master is necessary to keep changes.
        # Use .get for safety when accessing feature name dicts
        feature_names_ref = PRIME_FEATURE_NAMES.get(ref_p_loop, {})
        feature_names_corr = PRIME_FEATURE_NAMES.get(corr_p_loop, {})
        interaction_feature_names = PRIME_INTERACTION_FEATURE_NAMES.get(pair, {})

        if not feature_names_ref or not feature_names_corr or not interaction_feature_names:
            report_list_main.append(f"  Skipping interaction {pair}: Feature names not found in constants.")
            # Ensure expected output columns for this pair are added as NaN if skipped
            for feat_name in interaction_feature_names.values():
                 if feat_name not in df_master.columns:
                      df_master[feat_name] = np.nan
            continue # Skip to next pair

        df_master = _calculate_pairwise_interaction_features(
            df_master, ref_p_loop, corr_p_loop,
            feature_names_ref, feature_names_corr,
            interaction_feature_names,
            alpha_denom_epsilon=ALPHA_DENOM_EPSILON_GENERAL,
            impulse_zero_threshold=IMPULSE_ZERO_THRESHOLD_GENERAL,
            delta_phi_corr_pi_half_lower=DELTA_PHI_CORR_PI_HALF_LOWER_GENERAL,
            delta_phi_corr_pi_half_upper=DELTA_PHI_CORR_PI_HALF_UPPER_GENERAL,
            delta_phi_corr_3pi_half_lower=DELTA_PHI_CORR_3PI_HALF_LOWER_GENERAL,
            delta_phi_corr_3pi_half_upper=DELTA_PHI_CORR_3PI_HALF_UPPER_GENERAL,
            report_list=report_list_main, force_recalculate=True # Always recalculate during a full build
        )
        # Add a quick check that at least one expected interaction column was added
        expected_alpha_col = interaction_feature_names.get("alpha_scaling")
        if expected_alpha_col and expected_alpha_col not in df_master.columns:
            report_list_main.append(f"  Warning: Interaction {pair} features may not have been calculated correctly.")


    # 10. Handle NaNs from diff() (and other potential sources if needed)
    report_list_main.append("\nStep 10: Handling NaNs introduced by .diff() and other calculations")

    # Collect all columns that are expected to have leading NaNs from diff()
    dot_log_abs_cols_to_check = [
        PRIME_FEATURE_NAMES[p]["dot_log_abs"] for p in PRIMES_TO_PROCESS
        if p in PRIME_FEATURE_NAMES and "dot_log_abs" in PRIME_FEATURE_NAMES[p]
    ]
    # Collect all net_impulse columns
    net_impulse_cols_to_check = [
        PRIME_INTERACTION_FEATURE_NAMES[pair]["net_impulse"] for pair in INTERACTION_PAIRS_TO_PROCESS
        if pair in PRIME_INTERACTION_FEATURE_NAMES and "net_impulse" in PRIME_INTERACTION_FEATURE_NAMES[pair]
    ]

    # Combine and take only columns that actually exist in the DataFrame
    dropna_subset_cols = [col for col in (dot_log_abs_cols_to_check + net_impulse_cols_to_check) if col in df_master.columns]

    if dropna_subset_cols:
        report_list_main.append(f"  Dropping rows with NaNs in subset columns: {dropna_subset_cols}")
        rows_before_dropna = len(df_master)
        df_master.dropna(subset=dropna_subset_cols, inplace=True)
        rows_after_dropna = len(df_master)
        report_list_main.append(f"  Rows: {rows_before_dropna} -> {rows_after_dropna} ({rows_before_dropna - rows_after_dropna} dropped).")
        if rows_after_dropna == 0:
            report_list_main.append("  FATAL ERROR: All rows dropped after removing NaNs. Check data and lookback settings. Halting.")
            save_summary_report(report_list_main, "master_df_orchestration_log", summary_output_dir)
            sys.exit(1)
        # Reset index after dropping rows
        df_master.reset_index(drop=True, inplace=True)
        report_list_main.append("  Reset DataFrame index after dropping rows.")
    else:
        report_list_main.append(f"  No specified diff-based or impulse columns found for targeted dropna. Shape: {df_master.shape}")


    # Final Check on DataFrame size
    if initial_raw_rows > 0 and len(df_master) < initial_raw_rows * 0.5: # Arbitrary threshold for warning
         report_list_main.append(f"  WARNING: Significant number of rows dropped ({initial_raw_rows} -> {len(df_master)}). Consider investigating NaN sources.")
    elif initial_raw_rows == 0:
         report_list_main.append("  Warning: Initial raw data had 0 rows after basic cleaning.")


    # 11. Save Final Master DataFrame
    report_list_main.append(f"\nStep 11: Saving Master DataFrame to {master_df_filepath}")
    # Ensure the directory exists before saving
    ensure_dir(master_df_filepath.parent)
    save_dataframe_robust(df_master, master_df_filepath, report_list=report_list_main)

    # --- Orchestration Complete ---
    orchestration_end_time = time.time()
    report_list_main.append(f"\n--- Master Feature DataFrame Orchestration Ended: {time.ctime(orchestration_end_time)} ---")
    report_list_main.append(f"Total Orchestration Time: {orchestration_end_time - orchestration_start_time:.2f} seconds.")
    report_list_main.append(f"Final DataFrame Shape: {df_master.shape}")


    # Save the full orchestration log
    save_summary_report(report_list_main, "master_df_orchestration_log", summary_output_dir)
    print(f"\nMaster DataFrame orchestration complete. Log saved to: {summary_output_dir / 'master_df_orchestration_log.txt'}")