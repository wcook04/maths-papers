# Paper_12/scripts/constants.py

import numpy as np
from pathlib import Path

import numpy as np
from pathlib import Path
# --- I. Directory Path Constants ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = BASE_DIR / "results"
PREDICTOR_RESULTS_DIR_NAME = "unified_predictor_results" # <-- ADD THIS LINE

# --- II. Core Branch Definition Constants ---
BRANCH_PI_HALF = 'Branch_Pi_Half'
BRANCH_3PI_HALF = 'Branch_3Pi_Half'
BRANCH_TRANSITION_OTHER = 'Transition/Other' # Also "Spine"
ALL_BRANCHES = [BRANCH_PI_HALF, BRANCH_3PI_HALF, BRANCH_TRANSITION_OTHER]

# --- III. Core Static Column Name Constants (Minimal Set for Oracle Output) ---
COL_T = 't'
COL_PHASE_BRANCH = 'Phase_Branch' # Initial derivation based on C2 phase

COL_P_TOTAL = 'P_total'
COL_LOG_ABS_P_TOTAL = 'log_abs_P_total'

# C2_m1 Specific (will also be dynamically named via PRIME_FEATURE_NAMES[2])
COL_C2_M1 = 'C2_m1' # Retained for direct reference in some older/specific logic
COL_LOG_ABS_C2_M1 = 'log_abs_C2_m1' # Retained
COL_PHASE_C2_M1 = 'Phase_C2_m1' # Retained

COL_RATIO_P_C2M1 = 'Ratio_P_C2m1'

# Zeta Prime Related
COL_ZETA_PRIME_REAL = 'zeta_prime_real'
COL_ZETA_PRIME_IMAG = 'zeta_prime_imag'
COL_ARG_ZETA_PRIME = 'arg_zeta_prime'
COL_ABS_ZETA_PRIME = 'abs_zeta_prime'
COL_ABS_ARG_ZETA_PRIME_MOD_PI = 'abs_arg_zeta_prime_mod_pi'

# C2 Breath Sign
COL_C2_BREATH_SIGN_NUMERICAL = 'C2_breath_sign_numerical'

# --- Analysis-specific columns (not computed by core oracle, but defined for consistency) ---
COL_ACTUAL_NEXT_BRANCH = 'Actual_Next_Branch'
COL_NEXT_PHASE_BRANCH = 'Next_Phase_Branch'
COL_TRANSITION_TYPE = 'Transition_Type'
COL_LOG_ABS_C2_M1_BIN = 'log_abs_C2_m1_bin' # From main_analysis.py Phase 1
COL_PHASE_DIST_0_PI = 'Phase_Dist_0_pi'     # From main_analysis.py Phase 3
COL_MOD4_INDEX_C2M1 = 'Mod4_Index_C2m1'     # From main_analysis.py Phase 3
COL_DELTA_MOD4_INDEX = 'delta_Mod4_Index'   # From main_analysis.py Phase 5

# Predictor Output Column Names (from unified_hierarchical_predictor.py)
COL_ACTUAL_NEXT_BRANCH_TARGET = 'Actual_Next_Branch_Target'
COL_PREDICTED_FINAL_HIERARCHICAL = 'Predicted_Next_Branch_Final_Hierarchical'
COL_PREDICTED_BASE_C2LAW = 'Predicted_Next_Branch_Base_C2Law' # Add this too, it's likely used
COL_CORRECTION_LAYER_APPLIED = 'Correction_Layer_Applied'     # Add this too, it's used in error_analysis.py

# --- IV. Hierarchical Prediction Column Names ---
# (Defined for downstream scripts, not computed by shared_utils.py oracle builder)
COL_C2_LAW_PRED = 'C2_Law_Predicted_Next_Branch'
COL_P8_HIERARCHICAL_PRED = 'Hierarchical_Predicted_Next_Branch' # Output of C2+C3 layer (Part 8)
COL_P9_HIERARCHICAL_PRED = 'Final_Predicted_Next_Branch_P9'     # Output of P8 + C5 layer (Part 9)
COL_P10_HIERARCHICAL_PRED = 'Final_Predicted_Next_Branch_P10'   # Output of P9 + C7 layer (Part 10)
COL_P11_HIERARCHICAL_PRED = 'Final_Predicted_Next_Branch_P11'   # Output of P10 + C11 layer (Part 11)
COL_P12_HIERARCHICAL_PRED = 'Final_Predicted_Next_Branch_P12'   # Placeholder for P11 + C13

# --- V. Prime Processing Scope & P_total Parameters ---
PRIMES_TO_PROCESS = [2, 3, 5, 7, 11, 13]
P_PRIMES_FOR_P_TOTAL = [2, 3, 5, 7, 11]
P_M_TERMS_FOR_P_TOTAL = 1 # As per oracle decision
LN_P_PRIMES_FOR_P_TOTAL = [np.log(p) for p in P_PRIMES_FOR_P_TOTAL]
LN_2 = np.log(2.0)
# --- Add these to your constants.py file ---

# ... (previous constants) ...

# --- IX. Predictor Layer Specific Constants & Thresholds (P9+) ---

# Layer Names (defined for clarity and reporting)
PREDICTOR_LAYER_P9_NAME = "P9_SpineArmErrorCorrector" # Example descriptive name
# Add names for future layers as needed, e.g.:
# PREDICTOR_LAYER_P10_NAME = "P10_ArmSpineErrorCorrector"
# PREDICTOR_LAYER_P11_NAME = "P11_ResidualCorrector"
# PREDICTOR_LAYER_P12_NAME = "P12_FinalTune"


# P9 Rule Thresholds (These are PLACEHOLDERS - REPLACE WITH EMPIRICAL VALUES FROM ERROR ANALYSIS)
# Example Thresholds for correcting Predicted Spine (T/O) to Actual Arm
# You need to determine these specific ranges/values from your error analysis plots/data for errors
C2_PHASE_AMBIGUOUS_RANGE_1_LOWER = 0.0 # Placeholder, e.g., range near 0/2pi
C2_PHASE_AMBIGUOUS_RANGE_1_UPPER = 0.0 # Placeholder
C2_PHASE_AMBIGUOUS_RANGE_2_LOWER = 0.0 # Placeholder, e.g., range near Pi/2 (unlikely for P9 errors, but maybe P10)
C2_PHASE_AMBIGUOUS_RANGE_2_UPPER = 0.0 # Placeholder
C2_PHASE_AMBIGUOUS_RANGE_3_LOWER = 0.0 # Placeholder, e.g., range near Pi
C2_PHASE_AMBIGUOUS_RANGE_3_UPPER = 0.0 # Placeholder
C2_PHASE_AMBIGUOUS_RANGE_4_LOWER = 0.0 # Placeholder, e.g., range near 3Pi/2 (unlikely for P9 errors, but maybe P10)
C2_PHASE_AMBIGUOUS_RANGE_4_UPPER = 0.0 # Placeholder
C2_PHASE_AMBIGUOUS_RANGE_5_LOWER = 0.0 # Placeholder, e.g., range near 2Pi/3 or 4Pi/3 (likely for P9 errors based on analysis)
C2_PHASE_AMBIGUOUS_RANGE_5_UPPER = 0.0 # Placeholder
C2_PHASE_AMBIGUOUS_RANGE_6_LOWER = 0.0 # Placeholder
C2_PHASE_AMBIGUOUS_RANGE_6_UPPER = 0.0 # Placeholder

ALPHA_SCALING_LOW_THRESHOLD_P9 = 0.0 # Placeholder, tune based on alpha_scaling plots for P9 errors

IMPULSE_SMALL_POSITIVE_LOWER_P9 = 0.0 # Placeholder, likely slightly above 0 or IMPULSE_ZERO_THRESHOLD_GENERAL
IMPULSE_SMALL_POSITIVE_UPPER_P9 = 0.0 # Placeholder, tune based on Net_Impulse plots for P9 errors

DELTA_PHI_NEAR_PI_HALF_LOWER_P9 = 0.0 # Placeholder, tune based on Delta_Phi plots for P9 errors
DELTA_PHI_NEAR_PI_HALF_UPPER_P9 = 0.0 # Placeholder
DELTA_PHI_NEAR_3PI_HALF_LOWER_P9 = 0.0 # Placeholder, tune based on Delta_Phi plots for P9 errors
DELTA_PHI_NEAR_3PI_HALF_UPPER_P9 = 0.0 # Placeholder
# Add other delta phi ranges (near 0, pi, etc.) as needed based on analysis

# P10 Rule Thresholds (Placeholders for future layers)
# Add similar threshold constants for P10 rules (e.g., C2_PHASE_P10_SPECIFIC_RANGE_1_LOWER, ZETA_PRIME_ABS_HIGH_THRESHOLD_P10, etc.)

# ... (rest of constants.py) ...

# --- VI. Functions for Generating Dynamic Column Names ---
def get_prime_feature_names(prime_p: int) -> dict:
    """Generates standardized column names for individual prime features."""
    return {
        "m1_val": f'C{prime_p}_m1',
        "abs": f'abs_C{prime_p}',
        "log_abs": f'log_abs_C{prime_p}_m1',
        "phase": f'phase_C{prime_p}_m1',
        "dot_log_abs": f'dot_log_abs_C{prime_p}'
    }

def get_prime_interaction_feature_names(ref_prime: int, corr_prime: int) -> dict:
    """Generates standardized column names for pairwise prime interaction features."""
    return {
        "alpha_scaling": f'alpha_scaling_C{corr_prime}_by_C{ref_prime}',
        "delta_phi": f'Delta_Phi_C{corr_prime}_C{ref_prime}',
        "net_impulse": f'Net_C{ref_prime}C{corr_prime}_Impulse',
        "net_impulse_sign": f'Net_C{ref_prime}C{corr_prime}_Impulse_Sign',
        "delta_phi_pred_arm": f'delta_phi_C{corr_prime}C{ref_prime}_corr_predicted_arm'
    }

# --- VII. Derived Aggregated Column Name Constants ---
PRIME_FEATURE_NAMES = {p: get_prime_feature_names(p) for p in PRIMES_TO_PROCESS}
ALL_INDIVIDUAL_PRIME_FEATURE_COLUMNS = sorted(list(set(
    col_name for p_features in PRIME_FEATURE_NAMES.values() for col_name in p_features.values()
)))

# Define C2 as the reference prime for all interactions
REFERENCE_PRIME_FOR_INTERACTIONS = 2
INTERACTION_PAIRS_TO_PROCESS = [
    (REFERENCE_PRIME_FOR_INTERACTIONS, p)
    for p in PRIMES_TO_PROCESS if p != REFERENCE_PRIME_FOR_INTERACTIONS
] # Generates [(2, 3), (2, 5), (2, 7), (2, 11), (2, 13)]

PRIME_INTERACTION_FEATURE_NAMES = {
    pair: get_prime_interaction_feature_names(pair[0], pair[1])
    for pair in INTERACTION_PAIRS_TO_PROCESS
}
ALL_INTERACTION_FEATURE_COLUMNS = sorted(list(set(
    col_name for pair_features in PRIME_INTERACTION_FEATURE_NAMES.values() for col_name in pair_features.values()
)))

# --- VIII. General Calculation Parameter Constants ---
LOG_EPSILON_GENERAL = 1e-12 # Fallback for generic log10, primary log_abs uses floor logic
LOG_FLOOR_VALUE = -10.0
LOG_FLOOR_THRESHOLD = 1e-10

RATIO_C2M1_ZERO_THRESHOLD = 1e-10

DOT_LOG_LOOKBACK_GENERAL = 1
ALPHA_DENOM_EPSILON_GENERAL = 1e-12
IMPULSE_ZERO_THRESHOLD_GENERAL = 1e-9
C2_BREATH_SIGN_ZERO_THRESHOLD = 1e-9 # Used for COL_C2_BREATH_SIGN_NUMERICAL

MPMATH_DPS = 50

# Zeta Prime Subset Parameters (from main_analysis.py)
DERIVATIVE_SUBSET_FIRST_N = 10000
DERIVATIVE_SUBSET_RANDOM_N = 10000
RANDOM_SEED = 42

## --- Add or update this section in your constants.py file ---

# --- Add or update this section in your constants.py file ---

# --- Add or update this section in your constants.py file ---

# ... (Sections I through VIII) ...

# --- IX. General Phase/Delta Phi Boundary Constants ---
# These should be defined *before* the P9 constants below as they are referenced.
PHASE_PI_HALF_LOWER_GENERAL = np.pi / 4.0
PHASE_PI_HALF_UPPER_GENERAL = 3.0 * np.pi / 4.0
PHASE_3PI_HALF_LOWER_GENERAL = 5.0 * np.pi / 4.0
PHASE_3PI_HALF_UPPER_GENERAL = 7.0 * np.pi / 4.0

DELTA_PHI_CORR_PI_HALF_LOWER_GENERAL = 0.3 * np.pi # Approx 0.942
DELTA_PHI_CORR_PI_HALF_UPPER_GENERAL = 0.7 * np.pi # Approx 2.199
DELTA_PHI_CORR_3PI_HALF_LOWER_GENERAL = 1.3 * np.pi # Approx 4.084
DELTA_PHI_CORR_3PI_HALF_UPPER_GENERAL = 1.7 * np.pi # Approx 5.340

# --- X. Predictor Layer Specific Constants & Thresholds (P9+) ---
# This section MUST come *after* the constants defined in Section IX above.

# Layer Names (defined for clarity and reporting)
PREDICTOR_LAYER_P9_NAME = "P9_SpineArmErrorCorrector"
# PREDICTOR_LAYER_P10_NAME = "P10_ArmSpineErrorCorrector" # For future errors

# P9 Rule Thresholds (TUNED BASED ON THE PROVIDED NUMERICAL ANALYSIS OUTPUT)

# C2 Phase Ambiguous Ranges (in radians) - Where errors concentrate
# Range 1: Around 2pi/3 (targets some Actual Pi_Half errors)
C2_PHASE_AMBIGUOUS_RANGE_1_LOWER = 2.0
C2_PHASE_AMBIGUOUS_RANGE_1_UPPER = 2.2
# Range 2: Around 4pi/3 (placeholder, refine if needed from analysis of other error types)
C2_PHASE_AMBIGUOUS_RANGE_2_LOWER = 4.1
C2_PHASE_AMBIGUOUS_RANGE_2_UPPER = 4.3
# Range 3: Near 0 (targets some Actual Pi_Half errors)
C2_PHASE_AMBIGUOUS_RANGE_3_LOWER = 0.0
C2_PHASE_AMBIGUOUS_RANGE_3_UPPER = 0.12 * np.pi # Approx 0.377
# Range 4: Near Pi (targets some Actual 3Pi_Half errors)
C2_PHASE_AMBIGUOUS_RANGE_4_LOWER = 0.88 * np.pi # Approx 2.765
C2_PHASE_AMBIGUOUS_RANGE_4_UPPER = 1.12 * np.pi # Approx 3.519
# Range 5: Near 2pi (targets some Actual Pi_Half errors)
C2_PHASE_AMBIGUOUS_RANGE_5_LOWER = 1.88 * np.pi # Approx 5.901
C2_PHASE_AMBIGUOUS_RANGE_5_UPPER = 2.0 * np.pi - 1e-9 # Just under 2pi (Approx 6.283)
# Range 6: General mid-range where C2 law might be weak (can be refined, not strongly indicated for these errors)
C2_PHASE_AMBIGUOUS_RANGE_6_LOWER = 0.4 * np.pi
C2_PHASE_AMBIGUOUS_RANGE_6_UPPER = 0.6 * np.pi


# Low Alpha Scaling Threshold (Correcting prime is weaker than C2 relative to other points)
ALPHA_SCALING_LOW_THRESHOLD_P9 = 1.0 # Based on general indication that errors have alpha < 1.0 relative to mean/median of correct points

# Small Positive Impulse Range (Where errors concentrate, contrasting with P8 impulse < 0 rule)
# Lower bound is the zero threshold, upper bound captures the bulk (~75th percentile) of positive impulses in errors.
IMPULSE_SMALL_POSITIVE_LOWER_P9 = IMPULSE_ZERO_THRESHOLD_GENERAL # Use the general zero threshold (1e-9)
IMPULSE_SMALL_POSITIVE_UPPER_P9 = 0.6 # Based on 75th percentile being around 0.55-0.7 across primes

# Raw Delta Phi Ranges for P9 Rules (Where errors concentrate OUTSIDE P8 Arm ranges)
# These are ranges where errors were observed to cluster when P8 delta_phi_pred_arm failed

# Range 1: Near 0 (0 up to P8 PiHalf Lower Bound)
DELTA_PHI_NEAR_0_UPPER_P9 = DELTA_PHI_CORR_PI_HALF_LOWER_GENERAL # Approx 0.942

# Range 2: Near Pi (P8 PiHalf Upper Bound up to P8 3PiHalf Lower Bound)
DELTA_PHI_NEAR_PI_LOWER_P9 = DELTA_PHI_CORR_PI_HALF_UPPER_GENERAL # Approx 2.199
DELTA_PHI_NEAR_PI_UPPER_P9 = DELTA_PHI_CORR_3PI_HALF_LOWER_GENERAL # Approx 4.084

# Range 3: Near 2pi (P8 3PiHalf Upper Bound up to 2pi)
DELTA_PHI_NEAR_2PI_LOWER_P9 = DELTA_PHI_CORR_3PI_HALF_UPPER_GENERAL # Approx 5.340
DELTA_PHI_NEAR_2PI_UPPER_P9 = 2.0 * np.pi - 1e-9 # Add upper bound near 2pi

# Wider ranges near Pi/2 and 3Pi/2 that *exclude* the strict P8 arm definitions
# These capture points just outside the P8 arm zones where errors can occur
# These are for when the actual target is Pi_Half (checks if raw delta phi is in this wider band)
DELTA_PHI_NEAR_PI_HALF_WIDER_LOWER_P9 = 0.1 * np.pi  # Approx 0.314
DELTA_PHI_NEAR_PI_HALF_WIDER_UPPER_P9 = 0.9 * np.pi  # Approx 2.827
# These are for when the actual target is 3Pi_Half (checks if raw delta phi is in this wider band)
DELTA_PHI_NEAR_3PI_HALF_WIDER_LOWER_P9 = 1.1 * np.pi # Approx 3.456
DELTA_PHI_NEAR_3PI_HALF_WIDER_UPPER_P9 = 1.9 * np.pi # Approx 5.969


# P10 Rule Thresholds (Placeholders for future layers targeting other error types, e.g., Actual T/O -> Pred Arm)
# ZETA_PRIME_ABS_LOW_THRESHOLD_P10 = 0.0
# P_TOTAL_HIGH_MAGNITUDE_P10 = 0.0

# ... (Sections X, XI, XII, etc.) ...


# --- X. File Name Stem Constants ---
RAW_ZEROS_FILENAME_STEM = "zeros6_clean" # Raw input file (CSV)
MASTER_DF_FILENAME_STEM = "master_feature_df" # Output of shared_utils.py (Parquet)

# --- XI. Plotting Default Constants ---
PLT_STYLE = "seaborn-v0_8-darkgrid"
DEFAULT_DPI = 300

# --- XII. Optional Verification Print Block ---
if __name__ == '__main__':
    print("--- constants.py loaded and definitions processed successfully ---")
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"Raw Zeros File expected: {RAW_DATA_DIR / (RAW_ZEROS_FILENAME_STEM + '.csv')}")
    print(f"Master DataFrame to be created: {PROCESSED_DATA_DIR / (MASTER_DF_FILENAME_STEM + '.parquet')}")
    print(f"COL_T: {COL_T}")
    print(f"Primes to process for individual features: {PRIMES_TO_PROCESS}")
    print(f"Feature names for prime 2: {PRIME_FEATURE_NAMES.get(2, 'Not Found')}")
    print(f"Interaction pairs to process: {INTERACTION_PAIRS_TO_PROCESS}")
    if INTERACTION_PAIRS_TO_PROCESS:
        first_pair = INTERACTION_PAIRS_TO_PROCESS[0]
        print(f"Interaction feature names for pair {first_pair}: {PRIME_INTERACTION_FEATURE_NAMES.get(first_pair, 'Not Found')}")
    print(f"Log floor value for log_abs calculations: {LOG_FLOOR_VALUE}")
    print(f"Dot log lookback: {DOT_LOG_LOOKBACK_GENERAL}")
    print(f"All individual prime feature columns defined: {len(ALL_INDIVIDUAL_PRIME_FEATURE_COLUMNS)}")
    print(f"All interaction feature columns defined: {len(ALL_INTERACTION_FEATURE_COLUMNS)}")
    print("--- All constants accessible ---")