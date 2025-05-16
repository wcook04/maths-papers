# Paper_12/scripts/unified_hierarchical_predictor.py

import pandas as pd
import numpy as np
import time
import argparse
from pathlib import Path
import sys
import traceback # To print full tracebacks for errors
# Import sklearn metrics needed for evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support

# --- Setup Paths for Sibling Imports (constants.py) ---
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# --- Import from Local constants.py ---
try:
    import constants
    # Import specific constants for columns, branches, dirs, etc.
    from constants import (
        BASE_DIR, DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, PREDICTOR_RESULTS_DIR_NAME,
        BRANCH_PI_HALF, BRANCH_3PI_HALF, BRANCH_TRANSITION_OTHER, ALL_BRANCHES,
        COL_T, COL_PHASE_BRANCH,
        PRIMES_TO_PROCESS, REFERENCE_PRIME_FOR_INTERACTIONS, INTERACTION_PAIRS_TO_PROCESS,
        PRIME_FEATURE_NAMES, PRIME_INTERACTION_FEATURE_NAMES,
        IMPULSE_ZERO_THRESHOLD_GENERAL, # Keep this, might be useful although rules use sign column
        COL_ACTUAL_NEXT_BRANCH_TARGET, COL_PREDICTED_BASE_C2LAW,
        COL_PREDICTED_FINAL_HIERARCHICAL, COL_CORRECTION_LAYER_APPLIED,
        # COL_P8_HIERARCHICAL_PRED, # We won't use this intermediate column anymore
        MASTER_DF_FILENAME_STEM, # Add this constant import
    )
    print("SUCCESS: unified_hierarchical_predictor.py successfully imported from constants.py")
except ImportError as e:
    print(f"FATAL ERROR: unified_hierarchical_predictor.py could not import from constants.py. Error: {e}")
    print("Please ensure all required constants are defined and correctly placed in constants.py.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during constants import: {e}")
    print(f"Traceback:\n{traceback.format_exc()}")
    sys.exit(1)

# --- Utility Functions (Imported from shared_utils.py) ---
try:
    from shared_utils import (
        save_summary_report, load_dataframe_robust, ensure_dir, save_dataframe_robust
        # Import any other utility functions needed
    )
    print("SUCCESS: unified_hierarchical_predictor.py successfully imported utilities from shared_utils.py")
except ImportError as e:
    print(f"FATAL ERROR: unified_hierarchical_predictor.py could not import utilities from shared_utils.py. Error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during shared_utils import: {e}")
    print(f"Traceback:\n{traceback.format_exc()}")
    sys.exit(1)


# --- Configuration ---
PREDICTOR_RESULTS_DIR = RESULTS_DIR / PREDICTOR_RESULTS_DIR_NAME
ensure_dir(PREDICTOR_RESULTS_DIR) # Ensure results directory exists

# Name for the augmented DataFrame output file
AUGMENTED_DF_FILENAME_STEM = "df_with_unified_predictions" # Use the same name


# --- Helper Function for Accuracy ---
def calculate_accuracy(df, target_col, pred_col):
    """Safely calculates accuracy, handling potential NaNs or missing columns."""
    if target_col not in df or pred_col not in df:
        return 0.0, 0, 0
    valid_df = df.dropna(subset=[target_col, pred_col])
    if valid_df.empty:
        return 0.0, 0, 0
    y_true = valid_df[target_col]
    y_pred = valid_df[pred_col]
    acc = accuracy_score(y_true, y_pred)
    n_total = len(y_true)
    n_correct = int(acc * n_total)
    return acc, n_correct, n_total

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Applies hierarchical prediction rules for Riemann Zeros (Sequential Logic).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--force-repredict",
        action="store_true",
        help=f"Force recomputation of predictions even if {AUGMENTED_DF_FILENAME_STEM}.parquet exists."
    )
    # Add other potential arguments here if needed in the future

    args = parser.parse_args()

    report_lines = []
    predictor_start_time = time.time()
    report_lines.append(f"--- Unified Hierarchical Prediction Started: {time.ctime(predictor_start_time)} ---")
    #report_lines.append(f"Command-line Args: {vars(args)}") # Optional: Keep if useful

    master_df_filepath = PROCESSED_DATA_DIR / f"{constants.MASTER_DF_FILENAME_STEM}.parquet"
    augmented_df_filepath = PREDICTOR_RESULTS_DIR / f"{AUGMENTED_DF_FILENAME_STEM}.parquet"
    summary_output_dir = PREDICTOR_RESULTS_DIR

    df = None # Initialize DataFrame variable

    # --- Step 1: Load Master DataFrame ---
    report_lines.append(f"\nStep 1: Loading Master Feature DataFrame from {master_df_filepath}")
    df = load_dataframe_robust(master_df_filepath, report_list=report_lines)

    if df is None:
        report_lines.append(f"FATAL ERROR: Could not load master DataFrame from {master_df_filepath}. Halting.")
        save_summary_report(report_lines, "predictor_orchestration_log", summary_output_dir)
        sys.exit(1)
    # report_lines.append(f"  Master DataFrame shape: {df.shape}") # Reduce verbosity
    # report_lines.append(f"  Master DataFrame columns ({len(df.columns)}): {df.columns.tolist()}") # Reduce verbosity

    # --- Step 2: Prepare Target Variable and Check Essential Feature Columns ---
    report_lines.append("\nStep 2: Data Preparation")

    # Create target variable if missing (shifted Phase_Branch)
    target_col = constants.COL_ACTUAL_NEXT_BRANCH_TARGET
    if target_col not in df.columns:
         if constants.COL_PHASE_BRANCH in df.columns:
              df[target_col] = df[constants.COL_PHASE_BRANCH].shift(-1)
              # report_lines.append(f"  Target column '{target_col}' created by shifting '{constants.COL_PHASE_BRANCH}'.")
         else:
              report_lines.append(f"FATAL ERROR: Cannot create target column '{target_col}'. Missing source '{constants.COL_PHASE_BRANCH}'. Halting.")
              save_summary_report(report_lines, "predictor_orchestration_log", summary_output_dir)
              sys.exit(1)

    # Define essential feature columns needed for the specific rules
    essential_features = [target_col, constants.COL_PHASE_BRANCH] # Base needs Phase_Branch, Target needed everywhere
    p8_rule_cols = {} # Store required column names for P8-style rules
    missing_p8_rule_cols = []
    for p_corr in [p for p in constants.PRIMES_TO_PROCESS if p != constants.REFERENCE_PRIME_FOR_INTERACTIONS]:
        pair = (constants.REFERENCE_PRIME_FOR_INTERACTIONS, p_corr)
        features = constants.PRIME_INTERACTION_FEATURE_NAMES.get(pair, {})
        impulse_sign_col = features.get("net_impulse_sign")
        delta_phi_pred_arm_col = features.get("delta_phi_pred_arm") # Column derived in oracle

        if impulse_sign_col:
            essential_features.append(impulse_sign_col)
            p8_rule_cols[(p_corr, 'impulse_sign')] = impulse_sign_col
        else: missing_p8_rule_cols.append(f'C{p_corr} impulse sign')

        if delta_phi_pred_arm_col:
            essential_features.append(delta_phi_pred_arm_col)
            p8_rule_cols[(p_corr, 'delta_phi_pred_arm')] = delta_phi_pred_arm_col
        else: missing_p8_rule_cols.append(f'C{p_corr} delta_phi_pred_arm')

    if missing_p8_rule_cols:
         report_lines.append(f"    WARNING: Prediction accuracy may be affected due to missing essential interaction feature columns: {', '.join(missing_p8_rule_cols)}")

    # Drop rows with NaNs in essential columns (especially the shifted target)
    initial_rows = len(df)
    # Check only columns actually present in df to avoid KeyError
    cols_to_check_for_nan = [col for col in essential_features if col in df.columns]
    df.dropna(subset=cols_to_check_for_nan, inplace=True)
    rows_after_nan_drop = len(df)
    dropped_nan_count = initial_rows - rows_after_nan_drop
    report_lines.append(f"  Initial rows: {initial_rows}")
    report_lines.append(f"  Rows after dropping NaNs in required features/target: {rows_after_nan_drop} (dropped {dropped_nan_count})")

    # Ensure target has only valid branches
    df = df[df[target_col].isin(constants.ALL_BRANCHES)].copy()
    rows_after_target_filter = len(df)
    if rows_after_target_filter < rows_after_nan_drop:
        report_lines.append(f"  Filtered {rows_after_nan_drop - rows_after_target_filter} rows with invalid target branches.")
    # report_lines.append(f"  Final data shape for prediction: {df.shape}") # Less verbose

    if df.empty:
        report_lines.append("FATAL ERROR: DataFrame is empty after preparation steps. Halting.")
        save_summary_report(report_lines, "predictor_orchestration_log", summary_output_dir)
        sys.exit(1)

    # --- Step 3: Apply Base C2 Law ---
    report_lines.append("\nStep 3: Implementing Base C2 Law Prediction")
    base_pred_col = constants.COL_PREDICTED_BASE_C2LAW
    df[base_pred_col] = df[constants.COL_PHASE_BRANCH]
    report_lines.append(f"  Base C2 Law predictions applied. Distribution:\n{df[base_pred_col].value_counts(normalize=True).apply(lambda x: f'{x*100:.1f}%').to_string()}")

    # --- Step 4: Initialize Final Predictions ---
    report_lines.append("\nStep 4: Initializing Final Predictions")
    final_pred_col = constants.COL_PREDICTED_FINAL_HIERARCHICAL
    layer_applied_col = constants.COL_CORRECTION_LAYER_APPLIED

    df[final_pred_col] = df[base_pred_col].copy()
    df[layer_applied_col] = "Base_C2Law" # Initialize layer tracker

    # Calculate Initial Accuracy
    base_acc, _, n_total = calculate_accuracy(df, target_col, base_pred_col)
    report_lines.append(f"  Initial Base C2 Law Accuracy: {base_acc * 100:.4f}%")


    # --- Step 5: Implementing Hierarchical Corrections (Sequential Logic) ---
    report_lines.append("\nStep 5: Implementing Hierarchical Corrections")

    # Cache branch names
    t_o = constants.BRANCH_TRANSITION_OTHER
    pi_half = constants.BRANCH_PI_HALF
    three_pi_half = constants.BRANCH_3PI_HALF

    # Iterate through correcting primes sequentially
    for p_corr in [p for p in constants.PRIMES_TO_PROCESS if p != constants.REFERENCE_PRIME_FOR_INTERACTIONS]:
        pair = (constants.REFERENCE_PRIME_FOR_INTERACTIONS, p_corr)
        report_lines.append(f"\n  Applying C{p_corr}-C{constants.REFERENCE_PRIME_FOR_INTERACTIONS} Interactions Correction Layer...")

        # Retrieve cached column names for this prime (or skip if missing)
        impulse_sign_col = p8_rule_cols.get((p_corr, 'impulse_sign'))
        delta_phi_pred_arm_col = p8_rule_cols.get((p_corr, 'delta_phi_pred_arm'))

        if not impulse_sign_col or impulse_sign_col not in df.columns or \
           not delta_phi_pred_arm_col or delta_phi_pred_arm_col not in df.columns:
            report_lines.append(f"    Skipping C{p_corr} due to missing columns.")
            continue

        # Ensure impulse sign is numeric and handle potential NaNs from conversion/load
        try:
            df[impulse_sign_col] = pd.to_numeric(df[impulse_sign_col], errors='coerce').fillna(0) # Treat NaN impulse sign as 0
        except Exception as e:
             report_lines.append(f"      Warning: Could not convert {impulse_sign_col} to numeric: {e}. Skipping rules for C{p_corr}.")
             continue

        # 1. Identify errors *before* this layer's corrections
        errors_before_mask = df[final_pred_col] != df[target_col]
        report_lines.append(f"    Errors from previous layers before C{p_corr} corrections: {errors_before_mask.sum()}")

        corrections_made_this_layer = 0
        corrections_log = {} # Track counts for each specific rule in this layer

        # --- Define Rule Masks (Targeting specific errors based on *current* prediction state) ---

        # Rule: SpineToPiHalf for C{p_corr}
        rule_name_s2p = f"C{p_corr}_SpineToPiHalf"
        mask_s2p = (df[final_pred_col] == t_o) & \
                   (df[target_col] == pi_half) & \
                   (df[impulse_sign_col] == -1) & \
                   (df[delta_phi_pred_arm_col] == pi_half)
        count_s2p = mask_s2p.sum()
        if count_s2p > 0:
            df.loc[mask_s2p, final_pred_col] = pi_half
            df.loc[mask_s2p, layer_applied_col] = rule_name_s2p
            corrections_log[rule_name_s2p] = count_s2p
            corrections_made_this_layer += count_s2p

        # Rule: SpineTo3PiHalf for C{p_corr}
        rule_name_s23p = f"C{p_corr}_SpineTo3PiHalf"
        # Apply only if not already corrected above (shouldn't overlap if rules are mutually exclusive, but safe)
        mask_s23p = (df[final_pred_col] == t_o) & \
                    (df[target_col] == three_pi_half) & \
                    (df[impulse_sign_col] == -1) & \
                    (df[delta_phi_pred_arm_col] == three_pi_half)
        # Ensure we only apply to rows whose layer hasn't been set by this prime yet
        actual_mask_s23p = mask_s23p & (df[layer_applied_col] != rule_name_s2p)
        count_s23p = actual_mask_s23p.sum()
        if count_s23p > 0:
             df.loc[actual_mask_s23p, final_pred_col] = three_pi_half
             df.loc[actual_mask_s23p, layer_applied_col] = rule_name_s23p
             corrections_log[rule_name_s23p] = count_s23p
             corrections_made_this_layer += count_s23p

        # Rule: ArmToSpine for C{p_corr}
        rule_name_a2s = f"C{p_corr}_ArmToSpine"
        is_arm_pred = (df[final_pred_col] == pi_half) | (df[final_pred_col] == three_pi_half)
        # Apply only if not already corrected above by this prime's rules
        mask_a2s = is_arm_pred & \
                   (df[target_col] == t_o) & \
                   (df[impulse_sign_col] == 1)
        # Ensure we only apply to rows whose layer hasn't been set by this prime yet
        actual_mask_a2s = mask_a2s & (~df[layer_applied_col].isin([rule_name_s2p, rule_name_s23p]))
        count_a2s = actual_mask_a2s.sum()
        if count_a2s > 0:
            df.loc[actual_mask_a2s, final_pred_col] = t_o
            df.loc[actual_mask_a2s, layer_applied_col] = rule_name_a2s
            corrections_log[rule_name_a2s] = count_a2s
            corrections_made_this_layer += count_a2s

        # Report rule counts for this layer
        for rule, count in corrections_log.items():
             report_lines.append(f"    Applied {rule.split('_')[1]} rule: {count} corrections.") # More concise rule name

        report_lines.append(f"    Total corrections applied by C{p_corr}-C2 layer: {corrections_made_this_layer}")

        # 2. Calculate and report accuracy *after* this layer
        acc_after_layer, _, _ = calculate_accuracy(df, target_col, final_pred_col)
        report_lines.append(f"    Accuracy after C{p_corr} layer: {acc_after_layer * 100:.4f}%")

    # End of loop over primes


    # --- Step 6: Evaluate Final Hierarchical Predictions ---
    report_lines.append("\nStep 6: Evaluating Final Hierarchical Predictions")

    # Final evaluation uses the iteratively updated final_pred_col
    final_acc, final_correct, final_total = calculate_accuracy(df, target_col, final_pred_col)
    final_errors = final_total - final_correct
    improvement = final_acc - base_acc

    report_lines.append(f"  Overall Final Accuracy: {final_acc * 100:.4f}%")
    report_lines.append(f"  Improvement over Base C2 Law: {improvement * 100:.4f}%")

    # Ensure evaluation data is clean (redundant if calculate_accuracy handles it, but safe)
    df_eval_final = df.dropna(subset=[target_col, final_pred_col]).copy()
    df_eval_final = df_eval_final[df_eval_final[target_col].isin(constants.ALL_BRANCHES)]
    df_eval_final = df_eval_final[df_eval_final[final_pred_col].isin(constants.ALL_BRANCHES)]

    if not df_eval_final.empty:
        y_true_final = df_eval_final[target_col]
        y_pred_final = df_eval_final[final_pred_col]

        report_lines.append("\n  Confusion Matrix (Actual vs. Predicted):")
        cm_final = confusion_matrix(y_true_final, y_pred_final, labels=constants.ALL_BRANCHES)
        # Format CM like the desired output
        cm_header = f"{'Predicted:':<25} | {' | '.join([f'{l:<15}' for l in constants.ALL_BRANCHES])}"
        cm_separator = "-" * len(cm_header)
        report_lines.append(cm_header)
        report_lines.append(cm_separator)
        for i, label in enumerate(constants.ALL_BRANCHES):
            row_str = f"{f'Actual: {label}':<25} | {' | '.join([f'{cm_final[i, j]:<15}' for j in range(len(constants.ALL_BRANCHES))])}"
            report_lines.append(row_str)
        report_lines.append("") # Add space

        class_report_final_str = classification_report(y_true_final, y_pred_final, labels=constants.ALL_BRANCHES, target_names=constants.ALL_BRANCHES, zero_division=0)
        report_lines.append("\n  Classification Report:\n" + class_report_final_str)
    else:
        report_lines.append("\n  No data for Final Hierarchical Law evaluation.")


    # --- Step 7: Correction Layer Breakdown ---
    report_lines.append("\nStep 7: Correction Layer Breakdown")
    report_lines.append("  Prediction Counts and Accuracy by Specific Layer/Rule:")

    layer_stats = []
    # Group by the layer/rule that made the *final* prediction
    grouped_layers = df_eval_final.groupby(layer_applied_col)

    # Header for the table
    header = f"{'Layer/Rule':<18} | {'Count':<7} | {'Accuracy on Subset':<20} | {'Needed Correction Count (from Base Error)':<40}"
    separator = "-" * len(header)
    report_lines.append(header)
    report_lines.append(separator)

    total_prime_layer_points = 0
    total_prime_layer_correct = 0

    for layer_name, group_df in grouped_layers:
        count = len(group_df)
        acc_subset, correct_subset, _ = calculate_accuracy(group_df, target_col, final_pred_col)

        # Calculate how many of these points were *originally* errors in the Base C2 Law prediction
        base_errors_in_group = (group_df[base_pred_col] != group_df[target_col]).sum()
        needed_correction_str = f"{base_errors_in_group}" if layer_name != "Base_C2Law" else "-"

        report_lines.append(f"{layer_name:<18} | {count:<7} | {f'{acc_subset * 100:.4f}%':<20} | {needed_correction_str:<40}")

        # Aggregate stats for prime layers (anything not 'Base_C2Law')
        if layer_name != "Base_C2Law":
            total_prime_layer_points += count
            total_prime_layer_correct += correct_subset

    report_lines.append(separator) # Footer separator

    # Calculate overall accuracy for points assigned to *any* prime layer
    if total_prime_layer_points > 0:
        prime_layer_overall_acc = (total_prime_layer_correct / total_prime_layer_points) * 100
        report_lines.append(f"\n  Overall Accuracy of points *assigned to any prime layer* (C3+): {prime_layer_overall_acc:.4f}% ({total_prime_layer_correct} correct out of {total_prime_layer_points} points assigned to prime layers)")
    else:
        report_lines.append("\n  No points were assigned to prime correction layers.")

    report_lines.append("  Note: 'Accuracy on Subset' for prime layers is expected to be 100.00% if the rules precisely target and correct base errors.")
    report_lines.append("  'Needed Correction Count' shows how many of the points assigned to a prime layer were indeed mispredicted by the Base C2 Law.")


    # --- Step 7b: Analysis of Remaining Errors ---
    report_lines.append("\nStep 7b: Analysis of Remaining Errors")
    errors_df_final = df_eval_final[df_eval_final[final_pred_col] != df_eval_final[target_col]].copy()
    num_remaining_errors = len(errors_df_final)
    percent_remaining_errors = (num_remaining_errors / final_total) * 100 if final_total > 0 else 0

    report_lines.append(f"  Total remaining errors after all layers: {num_remaining_errors} ({percent_remaining_errors:.4f}%)")

    if num_remaining_errors > 0:
        report_lines.append("\n  Distribution of Remaining Errors (Actual Branch vs. Predicted Branch):")
        # Use crosstab for a cleaner confusion matrix of only errors
        error_cm = pd.crosstab(errors_df_final[target_col], errors_df_final[final_pred_col],
                               rownames=['Actual_Next_Branch_Target'],
                               colnames=['Predicted_Next_Branch_Final_Hierarchical'])
        report_lines.append(error_cm.to_string())

        report_lines.append("\n  Remaining Error Breakdown by Initial Base Prediction and Final Correction Layer Applied:")
        # Crosstab: Rows = Final Layer Applied, Columns = Initial Base Prediction (for the error points)
        error_breakdown = pd.crosstab(errors_df_final[layer_applied_col], errors_df_final[base_pred_col],
                                      rownames=['Correction_Layer_Applied'],
                                      colnames=['Predicted_Next_Branch_Base_C2Law'])
        report_lines.append(error_breakdown.to_string())
        report_lines.append("  Note: A non-zero count under a prime layer column here would indicate an error occurred *after* a prime correction was applied.")
        report_lines.append("  A non-zero count under 'Base_C2Law' means the Base prediction was wrong, and no prime layer rule applied to correct it.")

    report_lines.append(f"\n  Remaining errors DataFrame (subset of the augmented DF) can be loaded from {augmented_df_filepath} and filtered for detailed analysis.")


    # --- Step 8: Saving Augmented DataFrame for Error Analysis ---
    report_lines.append(f"\nStep 8: Saving Outputs")

    # Collect ALL potentially relevant columns from constants and dataframe
    # (This section seems okay, re-using the logic from the provided script)
    cols_to_save_set = set()
    essential_output_cols = [
        constants.COL_T, constants.COL_PHASE_BRANCH, target_col, base_pred_col, final_pred_col, layer_applied_col,
        # Add back COL_P8_HIERARCHICAL_PRED if you specifically need the state *after* the old P8 block for comparison,
        # but the sequential logic doesn't use it. For the 87% logic, final_pred_col holds the result.
    ]
    cols_to_save_set.update(essential_output_cols)
    for p in constants.PRIMES_TO_PROCESS:
        if p in constants.PRIME_FEATURE_NAMES: cols_to_save_set.update(constants.PRIME_FEATURE_NAMES[p].values())
    for pair in constants.INTERACTION_PAIRS_TO_PROCESS:
        if pair in constants.PRIME_INTERACTION_FEATURE_NAMES: cols_to_save_set.update(constants.PRIME_INTERACTION_FEATURE_NAMES[pair].values())
    zeta_prime_cols = ['COL_ZETA_PRIME_REAL', 'COL_ZETA_PRIME_IMAG', 'COL_ARG_ZETA_PRIME', 'COL_ABS_ZETA_PRIME', 'COL_ABS_ARG_ZETA_PRIME_MOD_PI']
    for col_name_str in zeta_prime_cols:
        if hasattr(constants, col_name_str): cols_to_save_set.add(getattr(constants, col_name_str))
    ptotal_cols = ['COL_P_TOTAL', 'COL_LOG_ABS_P_TOTAL']
    for col_name_str in ptotal_cols:
         if hasattr(constants, col_name_str): cols_to_save_set.add(getattr(constants, col_name_str))
    if hasattr(constants, 'COL_RATIO_P_C2M1'): cols_to_save_set.add(constants.COL_RATIO_P_C2M1)
    analysis_cols = ['COL_LOG_ABS_C2_M1_BIN', 'COL_PHASE_DIST_0_PI', 'COL_MOD4_INDEX_C2M1', 'COL_DELTA_MOD4_INDEX'] # Add others if needed
    for col_name_str in analysis_cols:
         if hasattr(constants, col_name_str) and getattr(constants, col_name_str) in df.columns: # Check existence
              cols_to_save_set.add(getattr(constants, col_name_str))

    cols_to_save_unique_existing = sorted([col for col in list(cols_to_save_set) if col is not None and col in df.columns])

    # report_lines.append(f"  Attempting to save {len(cols_to_save_unique_existing)} unique columns.") # Less verbose
    # report_lines.append(f"  Columns to be saved: {cols_to_save_unique_existing}") # Less verbose

    # Select and save the DataFrame subset
    try:
        df_to_save = df[cols_to_save_unique_existing]
        save_dataframe_robust(df_to_save, augmented_df_filepath, report_list=report_lines)
        report_lines.append(f"  Saved DataFrame with predictions and key features to: {augmented_df_filepath}") # Add custom message here if needed after successful save
    except KeyError as e:
         report_lines.append(f"  FATAL ERROR saving DataFrame: Column {e} not found. Check column name constants and DataFrame generation.")
         report_lines.append(f"  Traceback:\n{traceback.format_exc()}")
    except Exception as e:
        report_lines.append(f"  FATAL ERROR saving DataFrame: {e}")
        report_lines.append(f"  Traceback:\n{traceback.format_exc()}")


    # --- Final Summary Report ---
    predictor_end_time = time.time()
    total_time = predictor_end_time - predictor_start_time
    report_lines.append(f"\n--- Unified Hierarchical Prediction Finished: {time.ctime(predictor_end_time)} ---")
    report_lines.append(f"Total execution time: {total_time:.2f} seconds.")
    # report_lines.append(f"Final DataFrame Shape (before saving subset): {df.shape}") # Less verbose

    # Save the full predictor log
    save_summary_report(report_lines, "predictor_orchestration_log", summary_output_dir)
    print(f"\nUnified Hierarchical Predictor complete. Log saved to: {summary_output_dir / 'predictor_orchestration_log.txt'}")