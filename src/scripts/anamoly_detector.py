import pandas as pd
import numpy as np

def generate_synthetic_data(num_samples=1000, seed=42):
    """
    Generates synthetic time-series data for Blinks, Drowsiness, and Distance.
    Simulates normal behavior with a clear anomaly period (high-risk event).
    """
    np.random.seed(seed)
    
    # Time Index (e.g., in seconds)
    time = np.arange(num_samples)
    
    # Baseline Metrics (Normal State)
    blinks = np.clip(5 + 0.5 * np.sin(time / 50) + np.random.randn(num_samples) * 0.5, 3, 10)
    drowsiness = np.clip(0.1 + 0.05 * np.random.randn(num_samples), 0.05, 0.4) # Scale from 0 to 1
    distance = np.clip(60 + 2 * np.random.randn(num_samples), 55, 65) # Distance in cm

    # Induce an Anomaly (e.g., from sample 700 to 800)
    anomaly_start, anomaly_end = 700, 800
    anomaly_range = range(anomaly_start, anomaly_end)

    # Anomaly 1: User is tired (low blinks, high drowsiness)
    blinks[anomaly_range] = np.clip(blinks[anomaly_range] - 4, 1, 5) # Blinks drop significantly
    drowsiness[anomaly_range] = np.clip(drowsiness[anomaly_range] + 0.7, 0.6, 1.0) # Drowsiness spikes
    distance[anomaly_range] = np.clip(distance[anomaly_range] + 5, 65, 75) # Distance slightly increases (leaning back)

    data = pd.DataFrame({
        'Timestamp': time,
        'Blinks_per_min': blinks,
        'Drowsiness_Score': drowsiness, # 0.0 (Alert) to 1.0 (Drowsy)
        'Distance_cm': distance
    })
    
    # Convert 'Timestamp' to a timedelta for better representation
    data['Time'] = pd.to_timedelta(data['Timestamp'], unit='s')
    data = data.drop(columns=['Timestamp'])
    
    return data

def apply_zscore_normalization(series):
    """Calculates the Z-score for a series to standardize it."""
    return (series - series.mean()) / series.std()

def flag_high_risk_events(df, threshold_blinks_sd=-2.0, threshold_drowsiness_sd=2.0, min_flags=2):
    """
    Analyzes time-series data using Z-scores to flag simultaneous anomalies.

    Args:
        df (pd.DataFrame): DataFrame containing the time-series data.
        threshold_blinks_sd (float): Z-score threshold for low blinks.
        threshold_drowsiness_sd (float): Z-score threshold for high drowsiness.
        min_flags (int): Minimum number of concurrent flags needed to trigger a HIGH_RISK alert.
    
    Returns:
        pd.DataFrame: Original DataFrame with 'Is_Blink_Anomaly', 'Is_Drowsiness_Anomaly', and 'Risk_Level' columns added.
    """
    print("--- 1. Data Normalization and Flagging ---")
    
    # 1. Normalize (Calculate Z-scores)
    df['Z_Blinks'] = apply_zscore_normalization(df['Blinks_per_min'])
    df['Z_Drowsiness'] = apply_zscore_normalization(df['Drowsiness_Score'])
    
    # 2. Define Anomaly Flags (Boolean Series)
    # Low blinks (below 2.0 standard deviations of the mean is a flag)
    df['Is_Blink_Anomaly'] = df['Z_Blinks'] < threshold_blinks_sd
    # High drowsiness (above 2.0 standard deviations of the mean is a flag)
    df['Is_Drowsiness_Anomaly'] = df['Z_Drowsiness'] > threshold_drowsiness_sd
    
    # 3. Combine Flags for High-Risk Event Detection
    df['Total_Flags'] = df['Is_Blink_Anomaly'].astype(int) + df['Is_Drowsiness_Anomaly'].astype(int)
    
    # 4. Assign Risk Level based on concurrent flags
    df['Risk_Level'] = 'LOW'
    df.loc[df['Total_Flags'] >= min_flags, 'Risk_Level'] = 'HIGH'
    df.loc[df['Total_Flags'] == 1, 'Risk_Level'] = 'MEDIUM'

    return df

def summarize_high_risk_periods(df):
    """
    Identifies and summarizes continuous periods flagged as HIGH_RISK.
    """
    print("\n--- 2. Summarizing High-Risk Periods ---")
    
    high_risk_df = df[df['Risk_Level'] == 'HIGH']
    
    if high_risk_df.empty:
        print("No continuous HIGH_RISK periods detected based on thresholds.")
        return []

    # Identify changes in the 'Risk_Level' status
    is_high_risk = high_risk_df['Risk_Level'] == 'HIGH'
    group_ids = is_high_risk.cumsum()

    # Find the start and end of each continuous block
    summary = []
    for group_id, group in high_risk_df.groupby(group_ids):
        if not group.empty:
            start_time = group['Time'].iloc[0]
            end_time = group['Time'].iloc[-1]
            duration = end_time - start_time
            
            # Use the mean values during the anomaly period
            avg_blinks = group['Blinks_per_min'].mean()
            avg_drowsiness = group['Drowsiness_Score'].mean()
            
            summary.append({
                'Start_Time': str(start_time),
                'End_Time': str(end_time),
                'Duration': str(duration),
                'Avg_Blinks': f"{avg_blinks:.2f}",
                'Avg_Drowsiness': f"{avg_drowsiness:.2f}",
                'Reason': 'Simultaneous low blink rate and high drowsiness score.'
            })

    return summary


if __name__ == '__main__':
    # 1. Generate Data
    raw_data = generate_synthetic_data(num_samples=1000)

    # 2. Run the Anomaly Detector
    analyzed_data = flag_high_risk_events(
        raw_data.copy(),
        # Tweak these thresholds to make the detection more or less sensitive
        threshold_blinks_sd=-1.5,   # Flag if Blinks are 1.5 SD below mean
        threshold_drowsiness_sd=1.8, # Flag if Drowsiness is 1.8 SD above mean
        min_flags=2 # Require both flags to be HIGH_RISK
    )

    # 3. Summarize and Print Results
    risk_summary = summarize_high_risk_periods(analyzed_data)

    print("\n===============================================")
    print("      Eye Health Automated Risk Report         ")
    print("===============================================")

    if risk_summary:
        print(f"Total High-Risk Periods Detected: {len(risk_summary)}")
        for i, period in enumerate(risk_summary):
            print(f"\n--- Incident {i+1} ---")
            print(f"Time Range: {period['Start_Time']} to {period['End_Time']}")
            print(f"Duration: {period['Duration']}")
            print(f"Conditions: Blinks: {period['Avg_Blinks']} | Drowsiness: {period['Avg_Drowsiness']}")
            print(f"Reason: {period['Reason']}")
            
        print("\nACTION RECOMMENDED: Review the data around these timestamps for intervention.")
    else:
        print("System operating normally. No high-risk events detected.")

    # Optional: Save the full flagged dataset for visualization or review
    # analyzed_data.to_csv("eye_tracking_analysis_report.csv", index=False)
    print("\nFull dataset analysis complete (including Z-scores and Risk_Level).")