import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import os
from datetime import datetime

# Assuming logger.py and constants.py are in the same directory
from logger import HealthLogger
# *** FIX: Imported LOGGING_INTERVAL_SEC from constants.py ***
from constants import LOG_FILE_PATH, LOGGING_INTERVAL_SEC

# Initialize the logger (only used to read data)
logger = HealthLogger(LOG_FILE_PATH)

class HealthDashboard:
    """
    A Tkinter application that loads, processes, and displays eye health data
    from the CSV log file using Matplotlib for visualization.
    """
    def __init__(self, master):
        self.master = master
        master.title("Eye Health Doctor Dashboard")
        master.geometry("1000x750")
        
        self.data = pd.DataFrame()
        
        # Configure Grid Layout
        master.grid_rowconfigure(0, weight=0) # Title row
        master.grid_rowconfigure(1, weight=1) # Main content row (frames)
        master.grid_columnconfigure(0, weight=1) 
        master.grid_columnconfigure(1, weight=3) 

        # --- Styling ---
        style = ttk.Style()
        style.configure('TFrame', background='#f4f7f6')
        style.configure('TLabel', background='#f4f7f6', foreground='#333333', font=('Arial', 10))
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), foreground='#0056b3')
        style.configure('Metric.TLabel', font=('Arial', 14, 'bold'), foreground='#0056b3')
        style.configure('Info.TLabel', font=('Arial', 10, 'italic'), foreground='#777777')
        
        # --- Title and Reload Button ---
        title_frame = ttk.Frame(master, padding="10 10 10 10")
        title_frame.grid(row=0, column=0, columnspan=2, sticky="ew")
        
        ttk.Label(title_frame, text="Eye Strain & Fatigue Analysis Dashboard", style='Title.TLabel').pack(side=tk.LEFT, padx=10)
        ttk.Button(title_frame, text="Reload Data", command=self.load_data, cursor="hand2").pack(side=tk.RIGHT, padx=10)

        # --- Left Panel (Key Metrics and Doctor's Notes) ---
        self.metrics_frame = ttk.Frame(master, padding="10", style='TFrame')
        self.metrics_frame.grid(row=1, column=0, sticky="nsew")
        self.metrics_frame.grid_rowconfigure(5, weight=1) # Notes area expands
        
        ttk.Label(self.metrics_frame, text="Key Health Indicators (Long-Term)", style='Metric.TLabel').grid(row=0, column=0, sticky="w", pady=10)
        
        # Variables to hold metric values
        self.compliance_var = tk.StringVar(value="--")
        self.drowsy_var = tk.StringVar(value="--")
        self.blinkrate_var = tk.StringVar(value="--")
        self.sessions_var = tk.StringVar(value="--")

        self.create_metric_label(self.metrics_frame, "Avg Distance Compliance:", self.compliance_var, 1)
        self.create_metric_label(self.metrics_frame, "Total Drowsiness Time (min):", self.drowsy_var, 2)
        self.create_metric_label(self.metrics_frame, "Overall Avg Blink Rate (BPM):", self.blinkrate_var, 3)
        self.create_metric_label(self.metrics_frame, "Data Points/Sessions:", self.sessions_var, 4)

        # Doctor's Actionable Insight Section
        ttk.Label(self.metrics_frame, text="Actionable Insights for Doctor", style='Metric.TLabel').grid(row=5, column=0, sticky="nw", pady=(20, 5))
        self.insight_text = tk.Text(self.metrics_frame, height=15, width=40, wrap=tk.WORD, font=('Arial', 10))
        self.insight_text.grid(row=6, column=0, sticky="nsew", padx=5)
        self.create_insights()


        # --- Right Panel (Visualizations) ---
        self.chart_frame = ttk.Frame(master, padding="10", style='TFrame')
        self.chart_frame.grid(row=1, column=1, sticky="nsew")
        self.chart_frame.grid_columnconfigure(0, weight=1)
        self.chart_frame.grid_rowconfigure(0, weight=1)
        self.chart_frame.grid_rowconfigure(1, weight=1)

        # Matplotlib Figure for charts
        self.fig, self.axes = plt.subplots(2, 1, figsize=(6, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=0, rowspan=2, sticky="nsew")
        
        self.load_data() # Initial data load

    def create_metric_label(self, parent, label_text, var, row):
        """Helper to create consistent metric display labels."""
        ttk.Label(parent, text=label_text, font=('Arial', 10, 'bold')).grid(row=row, column=0, sticky="w", padx=5, pady=2)
        ttk.Label(parent, textvariable=var, style='Metric.TLabel').grid(row=row, column=0, sticky="e", padx=5, pady=2)
        
    def create_insights(self):
        """Generates the static text for doctor consultation guidance."""
        insights = (
            "This data provides a 3-point assessment of eye health risk:\n\n"
            "1. **Average Blink Rate (BPM):** The typical healthy range is 10-20 BPM. "
            "A rate significantly below 10 suggests reduced tear film stability and dry eye risk.\n\n"
            "2. **Drowsiness Time:** High values indicate excessive ocular fatigue or potential sleep deprivation. "
            "This suggests the need for forced, longer breaks.\n\n"
            "3. **Distance Compliance:** A score below 80% means the user is consistently working too close (Myopia Risk). "
            "Recommend professional assessment for proper ergonomic setup and lens correction."
        )
        self.insight_text.delete(1.0, tk.END)
        self.insight_text.insert(tk.END, insights)


    def load_data(self):
        """Loads data from the CSV file and triggers the update."""
        if not os.path.exists(LOG_FILE_PATH):
            messagebox.showerror("Error", f"Log file not found: {LOG_FILE_PATH}.\nPlease run 'eye_monitor_with_log.py' first to generate data.")
            self.sessions_var.set("0 (No Data)")
            return

        try:
            # Read data using the logger's function
            raw_data = logger.get_full_log()
            if not raw_data:
                 self.data = pd.DataFrame() # Empty dataframe
                 messagebox.showinfo("Info", "Log file is empty. Please run the monitor to log data.")
                 self.sessions_var.set("0 (Empty)")
                 return
                 
            self.data = pd.DataFrame(raw_data)
            
            # Convert timestamp to human-readable datetime for plotting
            self.data['datetime'] = self.data['timestamp'].apply(lambda x: datetime.fromtimestamp(x))

            self.update_metrics()
            self.plot_charts()

        except Exception as e:
            messagebox.showerror("Data Error", f"Could not process log file: {e}")
            print(f"Detailed Error: {e}")

    def update_metrics(self):
        """Calculates and updates the summary metrics in the left panel."""
        if self.data.empty:
            return

        total_sessions = len(self.data)
        
        # 1. Compliance (Distance and 20-20-20)
        avg_dist_compliance = self.data['distance_compliance_pct'].mean()
        avg_202020_compliance = self.data['twenty_twenty_twenty_compliance_pct'].mean()
        combined_compliance = (avg_dist_compliance * 0.6) + (avg_202020_compliance * 0.4) # Weighted avg
        self.compliance_var.set(f"{combined_compliance:.1f}%")

        # 2. Drowsiness Time
        total_drowsy_sec = self.data['time_in_drowsy_sec'].sum()
        self.drowsy_var.set(f"{total_drowsy_sec / 60:.1f} minutes")

        # 3. Overall Avg Blink Rate (calculate average blinks per minute)
        # Average Blink Rate = (Avg Blinks per Interval / Interval Duration in minutes)
        avg_interval_blink = self.data['blink_count'].mean()
        
        # Use LOGGING_INTERVAL_SEC to calculate BPM
        # BPM = Blinks / (Interval Duration in Seconds / 60)
        avg_bpm = avg_interval_blink / (LOGGING_INTERVAL_SEC / 60)
        
        self.blinkrate_var.set(f"{avg_bpm:.1f} BPM")
        
        self.sessions_var.set(f"{total_sessions} ({self.data['datetime'].iloc[0].strftime('%Y-%m-%d %H:%M')} to {self.data['datetime'].iloc[-1].strftime('%Y-%m-%d %H:%M')})")


    def plot_charts(self):
        """Generates and updates the two Matplotlib time-series charts."""
        if self.data.empty:
            for ax in self.axes:
                ax.clear()
                ax.text(0.5, 0.5, "No Data to Display", ha='center', va='center', fontsize=12, color='gray')
            self.canvas.draw()
            return
            
        # --- Chart 1: Distance and 20-20-20 Compliance (%) ---
        ax1 = self.axes[0]
        ax1.clear()
        
        # Plot Distance Compliance
        ax1.plot(self.data['datetime'], self.data['distance_compliance_pct'], label='Distance Compliance (%)', color='blue', linewidth=2)
        # Plot 20-20-20 Compliance
        ax1.plot(self.data['datetime'], self.data['twenty_twenty_twenty_compliance_pct'], label='20-20-20 Compliance (%)', color='orange', linestyle='--', linewidth=1)
        
        ax1.set_title('Compliance Scores Over Time', fontsize=12)
        ax1.set_ylabel('Compliance (%)', fontsize=10)
        ax1.legend(loc='lower left', fontsize=8)
        ax1.grid(True, linestyle=':', alpha=0.6)
        
        # Add a healthy threshold line (80%)
        ax1.axhline(y=80, color='red', linestyle='-', alpha=0.7, label='Healthy Goal (80%)')
        
        # Format X-axis for better time display
        self.fig.autofmt_xdate()

        # --- Chart 2: Blink Rate (BPM) and Drowsiness Time (Seconds) ---
        ax2 = self.axes[1]
        ax2.clear()
        
        # Calculate Instantaneous Blink Rate for the interval
        # Instant BPM = Blinks in Interval / (Interval Duration in Seconds / 60)
        instant_bpm = self.data['blink_count'] / (LOGGING_INTERVAL_SEC / 60)
        
        # Plot Blink Rate (Left Y-axis)
        ax2.plot(self.data['datetime'], instant_bpm, label='Blink Rate (BPM)', color='green', linewidth=2)
        ax2.set_ylabel('Blink Rate (BPM)', color='green', fontsize=10)
        ax2.tick_params(axis='y', labelcolor='green')
        
        # Create a second Y-axis for Drowsiness Time
        ax3 = ax2.twinx()
        ax3.bar(self.data['datetime'], self.data['time_in_drowsy_sec'], label='Drowsy Time (s)', color='red', alpha=0.3, width=0.005)
        ax3.set_ylabel('Drowsy Time (s)', color='red', fontsize=10)
        ax3.tick_params(axis='y', labelcolor='red')

        ax2.set_title('Blink Rate and Drowsiness Over Time', fontsize=12)
        ax2.set_xlabel('Time', fontsize=10)
        ax2.grid(True, linestyle=':', alpha=0.6)
        
        # Add healthy range for BPM
        ax2.axhspan(10, 20, color='green', alpha=0.1, label='Healthy BPM Range (10-20)')

        # Draw and update the canvas
        self.canvas.draw()

# --- RUN APPLICATION ---
if __name__ == '__main__':
    # We must ensure LOGGING_INTERVAL_SEC is available before running the class
    try:
        # Check if the constant is loaded (it should be via the import above)
        print(f"Using LOGGING_INTERVAL_SEC: {LOGGING_INTERVAL_SEC} seconds.")
        root = tk.Tk()
        app = HealthDashboard(root)
        root.mainloop()
    except NameError:
        print("FATAL ERROR: LOGGING_INTERVAL_SEC could not be found. Ensure constants.py is correct.")