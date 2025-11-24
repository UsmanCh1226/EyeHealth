import csv
import os
import time
from constants import LOG_FILE_PATH

class HealthLogger:
    """
    Handles logging of eye health metrics to a CSV file.
    The collected data is designed to be useful for medical consultation.
    """
    def __init__(self, log_file_path=LOG_FILE_PATH):
        self.log_file_path = log_file_path
        self.fieldnames = [
            'timestamp', 
            'blink_count', 
            'time_in_drowsy_sec', 
            'avg_distance_cm', 
            'distance_compliance_pct',
            'twenty_twenty_twenty_compliance_pct'
        ]
        self.initialize_log_file()

    def initialize_log_file(self):
        """Creates the log file with headers if it doesn't exist."""
        if not os.path.exists(self.log_file_path):
            with open(self.log_file_path, mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=self.fieldnames)
                writer.writeheader()
            print(f"Log file initialized at: {self.log_file_path}")

    def log_session_metrics(self, data):
        """
        Appends a new row of data to the CSV log file.
        :param data: Dictionary containing the metric values.
        """
        data['timestamp'] = time.time()
        
        try:
            with open(self.log_file_path, mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=self.fieldnames)
                writer.writerow(data)
            # print(f"Logged data point at {time.strftime('%H:%M:%S', time.localtime(data['timestamp']))}")
        except Exception as e:
            print(f"Error writing to log file: {e}")

    def get_full_log(self):
        """Reads and returns all historical data from the log file."""
        data = []
        if not os.path.exists(self.log_file_path):
            return data
            
        with open(self.log_file_path, mode='r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Convert relevant strings to floats/ints for analysis
                try:
                    row['timestamp'] = float(row['timestamp'])
                    row['blink_count'] = int(row['blink_count'])
                    row['time_in_drowsy_sec'] = float(row['time_in_drowsy_sec'])
                    row['avg_distance_cm'] = float(row['avg_distance_cm'])
                    row['distance_compliance_pct'] = float(row['distance_compliance_pct'])
                    row['twenty_twenty_twenty_compliance_pct'] = float(row['twenty_twenty_twenty_compliance_pct'])
                    data.append(row)
                except ValueError as e:
                    print(f"Skipping malformed row: {row}. Error: {e}")
        return data