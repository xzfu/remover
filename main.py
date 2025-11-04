import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector, Button, TextBox, CheckButtons
from matplotlib.gridspec import GridSpec
import sys
import os
from datetime import datetime

class VibrationDataRemover:
    def __init__(self, csv_path):
        """Initialize the data remover with CSV file"""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        self.csv_path = csv_path
        self.original_data = pd.read_csv(csv_path)
        self.data = self.original_data.copy()
        
        print(f"Loaded: {csv_path}")
        print(f"Columns: {list(self.data.columns)}")
        print(f"Data shape: {self.data.shape}")
        
        # Assume columns are: time, x, y, z (adjust if needed)
        self.time_col = self.data.columns[0]
        self.x_col = self.data.columns[1]
        self.y_col = self.data.columns[2]
        self.z_col = self.data.columns[3]
        
        # For undo functionality
        self.history = []
        
        # Selection range
        self.selected_start = None
        self.selected_end = None
        
        # Scale mode: True = same scale, False = individual scales
        self.same_scale = True
        
        # Create outputs folder if it doesn't exist
        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.setup_plot()
    
    def setup_plot(self):
        """Create the interactive plot interface"""
        # Create figure with GridSpec for layout
        self.fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(4, 3, figure=self.fig, height_ratios=[3, 3, 3, 0.5])
        
        # Three subplots for x, y, z
        self.ax_x = self.fig.add_subplot(gs[0, :])
        self.ax_y = self.fig.add_subplot(gs[1, :], sharex=self.ax_x)
        self.ax_z = self.fig.add_subplot(gs[2, :], sharex=self.ax_x)
        
        # Control panel area
        self.ax_controls = self.fig.add_subplot(gs[3, :])
        self.ax_controls.axis('off')
        
        # Initial plot
        self.plot_data()
        
        # Add SpanSelector to the top plot (works across all due to sharex)
        self.span = SpanSelector(
            self.ax_x,
            self.on_select,
            'horizontal',
            useblit=True,
            props=dict(alpha=0.3, facecolor='red'),
            interactive=True,
            drag_from_anywhere=True
        )
        
        # Add control widgets
        self.setup_controls()
        
        plt.tight_layout()
        plt.show()
    
    def plot_data(self):
        """Plot the three acceleration axes"""
        # Clear previous plots
        self.ax_x.clear()
        self.ax_y.clear()
        self.ax_z.clear()
        
        time = self.data[self.time_col]
        x = self.data[self.x_col]
        y = self.data[self.y_col]
        z = self.data[self.z_col]
        
        # Plot each axis
        self.ax_x.plot(time, x, 'b-', linewidth=0.5)
        self.ax_y.plot(time, y, 'g-', linewidth=0.5)
        self.ax_z.plot(time, z, 'r-', linewidth=0.5)
        
        # Set y-scale based on mode
        if self.same_scale:
            # Same scale for all plots
            all_values = pd.concat([x, y, z])
            y_min, y_max = all_values.min(), all_values.max()
            margin = (y_max - y_min) * 0.1
            
            self.ax_x.set_ylim(y_min - margin, y_max + margin)
            self.ax_y.set_ylim(y_min - margin, y_max + margin)
            self.ax_z.set_ylim(y_min - margin, y_max + margin)
        else:
            # Individual scales for best fit
            for ax, data_series in [(self.ax_x, x), (self.ax_y, y), (self.ax_z, z)]:
                y_min, y_max = data_series.min(), data_series.max()
                margin = (y_max - y_min) * 0.1
                ax.set_ylim(y_min - margin, y_max + margin)
        
        # Labels
        self.ax_x.set_ylabel('X Acceleration', fontsize=10)
        self.ax_y.set_ylabel('Y Acceleration', fontsize=10)
        self.ax_z.set_ylabel('Z Acceleration', fontsize=10)
        self.ax_z.set_xlabel('Time', fontsize=10)
        
        scale_mode = "Same Scale" if self.same_scale else "Individual Scales"
        self.ax_x.set_title(f'X-Axis Vibration ({scale_mode})', fontsize=11)
        self.ax_y.set_title('Y-Axis Vibration', fontsize=11)
        self.ax_z.set_title('Z-Axis Vibration', fontsize=11)
        
        self.ax_x.grid(True, alpha=0.3)
        self.ax_y.grid(True, alpha=0.3)
        self.ax_z.grid(True, alpha=0.3)
        
        self.fig.canvas.draw_idle()
    
    def setup_controls(self):
        """Setup control buttons and text boxes"""
        # Button positions (left, bottom, width, height)
        btn_remove_ax = plt.axes([0.05, 0.02, 0.08, 0.04])
        btn_undo_ax = plt.axes([0.14, 0.02, 0.08, 0.04])
        btn_save_ax = plt.axes([0.23, 0.02, 0.08, 0.04])
        btn_reset_ax = plt.axes([0.32, 0.02, 0.08, 0.04])
        
        # Checkbox for scale toggle (next to buttons)
        checkbox_ax = plt.axes([0.42, 0.02, 0.1, 0.04])
        
        # Text boxes for precise input
        txt_start_ax = plt.axes([0.62, 0.02, 0.12, 0.04])
        txt_end_ax = plt.axes([0.78, 0.02, 0.12, 0.04])
        
        # Create buttons
        self.btn_remove = Button(btn_remove_ax, 'Remove', color='lightcoral')
        self.btn_undo = Button(btn_undo_ax, 'Undo', color='lightyellow')
        self.btn_save = Button(btn_save_ax, 'Save', color='lightgreen')
        self.btn_reset = Button(btn_reset_ax, 'Reset', color='lightblue')
        
        # Create text boxes
        self.txt_start = TextBox(txt_start_ax, 'Start: ', initial='')
        self.txt_end = TextBox(txt_end_ax, 'End: ', initial='')
        
        # Create checkbox for scale toggle
        self.check_scale = CheckButtons(checkbox_ax, ['Same Scale'], [self.same_scale])
        
        # Connect callbacks
        self.btn_remove.on_clicked(self.remove_range)
        self.btn_undo.on_clicked(self.undo_removal)
        self.btn_save.on_clicked(self.save_data)
        self.btn_reset.on_clicked(self.reset_data)
        self.txt_start.on_submit(self.update_start)
        self.txt_end.on_submit(self.update_end)
        self.check_scale.on_clicked(self.toggle_scale)
    
    def toggle_scale(self, label):
        """Toggle between same scale and individual scales"""
        self.same_scale = not self.same_scale
        mode = "same scale" if self.same_scale else "individual scales"
        print(f"Scale mode: {mode}")
        self.plot_data()
    
    def on_select(self, xmin, xmax):
        """Callback when range is selected with SpanSelector"""
        self.selected_start = xmin
        self.selected_end = xmax
        self.txt_start.set_val(f'{xmin:.4f}')
        self.txt_end.set_val(f'{xmax:.4f}')
        print(f"Selected range: {xmin:.4f} to {xmax:.4f}")
    
    def update_start(self, text):
        """Update start time from text box"""
        try:
            self.selected_start = float(text)
        except ValueError:
            print("Invalid start time")
    
    def update_end(self, text):
        """Update end time from text box"""
        try:
            self.selected_end = float(text)
        except ValueError:
            print("Invalid end time")
    
    def find_nearest_index(self, time_value):
        """Find the index of the nearest actual data point to given time value"""
        time_array = self.data[self.time_col].values
        idx = np.abs(time_array - time_value).argmin()
        return self.data.index[idx]
    
    def remove_range(self, event):
        """Remove selected time range and close gaps"""
        if self.selected_start is None or self.selected_end is None:
            print("Please select a range first!")
            return
        
        start = min(self.selected_start, self.selected_end)
        end = max(self.selected_start, self.selected_end)
        
        # Find nearest actual data points
        start_idx = self.find_nearest_index(start)
        end_idx = self.find_nearest_index(end)
        
        # Get actual time values from the data
        actual_start = self.data.loc[start_idx, self.time_col]
        actual_end = self.data.loc[end_idx, self.time_col]
        
        print(f"Selected: {start:.4f} to {end:.4f}")
        print(f"Nearest actual points: {actual_start:.6f} to {actual_end:.6f}")
        
        # Save current state for undo
        self.history.append(self.data.copy())
        
        # Find indices in range (inclusive of start and end points)
        mask = (self.data[self.time_col] >= actual_start) & (self.data[self.time_col] <= actual_end)
        indices_to_remove = self.data[mask].index
        
        if len(indices_to_remove) == 0:
            print("No data in selected range")
            return
        
        # Calculate exact time gap from actual data points
        time_gap = actual_end - actual_start
        
        # Get the index just after the removed range for adjustment reference
        last_removed_idx = indices_to_remove[-1]
        
        # Remove the range
        self.data = self.data[~mask].copy()
        
        # Adjust all subsequent timestamps to maintain even spacing
        # All points after the removed range are shifted back by the time_gap
        subsequent_mask = self.data.index > last_removed_idx
        self.data.loc[subsequent_mask, self.time_col] -= time_gap
        
        # Reset index
        self.data.reset_index(drop=True, inplace=True)
        
        print(f"Removed {len(indices_to_remove)} points")
        print(f"Time gap closed: {time_gap:.6f} (based on actual data points)")
        
        # Calculate and display sampling rate info
        if len(self.data) > 1:
            time_diffs = np.diff(self.data[self.time_col].values)
            avg_sampling_interval = np.mean(time_diffs)
            sampling_rate = 1.0 / avg_sampling_interval if avg_sampling_interval > 0 else 0
            print(f"Avg sampling interval: {avg_sampling_interval:.6f}s ({sampling_rate:.2f} Hz)")
        
        # Clear selection
        self.selected_start = None
        self.selected_end = None
        self.txt_start.set_val('')
        self.txt_end.set_val('')
        
        # Replot
        self.plot_data()
    
    def undo_removal(self, event):
        """Undo last removal"""
        if len(self.history) > 0:
            self.data = self.history.pop()
            print("Undone last removal")
            self.plot_data()
        else:
            print("Nothing to undo")
    
    def reset_data(self, event):
        """Reset to original data"""
        self.data = self.original_data.copy()
        self.history = []
        print("Reset to original data")
        self.plot_data()
    
    def save_data(self, event):
        """Save cleaned data to outputs folder"""
        # Generate new filename with timestamp
        base_name = os.path.splitext(os.path.basename(self.csv_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{base_name}_cleaned_{timestamp}.csv"
        output_path = os.path.join(self.output_dir, output_filename)
        
        self.data.to_csv(output_path, index=False)
        print(f"✅ Saved cleaned data to: {output_path}")
        print(f"Original file unchanged: {self.csv_path}")
        print("Timestamps have been adjusted in the saved file")


def main():
    """Main function to handle file input"""
    # Check if file provided as command line argument
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        # Prompt for file path
        print("=" * 60)
        print("Vibration Data Anomaly Remover")
        print("=" * 60)
        csv_file = input("\nEnter the path to your CSV file: ").strip()
        
        # Remove quotes if user pasted path with quotes
        csv_file = csv_file.strip('"').strip("'")
    
    try:
        remover = VibrationDataRemover(csv_file)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease check the file path and try again.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()