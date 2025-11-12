import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector, Button, TextBox, CheckButtons, RadioButtons
from matplotlib.gridspec import GridSpec
from scipy import signal
import sys
import os
from datetime import datetime
from collections import deque

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
        
        # First column is time, rest are data columns
        self.time_col = self.data.columns[0]
        self.data_cols = list(self.data.columns[1:])
        self.num_plots = len(self.data_cols)
        
        if self.num_plots == 0:
            raise ValueError("CSV file must have at least 2 columns (time + data)")
        
        print(f"Time column: {self.time_col}")
        print(f"Data columns: {self.data_cols}")
        print(f"Number of plots: {self.num_plots}")
        
        # Convert to NumPy arrays for faster operations
        self.time_array = None
        self.data_arrays = {}
        self._convert_to_numpy()
        
        # For undo functionality - limit history to save memory
        self.history = deque(maxlen=10)
        
        # Selection range
        self.selected_start = None
        self.selected_end = None
        
        # Scale mode: True = same scale, False = individual scales
        self.same_scale = False
        
        # View mode: 'time' or 'fft'
        self.view_mode = 'time'
        
        # FFT comparison mode: 'current' or 'original'
        self.fft_compare_mode = 'current'
        
        # FFT y-axis scale mode: True = logarithmic, False = linear
        self.fft_log_scale = True
        
        # Downsample mode for display - more options with lower values
        self.downsample_mode = 'Auto'
        self.downsample_options = [
            'All',
            'Auto',
            '1k',
            '5k', 
            '10k',
            '25k',
            '50k',
            '100k',
            '250k',
            '500k'
        ]
        
        # Cache for FFT data to maintain consistent scaling
        self.fft_cache = {
            'original': None,
            'current': None
        }
        
        # Calculate sampling rate
        self.calculate_sampling_rate()
        
        # Create outputs folder if it doesn't exist
        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.setup_plot()
    
    def _convert_to_numpy(self):
        """Convert DataFrame columns to NumPy arrays for faster access"""
        self.time_array = self.data[self.time_col].values
        for col in self.data_cols:
            self.data_arrays[col] = self.data[col].values
    
    def _get_downsample_stride(self, data_length, for_fft=False):
        """Calculate stride for downsampling based on mode"""
        # FFT always uses full resolution data
        if for_fft:
            return 1
            
        if self.downsample_mode == 'All':
            return 1
        elif self.downsample_mode == 'Auto':
            # Auto: prefer higher downsampling for better performance
            # Aim for 10k-25k points depending on data size
            if data_length > 500000:
                target = 10000
            elif data_length > 100000:
                target = 25000
            else:
                target = 50000
            stride = max(1, data_length // target)
            return stride
        elif self.downsample_mode == '1k':
            return max(1, data_length // 1000)
        elif self.downsample_mode == '5k':
            return max(1, data_length // 5000)
        elif self.downsample_mode == '10k':
            return max(1, data_length // 10000)
        elif self.downsample_mode == '25k':
            return max(1, data_length // 25000)
        elif self.downsample_mode == '50k':
            return max(1, data_length // 50000)
        elif self.downsample_mode == '100k':
            return max(1, data_length // 100000)
        elif self.downsample_mode == '250k':
            return max(1, data_length // 250000)
        elif self.downsample_mode == '500k':
            return max(1, data_length // 500000)
        return 1
    
    def _downsample_for_display(self, time_array, data_array):
        """Downsample data for display only - preserves original data"""
        stride = self._get_downsample_stride(len(time_array))
        if stride > 1:
            return time_array[::stride], data_array[::stride]
        return time_array, data_array
    
    def calculate_sampling_rate(self):
        """Calculate the sampling rate from the data"""
        if len(self.data) > 1:
            time_diffs = np.diff(self.time_array[:min(1000, len(self.time_array))])  # Use first 1000 points
            avg_interval = np.mean(time_diffs)
            self.sampling_rate = 1.0 / avg_interval if avg_interval > 0 else 1000.0
        else:
            self.sampling_rate = 1000.0  # Default fallback
        print(f"Sampling rate: {self.sampling_rate:.2f} Hz")
    
    def setup_plot(self):
        """Create the interactive plot interface"""
        # Create figure with GridSpec for layout
        # Height ratios: equal for all data plots + smaller control panel
        height_ratios = [3] * self.num_plots + [0.5]
        
        self.fig = plt.figure(figsize=(14, 3 + 3 * self.num_plots))
        # Width ratios: 3 columns for plots, 1 for radio buttons (only visible in time view)
        gs = GridSpec(self.num_plots + 1, 4, figure=self.fig, height_ratios=height_ratios, width_ratios=[3, 3, 3, 0.8])
        
        # Create subplots dynamically based on number of data columns
        self.axes = []
        for i in range(self.num_plots):
            if i == 0:
                ax = self.fig.add_subplot(gs[i, :3])
            else:
                ax = self.fig.add_subplot(gs[i, :3], sharex=self.axes[0])
            self.axes.append(ax)
        
        # Control panel area (spans all 4 columns)
        self.ax_controls = self.fig.add_subplot(gs[self.num_plots, :])
        self.ax_controls.axis('off')
        
        # Radio buttons area for downsampling (on the right side, only for time view)
        self.ax_radio = self.fig.add_subplot(gs[:self.num_plots, 3])
        self.ax_radio.axis('off')
        
        # Add SpanSelector to all plots
        self.span_selectors = []
        for ax in self.axes:
            span = SpanSelector(
                ax,
                self.on_select,
                'horizontal',
                useblit=True,
                props=dict(alpha=0.3, facecolor='red'),
                interactive=True,
                drag_from_anywhere=True
            )
            self.span_selectors.append(span)
        
        # Plot the data
        self.plot_data()
        
        # Add control widgets
        self.setup_controls()
        
        plt.tight_layout()
        plt.show()
    
    def plot_data(self):
        """Plot the data based on current view mode"""
        if self.view_mode == 'time':
            self.plot_time_domain()
        else:
            self.plot_fft()
    
    def plot_time_domain(self):
        """Plot all data columns in time domain"""
        # Clear previous plots
        for ax in self.axes:
            ax.clear()
        
        # Define colors for plots
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
        
        # Calculate stride for downsampling
        stride = self._get_downsample_stride(len(self.time_array))
        display_time = self.time_array[::stride]
        
        # Collect all data for scale calculation (use full data for accurate min/max)
        if self.same_scale:
            all_min = min(self.data_arrays[col].min() for col in self.data_cols)
            all_max = max(self.data_arrays[col].max() for col in self.data_cols)
            margin = (all_max - all_min) * 0.1 if all_max != all_min else 0.1
        
        # Plot each data column
        for i, (ax, col) in enumerate(zip(self.axes, self.data_cols)):
            color = colors[i % len(colors)]
            
            # Downsample for display
            display_data = self.data_arrays[col][::stride]
            
            # Plot
            ax.plot(display_time, display_data, color=color, linewidth=0.5)
            
            # Set y-scale based on mode
            if self.same_scale:
                ax.set_ylim(all_min - margin, all_max + margin)
            else:
                # Individual scale for this plot (use full data for min/max)
                y_min, y_max = self.data_arrays[col].min(), self.data_arrays[col].max()
                margin = (y_max - y_min) * 0.1 if y_max != y_min else 0.1
                ax.set_ylim(y_min - margin, y_max + margin)
            
            # Labels
            ax.set_ylabel(col, fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Title for first plot
            if i == 0:
                scale_mode = "Same Scale" if self.same_scale else "Individual Scales"
                points_info = f" [{len(display_time):,} pts]" if stride > 1 else ""
                ax.set_title(f'{col} ({scale_mode}){points_info}', fontsize=11)
            else:
                ax.set_title(col, fontsize=11)
        
        # X-axis label on bottom plot only
        self.axes[-1].set_xlabel(self.time_col, fontsize=10)
        
        # Enable span selectors
        for span in self.span_selectors:
            span.set_active(True)
        
        self.fig.canvas.draw_idle()
    
    def compute_fft_data(self, data_source):
        """Compute FFT data for given data source"""
        if data_source == 'original':
            data_to_use = self.original_data
        else:
            data_to_use = self.data
        
        fft_data = {}
        
        # Always use full resolution for FFT - no downsampling
        for col in self.data_cols:
            values = data_to_use[col].values
            nperseg = min(256, len(values) // 4)
            freq, psd = signal.welch(values, fs=self.sampling_rate, nperseg=nperseg)
            fft_data[col] = (freq, psd)
        
        return fft_data
    
    def plot_fft(self):
        """Plot Welch's power spectral density for all data columns"""
        # Clear previous plots
        for ax in self.axes:
            ax.clear()
        
        # Compute or retrieve FFT data for both original and current
        if self.fft_cache['original'] is None:
            print("Computing FFT for original data...")
            self.fft_cache['original'] = self.compute_fft_data('original')
        
        # Always recompute current data as it may have changed
        print("Computing FFT for current data...")
        self.fft_cache['current'] = self.compute_fft_data('current')
        
        # Get data to display based on mode
        if self.fft_compare_mode == 'original':
            display_data = self.fft_cache['original']
            title_suffix = " (Original Data)"
        else:
            display_data = self.fft_cache['current']
            title_suffix = " (Current Data)"
        
        # Define colors for plots
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
        
        # Plot PSD for each column
        for i, (ax, col) in enumerate(zip(self.axes, self.data_cols)):
            color = colors[i % len(colors)]
            freq, psd = display_data[col]
            
            # Plot based on scale mode
            if self.fft_log_scale:
                ax.semilogy(freq, psd, color=color, linewidth=1)
            else:
                ax.plot(freq, psd, color=color, linewidth=1)
        
        # Calculate scales based on both original and current data
        orig_data = self.fft_cache['original']
        curr_data = self.fft_cache['current']
        
        if self.same_scale:
            # Same scale across all axes - find global max/min
            all_psd_values = []
            for col in self.data_cols:
                all_psd_values.extend(orig_data[col][1])
                all_psd_values.extend(curr_data[col][1])
            
            y_min = min(all_psd_values)
            y_max = max(all_psd_values)
            
            if self.fft_log_scale:
                # Add margin in log space
                log_range = np.log10(y_max) - np.log10(y_min)
                margin = log_range * 0.1
                y_min_plot = 10 ** (np.log10(y_min) - margin)
                y_max_plot = 10 ** (np.log10(y_max) + margin)
            else:
                # Add margin in linear space
                margin = (y_max - y_min) * 0.1
                y_min_plot = y_min - margin
                y_max_plot = y_max + margin
            
            for ax in self.axes:
                ax.set_ylim(y_min_plot, y_max_plot)
        else:
            # Individual scales - each axis based on its own original and current max/min
            for ax, col in zip(self.axes, self.data_cols):
                axis_psd_values = list(orig_data[col][1]) + list(curr_data[col][1])
                y_min = min(axis_psd_values)
                y_max = max(axis_psd_values)
                
                if self.fft_log_scale:
                    # Add margin in log space
                    log_range = np.log10(y_max) - np.log10(y_min)
                    margin = log_range * 0.1
                    y_min_plot = 10 ** (np.log10(y_min) - margin)
                    y_max_plot = 10 ** (np.log10(y_max) + margin)
                else:
                    # Add margin in linear space
                    margin = (y_max - y_min) * 0.1
                    y_min_plot = y_min - margin
                    y_max_plot = y_max + margin
                
                ax.set_ylim(y_min_plot, y_max_plot)
        
        # Labels
        for i, (ax, col) in enumerate(zip(self.axes, self.data_cols)):
            ax.set_ylabel('PSD [V²/Hz]', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Title for first plot
            if i == 0:
                scale_mode = "Same Scale" if self.same_scale else "Individual Scales"
                y_scale_mode = "Log" if self.fft_log_scale else "Linear"
                ax.set_title(f'{col} FFT (Welch){title_suffix} ({scale_mode}, {y_scale_mode})', fontsize=11)
            else:
                ax.set_title(f'{col} FFT (Welch){title_suffix}', fontsize=11)
        
        # X-axis label on bottom plot only
        self.axes[-1].set_xlabel('Frequency [Hz]', fontsize=10)
        
        # Disable span selectors in FFT mode
        for span in self.span_selectors:
            span.set_active(False)
        
        self.fig.canvas.draw_idle()
    
    def setup_controls(self):
        """Setup control buttons and text boxes"""
        # Button positions (left, bottom, width, height)
        btn_remove_ax = plt.axes([0.02, 0.02, 0.06, 0.04])
        btn_undo_ax = plt.axes([0.09, 0.02, 0.06, 0.04])
        btn_save_ax = plt.axes([0.16, 0.02, 0.06, 0.04])
        btn_reset_ax = plt.axes([0.23, 0.02, 0.06, 0.04])
        btn_view_ax = plt.axes([0.30, 0.02, 0.08, 0.04])
        
        # FFT comparison button (only visible in FFT mode)
        btn_fft_compare_ax = plt.axes([0.39, 0.02, 0.08, 0.04])
        
        # Checkbox for scale toggle
        checkbox_ax = plt.axes([0.48, 0.02, 0.1, 0.04])
        
        # Checkbox for FFT log/linear toggle (only visible in FFT mode)
        checkbox_fft_scale_ax = plt.axes([0.59, 0.02, 0.08, 0.04])
        
        # Text boxes for precise input
        txt_start_ax = plt.axes([0.78, 0.02, 0.09, 0.04])
        txt_end_ax = plt.axes([0.89, 0.02, 0.09, 0.04])
        
        # Create buttons
        self.btn_remove = Button(btn_remove_ax, 'Remove', color='lightcoral')
        self.btn_undo = Button(btn_undo_ax, 'Undo', color='lightyellow')
        self.btn_save = Button(btn_save_ax, 'Save', color='lightgreen')
        self.btn_reset = Button(btn_reset_ax, 'Reset', color='lightblue')
        self.btn_view = Button(btn_view_ax, 'FFT View', color='lightcyan')
        self.btn_fft_compare = Button(btn_fft_compare_ax, 'Original', color='lightsalmon')
        
        # Create text boxes
        self.txt_start = TextBox(txt_start_ax, 'Start: ', initial='')
        self.txt_end = TextBox(txt_end_ax, 'End: ', initial='')
        
        # Create checkboxes
        self.check_scale = CheckButtons(checkbox_ax, ['Same Scale'], [self.same_scale])
        self.check_fft_scale = CheckButtons(checkbox_fft_scale_ax, ['Log Scale'], [self.fft_log_scale])
        
        # Create radio buttons for downsampling (in the right panel)
        active_idx = self.downsample_options.index(self.downsample_mode)
        self.radio_downsample = RadioButtons(self.ax_radio, self.downsample_options, active=active_idx)
        # Make radio button text smaller and title
        for label in self.radio_downsample.labels:
            label.set_fontsize(8)
        self.ax_radio.text(0.5, 0.98, 'Points', ha='center', va='top', fontsize=9, weight='bold', transform=self.ax_radio.transAxes)
        
        # Connect callbacks
        self.btn_remove.on_clicked(self.remove_range)
        self.btn_undo.on_clicked(self.undo_removal)
        self.btn_save.on_clicked(self.save_data)
        self.btn_reset.on_clicked(self.reset_data)
        self.btn_view.on_clicked(self.toggle_view)
        self.btn_fft_compare.on_clicked(self.toggle_fft_compare)
        self.txt_start.on_submit(self.update_start)
        self.txt_end.on_submit(self.update_end)
        self.check_scale.on_clicked(self.toggle_scale)
        self.check_fft_scale.on_clicked(self.toggle_fft_scale)
        self.radio_downsample.on_clicked(self.change_downsample)
        
        # Update button visibility
        self.update_button_visibility()
    
    def change_downsample(self, label):
        """Change downsampling mode"""
        self.downsample_mode = label
        print(f"Downsample mode: {self.downsample_mode}")
        
        # Only replot if in time view
        if self.view_mode == 'time':
            self.plot_data()
    
    def update_button_visibility(self):
        """Update button visibility and labels based on view mode"""
        if self.view_mode == 'time':
            self.btn_view.label.set_text('FFT View')
            self.btn_fft_compare.ax.set_visible(False)
            self.check_fft_scale.ax.set_visible(False)
            self.btn_remove.ax.set_visible(True)
            self.txt_start.ax.set_visible(True)
            self.txt_end.ax.set_visible(True)
            
            # Show radio buttons panel in time view
            self.ax_radio.set_visible(True)
            
            # Disable FFT-only widgets
            self.btn_fft_compare.active = False
            
            # Enable time-domain widgets
            self.btn_remove.active = True
        else:
            self.btn_view.label.set_text('Time View')
            self.btn_fft_compare.ax.set_visible(True)
            self.check_fft_scale.ax.set_visible(True)
            self.btn_remove.ax.set_visible(False)
            self.txt_start.ax.set_visible(False)
            self.txt_end.ax.set_visible(False)
            
            # Completely hide radio buttons panel in FFT view
            self.ax_radio.set_visible(False)
            
            # Enable FFT-only widgets
            self.btn_fft_compare.active = True
            
            # Disable time-domain widgets
            self.btn_remove.active = False
            
            # Update FFT compare button label
            if self.fft_compare_mode == 'current':
                self.btn_fft_compare.label.set_text('Original')
            else:
                self.btn_fft_compare.label.set_text('Current')
        
        self.fig.canvas.draw_idle()
    
    def toggle_view(self, event):
        """Toggle between time domain and FFT view"""
        if self.view_mode == 'time':
            self.view_mode = 'fft'
            self.fft_compare_mode = 'current'
            print("Switched to FFT view (Welch's method)")
        else:
            self.view_mode = 'time'
            print("Switched to time domain view")
        
        self.update_button_visibility()
        self.plot_data()
    
    def toggle_fft_compare(self, event):
        """Toggle between current and original data in FFT view"""
        if self.fft_compare_mode == 'current':
            self.fft_compare_mode = 'original'
            print("Showing FFT of original data")
        else:
            self.fft_compare_mode = 'current'
            print("Showing FFT of current data")
        
        self.update_button_visibility()
        self.plot_data()
    
    def toggle_scale(self, label):
        """Toggle between same scale and individual scales"""
        self.same_scale = not self.same_scale
        mode = "same scale" if self.same_scale else "individual scales"
        print(f"Scale mode: {mode}")
        self.plot_data()
    
    def toggle_fft_scale(self, label):
        """Toggle between logarithmic and linear scale for FFT"""
        self.fft_log_scale = not self.fft_log_scale
        scale_type = "logarithmic" if self.fft_log_scale else "linear"
        print(f"FFT Y-axis scale: {scale_type}")
        self.plot_data()
    
    def on_select(self, xmin, xmax):
        """Callback when range is selected with SpanSelector"""
        if self.view_mode != 'time':
            return
        
        self.selected_start = xmin
        self.selected_end = xmax
        self.txt_start.set_val(f'{xmin:.4f}')
        self.txt_end.set_val(f'{xmax:.4f}')
        
        # Debounce: only sync if difference is significant
        # This reduces lag during selection
        for span in self.span_selectors:
            if abs(span.extents[0] - xmin) > 0.001 or abs(span.extents[1] - xmax) > 0.001:
                span.extents = (xmin, xmax)
    
    def update_start(self, text):
        """Update start time from text box"""
        try:
            self.selected_start = float(text)
            if self.selected_end is not None:
                self.sync_span_selectors(self.selected_start, self.selected_end)
        except ValueError:
            print("Invalid start time")
    
    def update_end(self, text):
        """Update end time from text box"""
        try:
            self.selected_end = float(text)
            if self.selected_start is not None:
                self.sync_span_selectors(self.selected_start, self.selected_end)
        except ValueError:
            print("Invalid end time")
    
    def sync_span_selectors(self, xmin, xmax):
        """Synchronize all span selectors to the same range"""
        for span in self.span_selectors:
            span.extents = (xmin, xmax)
        self.fig.canvas.draw_idle()
    
    def find_nearest_index(self, time_value):
        """Find the index of the nearest actual data point to given time value"""
        idx = np.abs(self.time_array - time_value).argmin()
        return idx
    
    def remove_range(self, event):
        """Remove selected time range and close gaps"""
        if self.selected_start is None or self.selected_end is None:
            print("Please select a range first!")
            return
        
        start = min(self.selected_start, self.selected_end)
        end = max(self.selected_start, self.selected_end)
        
        # Find nearest actual data points using full resolution data
        start_idx = self.find_nearest_index(start)
        end_idx = self.find_nearest_index(end)
        
        # Get actual time values from the data
        actual_start = self.time_array[start_idx]
        actual_end = self.time_array[end_idx]
        
        print(f"Selected: {start:.4f} to {end:.4f}")
        print(f"Nearest actual points: {actual_start:.6f} to {actual_end:.6f}")
        
        # Save current state for undo (save DataFrame, not arrays)
        self.history.append(self.data.copy())
        
        # Find indices in range (inclusive of start and end points)
        mask = (self.time_array >= actual_start) & (self.time_array <= actual_end)
        
        if not np.any(mask):
            print("No data in selected range")
            return
        
        # Calculate exact time gap from actual data points
        time_gap = actual_end - actual_start
        
        # Remove the range from DataFrame
        self.data = self.data[~mask].copy()
        
        # STEP 1: Close the gap by shifting data after removal
        time_values = self.data[self.time_col].values
        after_removal = time_values > actual_end
        time_values[after_removal] -= time_gap
        
        # STEP 2: Renormalize all times to start from 0.0
        min_time = time_values.min()
        time_values = time_values - min_time
    
        self.data[self.time_col] = time_values
    
        # Reset index
        self.data.reset_index(drop=True, inplace=True)
        
        # Update NumPy arrays
        self._convert_to_numpy()
        
        print(f"Removed {np.sum(mask)} points")
        print(f"Time gap closed: {time_gap:.6f}")
        print(f"Times renormalized: started at {min_time:.6f} → now at 0.0")
        
        # Recalculate sampling rate and invalidate FFT cache
        self.calculate_sampling_rate()
        self.fft_cache['current'] = None
        
        # Clear selection
        self.selected_start = None
        self.selected_end = None
        self.txt_start.set_val('')
        self.txt_end.set_val('')
        
        # Clear all span selectors
        for span in self.span_selectors:
            span.extents = (0, 0)
        
        # Replot
        self.plot_data()
    
    def undo_removal(self, event):
        """Undo last removal"""
        if len(self.history) > 0:
            self.data = self.history.pop()
            self._convert_to_numpy()
            print("Undone last removal")
            self.calculate_sampling_rate()
            self.fft_cache['current'] = None
            self.plot_data()
        else:
            print("Nothing to undo")
    
    def reset_data(self, event):
        """Reset to original data"""
        self.data = self.original_data.copy()
        self._convert_to_numpy()
        self.history.clear()
        print("Reset to original data")
        self.calculate_sampling_rate()
        self.fft_cache['current'] = None
        self.plot_data()
    
    def save_data(self, event):
        """Save cleaned data to outputs folder"""
        # Generate new filename with timestamp
        base_name = os.path.splitext(os.path.basename(self.csv_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{base_name}_cleaned_{timestamp}.csv"
        output_path = os.path.join(self.output_dir, output_filename)
        
        # Save full resolution data
        self.data.to_csv(output_path, index=False)
        print(f"✅ Saved cleaned data to: {output_path}")
        print(f"Original file unchanged: {self.csv_path}")
        print("Timestamps have been adjusted in the saved file")


def main():
    """Main function to handle file input"""
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        print("Remover")
        csv_file = input("\nEnter the path to your CSV file: ").strip()
        csv_file = csv_file.strip('"').strip("'")
    
    try:
        remover = VibrationDataRemover(csv_file)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease check the file path and try again.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()