# Patient BG Trace Visualization with Meal Bounding Boxes

## Quick Start

Generate a plot with default settings (auto-selects a low-mean-BG patient):
```bash
python scripts/visualization/plot_patient_bg_trace_with_meals.py
```

## Usage Examples

### View patient statistics to choose a patient
```bash
python scripts/visualization/plot_patient_bg_trace_with_meals.py --show_patient_stats
```

This shows all 226 patients sorted by mean BG (lowest first), then by standard deviation.

### Specify a particular patient
```bash
python scripts/visualization/plot_patient_bg_trace_with_meals.py --patient_id ale_10
```

### Specify both patient and date
```bash
python scripts/visualization/plot_patient_bg_trace_with_meals.py --patient_id ale_10 --date 2020-03-15
```

### Custom output filename
```bash
python scripts/visualization/plot_patient_bg_trace_with_meals.py --output my_figure
```

## Configuration

All visual parameters are in the `CONFIGURATION_*` variables at the top of the script. You can easily adjust:

### Patient Selection
- `CONFIGURATION_PATIENT_ID`: Specific patient (e.g., `"ale_10"`) or `None` for auto-select
- `CONFIGURATION_DATE`: Specific date (e.g., `"2020-03-15"`) or `None` for random
- `CONFIGURATION_SEED`: Random seed for reproducibility (default: 42)
- `CONFIGURATION_NUM_CANDIDATES`: Number of low-mean-BG patients to consider (default: 20)

### Time Window
- `CONFIGURATION_START_HOUR`: Start time in 24h format (default: 6 = 6AM)
- `CONFIGURATION_END_HOUR`: End time in 24h format (default: 22 = 10PM)

### Y-Axis Range
- `CONFIGURATION_Y_MIN`: Minimum BG value (default: 0.0 mmol/L)
- `CONFIGURATION_Y_MAX`: Maximum BG value (default: 13.3 mmol/L)

### Bounding Boxes (3 boxes total)
Each box has 4 parameters (x_start, width, y_bottom, height):

**Box 1:**
- `CONFIGURATION_BOX1_X_START`: Hours from start time (default: 2.0)
- `CONFIGURATION_BOX1_WIDTH`: Box width in hours (default: 1.5)
- `CONFIGURATION_BOX1_Y_BOTTOM`: Bottom edge in mmol/L (default: 4.0)
- `CONFIGURATION_BOX1_HEIGHT`: Box height in mmol/L (default: 4.0)

**Box 2 & 3:** Similar parameters with BOX2_ and BOX3_ prefixes

### Visual Styling
- `CONFIGURATION_LINE_COLOR`: BG trace color (default: `"#1565c0"` blue)
- `CONFIGURATION_LINE_WIDTH`: BG trace line width (default: 1.5 pt)
- `CONFIGURATION_MARKER_SIZE`: BG trace marker size (default: 3.0 pt)
- `CONFIGURATION_BOX_EDGE_COLOR`: Bounding box color (default: `"#d32f2f"` red)
- `CONFIGURATION_BOX_LINE_WIDTH`: Bounding box line width (default: 2.0 pt)
- `CONFIGURATION_BOX_LINE_STYLE`: Bounding box style (default: `"--"` dashed)
- `CONFIGURATION_BOX_FILL_ALPHA`: Box fill transparency (default: 0.05)
- And many more styling options...

### Output
- `CONFIGURATION_OUTPUT_DIR`: Output directory (default: `results/figures/`)
- `CONFIGURATION_OUTPUT_FILENAME`: Base filename (default: `"patient_bg_trace_with_meals"`)
- `CONFIGURATION_OUTPUT_FORMATS`: List of formats (default: `["png", "pdf"]`)
- `CONFIGURATION_DPI`: Resolution for PNG (default: 300)
- `CONFIGURATION_FIGURE_WIDTH`: Figure width in inches (default: 10.0)
- `CONFIGURATION_FIGURE_HEIGHT`: Figure height in inches (default: 5.0)

## Tips for Finding Good Examples

1. **Run with `--show_patient_stats`** to see all patients ranked by mean BG and SD
2. **Patient ale_10** has the lowest mean BG (5.44 mmol/L) and very low SD (1.16)
3. **Run without arguments** multiple times with different `CONFIGURATION_SEED` values to see different random selections
4. **Manually adjust bounding boxes** by editing the `CONFIGURATION_BOX*_*` variables to highlight specific meal spikes

## Output Files

By default, the script generates:
- `results/figures/patient_bg_trace_with_meals.png` (raster, 300 DPI)
- `results/figures/patient_bg_trace_with_meals.pdf` (vector, publication-quality)

## Example Workflow

1. Find interesting patients:
   ```bash
   python scripts/visualization/plot_patient_bg_trace_with_meals.py --show_patient_stats | head -30
   ```

2. Generate plot for a specific patient:
   ```bash
   python scripts/visualization/plot_patient_bg_trace_with_meals.py --patient_id ale_10
   ```

3. Try different dates for that patient by running again (random date selection)

4. Once you find a good example, note the date shown in the output

5. Edit the configuration variables at the top of the script to:
   - Set `CONFIGURATION_PATIENT_ID` and `CONFIGURATION_DATE` to lock in your selection
   - Adjust bounding box positions to highlight actual meal spikes
   - Fine-tune visual styling

6. Generate final figure:
   ```bash
   python scripts/visualization/plot_patient_bg_trace_with_meals.py
   ```

## Data Source

- **Dataset**: ReplaceBG/Aleppo 2017 (226 patients with Type 1 diabetes)
- **Resolution**: 5-minute intervals
- **Units**: mmol/L (millimoles per liter)
- **Time period**: Various dates in 2020 (normalized for privacy)
