"""
Excel-Based Custom Pattern Module
===================================

This module allows users to create custom spatial patterns by drawing them in Excel.
Simply fill cells with colors (black = 1.0, white/no fill = 0.0, gray = interpolated).

Usage:
    1. Create an Excel file with filled cells representing your pattern
    2. Load it using ExcelPatternLoader
    3. Use it like any other SpatialPattern

The pattern is automatically resized to match your network dimensions.
"""

import numpy as np
from openpyxl import load_workbook
from openpyxl.styles import PatternFill
import os


class ExcelPatternLoader:
    """Load and convert Excel files into network-compatible patterns."""
    
    @staticmethod
    def rgb_to_intensity(rgb_color):
        """
        Convert RGB color to grayscale intensity value (0 to 1).
        
        Args:
            rgb_color: RGB color string (e.g., 'FF000000' for black) or None
        
        Returns:
            float: Intensity value where 0.0 = white/no fill, 1.0 = black
        """
        if rgb_color is None or rgb_color == '00000000':
            # No fill or completely transparent = 0 (no stimulation)
            return 0.0
        
        # Extract RGB components (format is typically 'AARRGGBB')
        if len(rgb_color) >= 6:
            # Take last 6 characters for RGB
            rgb_hex = rgb_color[-6:]
            try:
                r = int(rgb_hex[0:2], 16)
                g = int(rgb_hex[2:4], 16)
                b = int(rgb_hex[4:6], 16)
                
                # Convert to grayscale using standard luminance formula
                grayscale = 0.299 * r + 0.587 * g + 0.114 * b
                
                # Invert: black (0) -> 1.0, white (255) -> 0.0
                intensity = 1.0 - (grayscale / 255.0)
                
                return intensity
            except ValueError:
                return 0.0
        
        return 0.0
    
    @staticmethod
    def load_pattern_from_excel(filepath, max_size=500):
        """
        Load a spatial pattern from an Excel file.
        
        Args:
            filepath (str): Path to the Excel file (.xlsx, .xlsm)
            max_size (int): Maximum number of rows/cols to read (default 500)
        
        Returns:
            np.ndarray: Pattern array with values from 0.0 to 1.0
        
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is invalid
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Excel file not found: {filepath}")
        
        # Load workbook
        try:
            wb = load_workbook(filepath, data_only=True)
        except Exception as e:
            raise ValueError(f"Failed to load Excel file: {e}")
        
        # Get the first (active) worksheet
        ws = wb.active

        # Determine actual dimensions (up to max_size).
        #
        # M21: `ws.max_row`/`max_column` report the extent of anything openpyxl
        # considers "used", INCLUDING cells that carry only stray formatting.
        # A 10x10 drawing in a sheet where someone once styled row 500 loaded
        # as a 500x10 array that was 98% empty, and the resize then squashed
        # the actual drawing into near-nothing. The used range is trimmed below
        # to the rows and columns that actually carry content.
        max_row = min(ws.max_row or 0, max_size)
        max_col = min(ws.max_column or 0, max_size)

        if max_row < 1 or max_col < 1:
            raise ValueError(
                f"'{filepath}' contains no usable cells "
                f"({max_row} rows x {max_col} columns)."
            )

        pattern = np.zeros((max_row, max_col))

        # Cells whose fill could not be interpreted, for the diagnostic below.
        theme_coloured = 0
        indexed_coloured = 0

        for i, row in enumerate(ws.iter_rows(min_row=1, max_row=max_row,
                                             min_col=1, max_col=max_col)):
            for j, cell in enumerate(row):
                intensity = None

                # 1) Cell VALUE first. A numeric 0..1 (or 0/1) is the most
                #    explicit way to specify a pattern, and it was ignored
                #    entirely — only fills were read, so a sheet typed as
                #    numbers loaded as all zeros.
                value = cell.value
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    intensity = float(value)
                    # Accept 0-100 as a percentage as well as 0-1.
                    if intensity > 1.0:
                        intensity = intensity / 100.0
                    intensity = float(np.clip(intensity, 0.0, 1.0))

                # 2) Otherwise fall back to the cell fill.
                if intensity is None:
                    fill = cell.fill
                    if fill and fill.patternType == 'solid' and fill.fgColor:
                        fg = fill.fgColor
                        rgb_color = getattr(fg, 'rgb', None)
                        if isinstance(rgb_color, str):
                            intensity = ExcelPatternLoader.rgb_to_intensity(rgb_color)
                        elif getattr(fg, 'type', None) == 'theme':
                            # M21: a THEME colour has rgb = None. Excel's own
                            # palette buttons produce theme colours, so a fully
                            # drawn pattern loaded as all zeros with no warning
                            # whatsoever. The exact RGB needs the workbook
                            # theme, which openpyxl does not resolve; the cell
                            # is treated as filled (intensity 1) and counted so
                            # the user is told.
                            intensity = 1.0
                            theme_coloured += 1
                        elif getattr(fg, 'type', None) == 'indexed':
                            intensity = 1.0
                            indexed_coloured += 1

                if intensity is not None:
                    pattern[i, j] = intensity

        wb.close()

        # Trim trailing all-zero rows/columns left by stray formatting.
        nonzero_rows = np.flatnonzero(pattern.any(axis=1))
        nonzero_cols = np.flatnonzero(pattern.any(axis=0))

        # M21: an all-zero pattern is never a valid stimulus, and it used to be
        # returned silently — indistinguishable from a blank sheet, a theme
        # colour, or a sheet the user filled in with numbers.
        if nonzero_rows.size == 0 or nonzero_cols.size == 0:
            raise ValueError(
                f"'{filepath}' produced an ALL-ZERO pattern, which cannot be "
                "used as a stimulus.\n"
                "Check that the cells are either:\n"
                "  - filled with a solid colour (Home > Fill Color), or\n"
                "  - given a numeric value between 0 and 1.\n"
                "Note that conditional formatting and cell BORDERS are not "
                "read — only solid fills and cell values."
            )

        pattern = pattern[nonzero_rows[0]:nonzero_rows[-1] + 1,
                          nonzero_cols[0]:nonzero_cols[-1] + 1]

        if theme_coloured or indexed_coloured:
            print(
                f"Note: {theme_coloured + indexed_coloured} cell(s) use a theme "
                "or indexed fill colour, whose exact RGB is not resolvable "
                "without the workbook theme. They have been treated as fully "
                "filled (intensity 1.0). For graded intensities use standard "
                "RGB fills or numeric cell values."
            )

        return pattern
    
    @staticmethod
    def resize_pattern(pattern, target_shape):
        """
        Resize a pattern to match target network dimensions.
        
        Uses nearest-neighbor interpolation for discrete patterns or
        bilinear interpolation for smooth patterns.
        
        Args:
            pattern (np.ndarray): Original pattern array
            target_shape (tuple): Target dimensions (rows, cols)
        
        Returns:
            np.ndarray: Resized pattern
        """
        from scipy.ndimage import zoom

        source_shape = pattern.shape
        zoom_factors = (target_shape[0] / source_shape[0],
                        target_shape[1] / source_shape[1])

        downsampling = (zoom_factors[0] < 1.0) or (zoom_factors[1] < 1.0)

        if downsampling:
            # AREA AVERAGING (M21).
            #
            # The original used order=1 (bilinear) unconditionally. Both
            # bilinear and nearest-neighbour SAMPLE the source at computed
            # positions, so a thin feature that falls between sample points
            # disappears completely: a one-cell-wide line in an 8x8 pattern
            # scaled to 4x4 is sampled at source rows 0, 2, 4, 6 and a line on
            # row 3 vanishes without trace. (Nearest-neighbour, which the audit
            # suggested, has exactly the same failure — verified.)
            #
            # Averaging every source cell that maps into an output cell is the
            # physically right operation here: the pattern is an illumination
            # map, and a target pixel covering a half-lit region really is half
            # lit. A thin line becomes a dimmer line rather than nothing.
            out = np.zeros(target_shape, dtype=float)
            row_edges = np.linspace(0, source_shape[0], target_shape[0] + 1)
            col_edges = np.linspace(0, source_shape[1], target_shape[1] + 1)
            for i in range(target_shape[0]):
                r0, r1 = int(np.floor(row_edges[i])), int(np.ceil(row_edges[i + 1]))
                r1 = max(r1, r0 + 1)
                for j in range(target_shape[1]):
                    c0, c1 = int(np.floor(col_edges[j])), int(np.ceil(col_edges[j + 1]))
                    c1 = max(c1, c0 + 1)
                    out[i, j] = float(np.mean(pattern[r0:r1, c0:c1]))
            resized = out
        else:
            # Upsampling samples a coarser grid onto a finer one, where no
            # feature can be lost. Bilinear keeps the result smooth.
            resized = zoom(pattern, zoom_factors, order=1)

        # Ensure values stay in [0, 1] range
        resized = np.clip(resized, 0.0, 1.0)

        return resized
    
    @staticmethod
    def load_and_resize(filepath, target_shape, max_size=500):
        """
        Load an Excel pattern and automatically resize it to target network shape.
        
        This is the main convenience function for users.
        
        Args:
            filepath (str): Path to Excel file
            target_shape (tuple): Network dimensions (rows, cols)
            max_size (int): Maximum size to read from Excel
        
        Returns:
            np.ndarray: Pattern array sized for the network
        
        Example:
            >>> pattern = ExcelPatternLoader.load_and_resize('my_pattern.xlsx', (10, 10))
            >>> network.apply_spatial_pattern(pattern, intensity=20, wavelength=550, 
            ...                               duration_ms=100, mode='potentiation',
            ...                               pattern_name='Custom Excel Pattern')
        """
        # Load the raw pattern
        raw_pattern = ExcelPatternLoader.load_pattern_from_excel(filepath, max_size)
        
        # Resize to target
        resized_pattern = ExcelPatternLoader.resize_pattern(raw_pattern, target_shape)
        
        return resized_pattern
    
    @staticmethod
    def preview_pattern(pattern, title="Pattern Preview"):
        """
        Visualize a loaded pattern.
        
        Args:
            pattern (np.ndarray): Pattern array to visualize
            title (str): Plot title
        """
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(pattern, cmap='Greys', interpolation='nearest', 
                      vmin=0, vmax=1)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Intensity (0=no stim, 1=max stim)', fontsize=10)
        
        # Add grid for small patterns
        if pattern.shape[0] <= 20 and pattern.shape[1] <= 20:
            ax.set_xticks(np.arange(-0.5, pattern.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-0.5, pattern.shape[0], 1), minor=True)
            ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        return fig


# =============================================================================
# HELPER FUNCTION FOR GUI INTEGRATION
# =============================================================================

def validate_excel_file(filepath):
    """
    Validate that a file is a readable Excel file.
    
    Args:
        filepath (str): Path to file
    
    Returns:
        tuple: (is_valid: bool, message: str)
    """
    if not os.path.exists(filepath):
        return False, "File does not exist"
    
    if not filepath.lower().endswith(('.xlsx', '.xlsm', '.xls')):
        return False, "File must be an Excel file (.xlsx, .xlsm, or .xls)"
    
    try:
        wb = load_workbook(filepath, data_only=True)
        wb.close()
        return True, "Valid Excel file"
    except Exception as e:
        return False, f"Cannot read Excel file: {str(e)}"


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    """
    Example of how to create and use custom Excel patterns.
    
    To create an Excel pattern:
    1. Open Excel
    2. Select cells and fill them with colors:
       - Black = maximum stimulation (1.0)
       - White/No fill = no stimulation (0.0)
       - Gray shades = intermediate values
    3. Save as .xlsx file
    4. Load using this module
    """
    
    print("Excel Pattern Loader - Example Usage")
    print("="*60)
    print("\nTo create a custom pattern:")
    print("  1. Open Excel and create a new workbook")
    print("  2. Fill cells with colors:")
    print("     • Black cells → Maximum stimulation (1.0)")
    print("     • White/empty cells → No stimulation (0.0)")
    print("     • Gray cells → Intermediate values")
    print("  3. Save as 'my_pattern.xlsx'")
    print("  4. Load it in your simulation:")
    print()
    print("     from excel_patterns import ExcelPatternLoader")
    print("     pattern = ExcelPatternLoader.load_and_resize(")
    print("         'my_pattern.xlsx', network.shape)")
    print("     network.apply_spatial_pattern(pattern, ...)")
    print()
    print("="*60)
    
    # Try to create a simple demo pattern programmatically
    try:
        from openpyxl import Workbook
        from openpyxl.styles import PatternFill
        
        print("\nCreating demo Excel pattern: 'demo_pattern.xlsx'")
        
        wb = Workbook()
        ws = wb.active
        
        # Create a simple smiley face pattern
        black_fill = PatternFill(start_color="FF000000", end_color="FF000000", 
                                 fill_type="solid")
        gray_fill = PatternFill(start_color="FF808080", end_color="FF808080",
                               fill_type="solid")
        
        # Simple 10x10 smiley pattern
        # Eyes
        ws['C3'].fill = black_fill
        ws['H3'].fill = black_fill
        
        # Smile
        for col in ['D', 'E', 'F', 'G']:
            ws[f'{col}7'].fill = gray_fill
        ws['C6'].fill = gray_fill
        ws['H6'].fill = gray_fill
        
        wb.save('demo_pattern.xlsx')
        print("✓ Created 'demo_pattern.xlsx' (10×10 smiley face)")
        
        # Load and preview
        pattern = ExcelPatternLoader.load_pattern_from_excel('demo_pattern.xlsx', max_size=10)
        print(f"✓ Loaded pattern with shape {pattern.shape}")
        print(f"  Intensity range: {pattern.min():.2f} to {pattern.max():.2f}")
        
        # Show a text representation
        print("\nPattern visualization (text):")
        print("█ = high intensity, ░ = low intensity, · = zero")
        print("-" * (pattern.shape[1] * 2 + 2))
        for row in pattern:
            print("|", end="")
            for val in row:
                if val > 0.7:
                    print("█", end=" ")
                elif val > 0.3:
                    print("░", end=" ")
                else:
                    print("·", end=" ")
            print("|")
        print("-" * (pattern.shape[1] * 2 + 2))
        
    except ImportError:
        print("\nNote: Install openpyxl to create Excel files:")
        print("  pip install openpyxl --break-system-packages")
    except Exception as e:
        print(f"\nError creating demo: {e}")