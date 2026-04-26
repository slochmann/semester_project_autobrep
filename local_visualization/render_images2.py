import glob
import math
import os
import subprocess
import sys
from datetime import datetime

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Display.backend import load_backend
from PyQt5 import QtWidgets

# Load backend and create QApplication once at module level
load_backend("qt-pyqt5")
from OCC.Display.qtDisplay import qtViewer3d

app = QtWidgets.QApplication.instance()
if app is None:
    app = QtWidgets.QApplication(sys.argv)


def process_step_files_in_directory(step_dir, output_base_dir="output_images", num_files=None):
    """Process STEP files by spawning a new subprocess for each file.
    
    Args:
        step_dir: Directory containing STEP files
        output_base_dir: Base directory for output images
        num_files: Maximum number of files to process (None = all files)
    """

    # Find all STEP files in the given directory
    step_files = glob.glob(os.path.join(step_dir, "*.step"))

    if not step_files:
        print(f"No STEP files found in directory: {step_dir}")
        return

    # Limit number of files if specified
    if num_files is not None:
        step_files = step_files[:num_files]
        print(f"Processing {len(step_files)} out of available files (--num-files={num_files})")

    os.makedirs(output_base_dir, exist_ok=True)

    for step_file in step_files:
        # Extract the base name of the STEP file (e.g., "file1.step" -> "file1")
        base_name = os.path.splitext(os.path.basename(step_file))[0]

        # Create a directory for the images of this STEP file
        output_dir = os.path.join(output_base_dir, base_name)
        os.makedirs(output_dir, exist_ok=True)

        # Generate images for the STEP file
        print(f"Processing STEP file: {step_file}")

        # Spawn a new subprocess for each file to avoid resource accumulation
        subprocess.run(
            [sys.executable, __file__, "--process-single", step_file, output_dir],
            check=False,
        )

        print(f"Images saved in: {output_dir}")


def process_single_file(step_file, output_dir):
    """Process a single STEP file. Called in a subprocess."""
    step_reader = STEPControl_Reader()
    if step_reader.ReadFile(step_file) != 1:
        print(f"Error: Cannot load {step_file}")
        return False

    step_reader.TransferRoots()
    shape = step_reader.OneShape()

    # Create new viewer for this file
    viewer = qtViewer3d()
    viewer.resize(2000, 2000)
    viewer.setWindowTitle("STEP Viewer")
    viewer.show()

    # Display the shape
    viewer._display.DisplayShape(shape, update=True)
    viewer._display.FitAll()  # First fit all to get proper bounding box

    # Values > 1 = zoom in, Values < 1 = zoom out
    viewer._display.ZoomFactor(50.0)

    # Update display
    viewer._display.Context.UpdateCurrentViewer()

    # Define angles for diagonal rotation (in degrees)
    angles = range(0, 360, 10)  # Rotation angles for both axes with 45-degree steps

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for i, angle in enumerate(angles):
        # Rotate the view diagonally around X and Z axes
        rad_angle = math.radians(angle)
        viewer._display.View.SetProj(
            math.cos(rad_angle),  # X-axis component
            math.sin(rad_angle),  # Z-axis component
            math.sin(rad_angle),  # Diagonal component
        )

        # Update display
        QtWidgets.QApplication.processEvents()
        viewer._display.FitAll()

        # Save the image to the specified file with zero-padded numbering
        output_image = os.path.join(output_dir, f"output_diagonal_{i:03d}.png")
        viewer._display.View.Dump(output_image)
        print(f"Image saved to {output_image}")

    # Close the viewer window after all images are saved
    viewer.close()
    app.quit()


# Example usage
# step_file = "/home/sebi/MSc/3.Sem/semester_thesis/brepgen-semester-thesis/cloned_project/AutoBrep/samples/9pMgmQEZbGz5EYp5FiE9_000.step"
# step_to_image_with_viewer(step_file, "my_output.png", zoom_factor=500.0)  # 5x zoom
# step_to_images_with_different_angles(    step_file, "output_images", zoom_factor=500.0)  # 5x zoom
# step_to_images_with_two_axis_rotation(    step_file, "output_images_two_axes", zoom_factor=500.0)  # 5x zoom

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--process-single":
        # Called from subprocess to process a single file
        step_file = sys.argv[2]
        output_dir = sys.argv[3]
        process_single_file(step_file, output_dir)
    else:
        # Main process: find and spawn subprocesses for each file
        step_dir = "/home/sebi/MSc/3.Sem/semester_thesis/local-git/semester_project_autobrep/euler_remote_mount/scratch/AutoBrep/checkpoints/600_cylinders_T0.5_comp17_lr3e-5_e4_r32_a128_20260424_154843/samples_step_000528"
        
        # Parse --num-files argument
        num_files = None
        if "--num-files" in sys.argv:
            try:
                idx = sys.argv.index("--num-files")
                num_files = int(sys.argv[idx + 1])
                print(f"Limiting to {num_files} files")
            except (IndexError, ValueError):
                print("Invalid --num-files argument. Usage: python render_images2.py --num-files <N>")
        
        # Parse --output-dir argument, or create timestamped directory
        output_base_dir = "output_images"
        if "--output-dir" in sys.argv:
            try:
                idx = sys.argv.index("--output-dir")
                output_base_dir = sys.argv[idx + 1]
                print(f"Using output directory: {output_base_dir}")
            except IndexError:
                print("Invalid --output-dir argument. Usage: python render_images2.py --output-dir <DIR>")
        else:
            # Create timestamped directory by default
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_base_dir = f"output_images_{timestamp}"
            print(f"Creating timestamped output directory: {output_base_dir}")
        
        process_step_files_in_directory(step_dir, output_base_dir=output_base_dir, num_files=num_files)


# xvfb-run -a python render_images2.py --num_files 36
