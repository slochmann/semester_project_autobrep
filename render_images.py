import glob
import math
import os
import subprocess
import sys

from OCC.Core.AIS import AIS_InteractiveContext
from OCC.Core.Aspect import Aspect_DisplayConnection
from OCC.Core.OpenGl import OpenGl_GraphicDriver
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.V3d import V3d_Viewer
from OCC.Display.backend import load_backend

# Initialize for GUI mode (only when needed)
_gui_app = None
_qtViewer3d = None


def get_gui_viewer():
    """Lazy-load GUI components only if needed."""
    global _gui_app, _qtViewer3d
    if _qtViewer3d is None:
        from PyQt5 import QtWidgets

        load_backend("qt-pyqt5")
        from OCC.Display.qtDisplay import qtViewer3d

        _gui_app = QtWidgets.QApplication.instance()
        if _gui_app is None:
            _gui_app = QtWidgets.QApplication(sys.argv)
        _qtViewer3d = qtViewer3d
    return _qtViewer3d, _gui_app


def process_step_files_in_directory(step_dir, output_base_dir="output_images"):
    """Process STEP files by spawning a new subprocess for each file."""

    # Find all STEP files in the given directory
    step_files = glob.glob(os.path.join(step_dir, "*.step"))

    if not step_files:
        print(f"No STEP files found in directory: {step_dir}")
        return

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
    """Process a single STEP file using offscreen rendering."""
    try:
        step_reader = STEPControl_Reader()
        if step_reader.ReadFile(step_file) != 1:
            print(f"Error: Cannot load {step_file}")
            return False

        step_reader.TransferRoots()
        shape = step_reader.OneShape()

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Use offscreen rendering via X virtual framebuffer
        # OCCT will render directly to file
        _render_shape_offscreen(shape, output_dir)

        return True
    except Exception as e:
        print(f"Error processing {step_file}: {e}")
        import traceback

        traceback.print_exc()
        return False


def _render_shape_offscreen(shape, output_dir, image_size=2000, num_angles=36):
    """Render a shape from multiple angles using offscreen rendering."""
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import BRepBndLib
    from OCC.Core.Graphic3d import Graphic3d_Camera

    # Create viewer components
    display_connection = Aspect_DisplayConnection()
    graphic_driver = OpenGl_GraphicDriver(display_connection)
    viewer = V3d_Viewer(graphic_driver, 0, V3d_Viewer.Orthographic_s())
    viewer.SetDefaultBackgroundColor(1.0, 1.0, 1.0)  # White background

    # Create view and context
    view = viewer.CreateView()
    context = AIS_InteractiveContext(viewer)

    # Display the shape
    context.Display(shape)
    view.FitAll()

    # Get bounding box for proper framing
    bbox = Bnd_Box()
    BRepBndLib.Add(shape, bbox)

    # Define angles for rotation (in degrees)
    angles = [i * (360 / num_angles) for i in range(num_angles)]

    for i, angle in enumerate(angles):
        # Rotate the view diagonally around X and Z axes
        rad_angle = math.radians(angle)

        # Set view direction for diagonal rotation
        view.SetProj(
            math.cos(rad_angle),  # X-axis component
            math.sin(rad_angle),  # Z-axis component
            math.sin(rad_angle * 0.5),  # Diagonal component
        )

        view.FitAll()

        # Save the image
        output_image = os.path.join(output_dir, f"output_diagonal_{i:03d}.png")
        view.Dump(output_image, Graphic3d_Camera.Projection_Orthographic)
        print(f"Image saved to {output_image}")


# Example usage
# step_file = "/home/sebi/MSc/3.Sem/semester_thesis/brepgen-semester-thesis/cloned_project/AutoBrep/samples/9pMgmQEZbGz5EYp5FiE9_000.step"
# step_to_image_with_viewer(step_file, "my_output.png", zoom_factor=500.0)  # 5x zoom
# step_to_images_with_different_angles(    step_file, "output_images", zoom_factor=500.0)  # 5x zoom
# step_to_images_with_two_axis_rotation(    step_file, "output_images_two_axes", zoom_factor=500.0)  # 5x zoom

if __name__ == "__main__":
    if len(sys.argv) == 3:
        # Called with two positional arguments: STEP_FILE and output_dir
        step_file = sys.argv[1]
        output_dir = sys.argv[2]
        process_single_file(step_file, output_dir)
    elif len(sys.argv) > 1 and sys.argv[1] == "--process-single":
        # Called from subprocess to process a single file (backward compatibility)
        step_file = sys.argv[2]
        output_dir = sys.argv[3]
        process_single_file(step_file, output_dir)
    else:
        # Main process: find and spawn subprocesses for each file
        step_dir = "$SCRATCH/AutoBrep/samples"
        process_step_files_in_directory(step_dir)
