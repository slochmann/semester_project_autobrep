import os
import sys

import vtk
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.StlAPI import StlAPI_Writer


def process_single_file(step_file, output_dir):
    """Process a single STEP file and generate rotated views."""
    # Read STEP file
    step_reader = STEPControl_Reader()
    if step_reader.ReadFile(step_file) != 1:
        print(f"Error: Cannot load {step_file}")
        return False

    step_reader.TransferRoots()
    shape = step_reader.OneShape()

    # Create mesh
    mesh = BRepMesh_IncrementalMesh(shape, 0.001, False, 0.01, True)
    mesh.Perform()

    # Export to STL
    temp_stl = os.path.join(output_dir, "_temp_model.stl")
    stl_writer = StlAPI_Writer()
    stl_writer.Write(shape, temp_stl)

    # Read STL with VTK
    stl_reader = vtk.vtkSTLReader()
    stl_reader.SetFileName(temp_stl)
    stl_reader.Update()

    # Create visualization
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(stl_reader.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(1, 1, 1)

    render_window = vtk.vtkRenderWindow()
    render_window.SetOffScreenRendering(True)
    render_window.SetSize(2000, 2000)
    render_window.AddRenderer(renderer)

    os.makedirs(output_dir, exist_ok=True)

    # Render rotated views
    for i, angle in enumerate(range(0, 360, 10)):
        renderer.ResetCamera()
        renderer.GetActiveCamera().Azimuth(angle)
        renderer.GetActiveCamera().Elevation(angle)
        render_window.Render()

        window_to_image = vtk.vtkWindowToImageFilter()
        window_to_image.SetInput(render_window)
        window_to_image.Update()

        writer = vtk.vtkPNGWriter()
        output_image = os.path.join(output_dir, f"output_diagonal_{i:03d}.png")
        writer.SetFileName(output_image)
        writer.SetInputConnection(window_to_image.GetOutputPort())
        writer.Write()
        print(f"Saved: {output_image}")

    os.remove(temp_stl)
    return True


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python render_images.py <step_file> <output_dir>")
        sys.exit(1)

    process_single_file(sys.argv[1], sys.argv[2])
