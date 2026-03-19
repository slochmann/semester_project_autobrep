import os
import subprocess


def create_videos_from_folders(output_base_dir="output_images", framerate=10):
    """Iterate over folders in output_images and create an MP4 video for each."""
    if not os.path.exists(output_base_dir):
        print(f"Output directory does not exist: {output_base_dir}")
        return

    # Iterate over subdirectories in the output base directory
    for folder in os.listdir(output_base_dir):
        folder_path = os.path.join(output_base_dir, folder)
        if os.path.isdir(folder_path):
            # Create an MP4 from the images in the folder
            video_path = os.path.join(output_base_dir, f"{folder}.mp4")
            ffmpeg_command = [
                "ffmpeg",
                "-framerate",
                str(framerate),
                "-i",
                os.path.join(folder_path, "output_diagonal_%03d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "medium",
                video_path,
            ]
            try:
                subprocess.run(ffmpeg_command, check=True)
                print(f"Created video: {video_path}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to create video for folder: {folder_path}. Error: {e}")


def combine_videos_with_xstack(
    output_base_dir="output_images",
    output_path="output_collage.mp4",
    grid_size=3,
    tile_size=800,
):
    """Combine MP4 videos in the output directory into a collage using FFmpeg's xstack filter.

    Args:
        output_base_dir: Directory containing MP4 files
        output_path: Path for the output collage MP4
        grid_size: Size of the grid (e.g., 3 for 3x3, 2 for 2x2)
        tile_size: Size of each tile in pixels (default 800 for higher resolution)
    """
    video_paths = []

    # Collect all MP4 videos in the output directory
    for file in sorted(os.listdir(output_base_dir)):
        if file.endswith(".mp4"):
            video_paths.append(os.path.join(output_base_dir, file))

    num_videos_needed = grid_size * grid_size
    if len(video_paths) < num_videos_needed:
        print(
            f"At least {num_videos_needed} MP4 videos are required to create a {grid_size}x{grid_size} collage."
        )
        return

    # Build the FFmpeg command
    inputs = []
    for video in video_paths[:num_videos_needed]:  # Use the first n videos
        inputs.extend(["-i", video])

    # Normalize all inputs to the same size and generate layout positions
    tile_width = tile_size
    tile_height = tile_size

    # Build input references with scaling
    filter_parts = []
    for i in range(num_videos_needed):
        filter_parts.append(f"[{i}:v]scale={tile_width}:{tile_height}[v{i}]")

    # Generate layout positions with absolute pixel coordinates
    layout_positions = []
    for row in range(grid_size):
        for col in range(grid_size):
            x = col * tile_width
            y = row * tile_height
            layout_positions.append(f"{x}_{y}")

    layout_str = "|".join(layout_positions)

    # Build the video stream references for xstack
    video_refs = "".join([f"[v{i}]" for i in range(num_videos_needed)])

    # Combine scaling and xstack filters
    scale_filter = ";".join(filter_parts)
    filter_complex = f"{scale_filter};{video_refs}xstack=inputs={num_videos_needed}:layout={layout_str}[v]"

    ffmpeg_command = [
        "ffmpeg",
        *inputs,
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "medium",
        output_path,
    ]

    try:
        subprocess.run(ffmpeg_command, check=True)
        print(f"Collage video created: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to create collage. Error: {e}")


# Example usage
if __name__ == "__main__":
    create_videos_from_folders("output_images")
    combine_videos_with_xstack(
        "output_images", "output_collage.mp4", grid_size=6, tile_size=800
    )
