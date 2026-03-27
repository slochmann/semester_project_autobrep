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
    output_prefix="output_collage",
    grid_size=6,
    tile_size=800,
):
    """Combine MP4 videos in the output directory into multiple collages using FFmpeg's xstack filter.
    If n x grid_size^2 videos are available, create n GIFs.

    Args:
        output_base_dir: Directory containing MP4 files
        output_prefix: Prefix for output collage MP4 files
        grid_size: Size of the grid (e.g., 3 for 3x3, 2 for 2x2)
        tile_size: Size of each tile in pixels (default 800 for higher resolution)
    """
    video_paths = []

    # Collect all MP4 videos in the output directory
    for file in sorted(os.listdir(output_base_dir)):
        if file.endswith(".mp4"):
            video_paths.append(os.path.join(output_base_dir, file))

    videos_per_collage = grid_size * grid_size
    num_collages = len(video_paths) // videos_per_collage

    if num_collages == 0:
        print(
            f"At least {videos_per_collage} MP4 videos are required to create a {grid_size}x{grid_size} collage."
        )
        return

    print(f"Creating {num_collages} collage(s) from {len(video_paths)} videos...")

    # Create multiple collages
    for collage_idx in range(num_collages):
        start_idx = collage_idx * videos_per_collage
        end_idx = start_idx + videos_per_collage
        videos_for_collage = video_paths[start_idx:end_idx]

        output_path = f"{output_prefix}_{collage_idx}.mp4"

        # Build the FFmpeg command
        inputs = []
        for video in videos_for_collage:
            inputs.extend(["-i", video])

        # Normalize all inputs to the same size and generate layout positions
        tile_width = tile_size
        tile_height = tile_size

        # Build input references with scaling
        filter_parts = []
        for i in range(videos_per_collage):
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
        video_refs = "".join([f"[v{i}]" for i in range(videos_per_collage)])

        # Combine scaling and xstack filters
        scale_filter = ";".join(filter_parts)
        filter_complex = f"{scale_filter};{video_refs}xstack=inputs={videos_per_collage}:layout={layout_str}[v]"

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
            print(f"Failed to create collage {collage_idx}. Error: {e}")


# Example usage
if __name__ == "__main__":
    create_videos_from_folders("output_images")
    combine_videos_with_xstack("output_images", grid_size=6, tile_size=800)
