import argparse
import io
import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from OCC.Core.STEPControl import STEPControl_Reader
from occwl.io import load_step
from occwl.uvgrid import ugrid, uvgrid
from occwl.solid import Solid
from occwl.shell import Shell
from occwl.compound import Compound
from occwl.entity_mapper import EntityMapper

import warnings
warnings.filterwarnings('ignore')

# Copy of AutoBrep serialization functions (autobrep.data.serialize)
def serialize_array(array: np.ndarray) -> bytes:
    """Serialize a numpy array to bytes using numpy format."""
    memfile = io.BytesIO()
    np.save(memfile, array)
    return memfile.getvalue()

def deserialize_array(serialized: bytes) -> np.ndarray:
    """Deserialize bytes back to numpy array."""
    memfile = io.BytesIO(serialized)
    return np.load(memfile)


def extract_brep_features(solids, filename=""):
    """
    Extract faces, edges, and face-edge adjacency from B-rep solids.
    Follows AutoBrep specs: 32x32 face grids, 32-point edge curves.

    Returns:
        face_pts: (num_faces, 32, 32, 3) float32 normalized to [-1,1]
        edge_pts: (num_edges, 32, 3) float32 normalized to [-1,1]
        edgeFace_IncM: edge-face adjacency mapping
        num_faces, num_edges: counts
    """
    try:
        from occwl.compound import Compound
        from occwl.shell import Shell
        from occwl.entity_mapper import EntityMapper

        # Handle both single solid and list of solids
        if isinstance(solids, list):
            solid_list = solids
        else:
            solid_list = [solids]

        all_faces = {}
        all_edges = {}
        all_incidence = {}
        face_offset = 0
        edge_offset = 0

        for solid in solid_list:
            try:
                # Split closed surfaces/curves (skip if fails)
                try:
                    solid = solid.split_all_closed_faces(num_splits=0)
                except:
                    pass
                try:
                    solid = solid.split_all_closed_edges(num_splits=0)
                except:
                    pass

                mapper = None
                try:
                    mapper = EntityMapper(solid)
                except:
                    pass

                # Extract faces
                solid_faces = {}
                if mapper:
                    try:
                        for face in solid.faces():
                            try:
                                face_idx = mapper.face_index(face)
                                points = uvgrid(face, method="point", num_u=32, num_v=32)  # AutoBrep: 32x32
                                if points is not None and len(points) > 0:
                                    solid_faces[face_idx] = points
                            except:
                                continue
                    except:
                        solid_faces = {}
                else:
                    try:
                        for face_idx, face in enumerate(solid.faces()):
                            try:
                                points = uvgrid(face, method="point", num_u=32, num_v=32)  # AutoBrep: 32x32
                                if points is not None and len(points) > 0:
                                    solid_faces[face_idx] = points
                            except:
                                # Fallback: synthetic face from edges
                                try:
                                    edges_of_face = list(face.edges())
                                    if edges_of_face:
                                        edge_pts_list = []
                                        for edge in edges_of_face:
                                            try:
                                                edge_pts = ugrid(edge, method="point", num_u=16)
                                                if edge_pts is not None:
                                                    edge_pts_list.append(edge_pts)
                                            except:
                                                continue
                                        if edge_pts_list:
                                            combined = np.concatenate([pts for pts in edge_pts_list], axis=0)
                                            if len(combined) >= 32:
                                                points = np.tile(combined[:32].reshape(32, 1, 3), (1, 32, 1))
                                                solid_faces[face_idx] = points
                                except:
                                    pass
                    except:
                        pass

                # Extract edges
                solid_edges = {}
                solid_incidence = {}

                if mapper:
                    try:
                        for edge in solid.edges():
                            try:
                                if not edge.has_curve():
                                    continue
                                connected_faces = list(solid.faces_from_edge(edge))
                                if len(connected_faces) != 2:
                                    continue
                                try:
                                    if edge.seam(connected_faces[0]) or edge.seam(connected_faces[1]):
                                        continue
                                except:
                                    pass

                                left_face, right_face = edge.find_left_and_right_faces(connected_faces)
                                if left_face is None or right_face is None:
                                    continue

                                edge_idx = mapper.edge_index(edge)
                                points = ugrid(edge, method="point", num_u=32)  # AutoBrep: 32 points
                                if points is not None and len(points) > 0:
                                    solid_edges[edge_idx] = points
                                    left_idx = mapper.face_index(left_face)
                                    right_idx = mapper.face_index(right_face)
                                    solid_incidence[edge_idx] = [left_idx, right_idx]
                            except:
                                continue
                    except:
                        solid_edges = {}
                else:
                    try:
                        for edge_idx, edge in enumerate(solid.edges()):
                            try:
                                if not edge.has_curve():
                                    continue
                                connected_faces = list(solid.faces_from_edge(edge))
                                if len(connected_faces) != 2:
                                    continue
                                try:
                                    if edge.seam(connected_faces[0]) or edge.seam(connected_faces[1]):
                                        continue
                                except:
                                    pass

                                points = ugrid(edge, method="point", num_u=32)  # AutoBrep: 32 points
                                if points is not None and len(points) > 0:
                                    solid_edges[edge_idx] = points
                            except:
                                continue
                    except:
                        pass

                # Add to global lists with offsets
                for face_idx, points in solid_faces.items():
                    all_faces[face_offset + face_idx] = points
                for edge_idx, points in solid_edges.items():
                    all_edges[edge_offset + edge_idx] = points
                for edge_idx, (left_idx, right_idx) in solid_incidence.items():
                    all_incidence[edge_offset + edge_idx] = [face_offset + left_idx, face_offset + right_idx]

                face_offset += len(solid_faces)
                edge_offset += len(solid_edges)

            except:
                continue

        if not all_faces or not all_edges:
            return None, None, None, 0, 0

        # Stack face and edge points
        face_pts = np.stack([all_faces[i] for i in sorted(all_faces.keys())], axis=0)  # (F, 32, 32, 3)
        edge_pts = np.stack([all_edges[i] for i in sorted(all_edges.keys())], axis=0)  # (E, 32, 3)

        num_faces = len(all_faces)
        num_edges = len(all_edges)

        return face_pts, edge_pts, all_incidence, num_faces, num_edges

    except Exception:
        return None, None, None, 0, 0

def compute_bboxes(face_pts, edge_pts):
    """
    Compute axis-aligned bounding boxes (AutoBrep Section 5.1).
    Stores bounding boxes in normalized [-1, 1] space.

    Args:
        face_pts: (num_faces, 32, 32, 3) - normalized to [-1, 1]
        edge_pts: (num_edges, 32, 3) - normalized to [-1, 1]

    Returns:
        face_bboxes: (num_faces, 6) - [xmin, ymin, zmin, xmax, ymax, zmax] float32
        edge_bboxes: (num_edges, 6)
    """
    face_bboxes = []
    for face in face_pts:
        points = face.reshape(-1, 3)
        min_pt = np.min(points, axis=0)
        max_pt = np.max(points, axis=0)
        face_bboxes.append(np.concatenate([min_pt, max_pt]))

    edge_bboxes = []
    for edge in edge_pts:
        min_pt = np.min(edge, axis=0)
        max_pt = np.max(edge, axis=0)
        edge_bboxes.append(np.concatenate([min_pt, max_pt]))

    return np.array(face_bboxes, dtype=np.float32), np.array(edge_bboxes, dtype=np.float32)

def normalize_geometry(face_pts, edge_pts):
    """
    Normalize face and edge points to [-1, 1] range using global bounding box.
    Keeps bounding boxes in normalized space for AutoBrep.

    Args:
        face_pts: (num_faces, 32, 32, 3)
        edge_pts: (num_edges, 32, 3)

    Returns:
        face_pts_norm, edge_pts_norm: Normalized points in [-1, 1]
    """
    # Combine all points to compute global bounding box
    all_points = np.concatenate([
        face_pts.reshape(-1, 3),
        edge_pts.reshape(-1, 3)
    ], axis=0)

    # Global normalization to [-1, 1]
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)
    global_center = (min_vals + max_vals) / 2.0
    global_scale = np.max(max_vals - min_vals) / 2.0

    if global_scale == 0:
        return face_pts, edge_pts

    face_pts_norm = (face_pts - global_center[np.newaxis, np.newaxis, np.newaxis, :]) / global_scale
    edge_pts_norm = (edge_pts - global_center[np.newaxis, np.newaxis, :]) / global_scale

    return face_pts_norm.astype(np.float32), edge_pts_norm.astype(np.float32)

def build_face_edge_incidence(num_faces, num_edges, edgeFace_IncM):
    """
    Build face-edge incidence matrix: (num_faces, num_edges) boolean matrix.

    Args:
        num_faces: Total number of faces
        num_edges: Total number of edges
        edgeFace_IncM: Dict mapping edge_idx to [face_idx1, face_idx2]

    Returns:
        incidence_matrix: (num_faces, num_edges) boolean array
    """
    incidence = np.zeros((num_faces, num_edges), dtype=bool)

    for edge_idx, (face_idx1, face_idx2) in edgeFace_IncM.items():
        if edge_idx < num_edges:
            if face_idx1 < num_faces:
                incidence[face_idx1, edge_idx] = True
            if face_idx2 < num_faces:
                incidence[face_idx2, edge_idx] = True

    return incidence

def process_step_file(step_path, verbose=False):
    """Process a single STEP file and extract B-Rep geometry."""
    try:
        cad_solids = load_step(str(step_path))

        if not cad_solids or len(cad_solids) == 0:
            if verbose:
                print(f"  ❌ {step_path.name}: No solids found")
            return None

        # BrepGen/AutoBrep filtering: keep only solids with <=30 faces
        filtered_solids = [s for s in cad_solids if len(list(s.faces())) <= 30]

        if len(filtered_solids) == 0:
            solids_to_process = cad_solids
            if verbose:
                print(f"  ⚠️  {step_path.name}: All {len(cad_solids)} solids exceed 30 faces, using all")
        else:
            solids_to_process = filtered_solids
            if verbose and len(filtered_solids) < len(cad_solids):
                print(f"  ℹ️  {step_path.name}: {len(cad_solids)} solids -> filtered to {len(filtered_solids)} (<=30 faces)")

        if verbose and len(cad_solids) > 1:
            total_faces = sum(len(list(s.faces())) for s in solids_to_process)
            print(f"    Merging {len(solids_to_process)} solids (~{total_faces} total faces)")

        # Extract B-rep features
        face_pts, edge_pts, edgeFace_IncM, num_faces, num_edges = extract_brep_features(solids_to_process, step_path.name)

        if face_pts is None or num_faces < 1 or num_edges < 1:
            if verbose:
                print(f"  ❌ {step_path.name}: Insufficient geometry (faces={num_faces}, edges={num_edges})")
            return None

        # Normalize to [-1, 1]
        face_pts_norm, edge_pts_norm = normalize_geometry(face_pts, edge_pts)

        # Compute bboxes in normalized space
        face_bboxes, edge_bboxes = compute_bboxes(face_pts_norm, edge_pts_norm)

        # Build face-edge incidence matrix
        face_edge_incidence = build_face_edge_incidence(num_faces, num_edges, edgeFace_IncM)

        return {
            'face_points_normalized': serialize_array(face_pts_norm),
            'edge_points_normalized': serialize_array(edge_pts_norm),
            'face_bbox_world': serialize_array(face_bboxes),
            'edge_bbox_world': serialize_array(edge_bboxes),
            'face_edge_incidence': serialize_array(face_edge_incidence),
            'num_faces_after_splitting': num_faces,
            'scaled_unique': True,
            'filename': step_path.name,
        }
    except Exception as e:
        if verbose:
            print(f"  ❌ {step_path.name}: {type(e).__name__}: {str(e)[:100]}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Preprocess STEP files to AutoBrep Parquet format")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input STEP files")
    parser.add_argument("--output_file", type=str, required=True, help="Output path for the Parquet file")
    args = parser.parse_args()

    step_dir = Path(args.input_dir)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Input STEP files: {step_dir}")
    print(f"Output Parquet file: {output_file}")

    print("Processing STEP files to point clouds...\n")

    step_files = list(step_dir.glob("*.step"))
    print(f"Found {len(step_files)} STEP files\n")

    data_list = []
    success_count = 0

    for step_file in tqdm(step_files, desc="Processing"):
        result = process_step_file(step_file, verbose=True)

        if result is not None:
            data_list.append(result)
            success_count += 1

        if len(data_list) > 0 and len(data_list) % 20 == 0:
            print(f"  Processed {len(data_list)} successful conversions")

    print(f"\n✓ Successfully processed: {success_count}/{len(step_files)} files")

    # Save to Parquet
    if data_list:
        df = pd.DataFrame(data_list)
        df.to_parquet(output_file, engine='pyarrow', index=False)

        # Verify
        df_check = pd.read_parquet(output_file)
        print(f"\n✅ Saved {len(df_check)} samples to {output_file}")
        print(f"  Shape: {df_check.shape}")
        print(f"  Size: {output_file.stat().st_size / 1e6:.1f} MB")
    else:
        print("❌ No valid samples to save")

if __name__ == "__main__":
    main()
