# AutoBrep Data Parquet Format - Technical Details

## 1. Complete Data Flow Architecture

```
STEP Files (CAD Models)
        ↓
[occwl.io.load_step()]  → Load CAD solids from STEP
        ↓
[extract_brep_features()]  → Extract:
        │                      - Faces with UV-grids
        │                      - Edges with parametric curves
        │                      - Face-edge incidence matrix
        │                      - Bounding boxes
        ↓
[normalize_geometry()]  → Normalize to [-1, 1] range
        ↓
[serialize_array()]  → Convert numpy arrays to bytes
        ↓
[pd.DataFrame()]  → Create table with serialized columns
        ↓
[df.to_parquet()]  → Write PyArrow table to disk
        ↓
Parquet File (train/val splits)
        ↓
[load_step in training] → ParquetRowIterable
        ↓
[unpickle()] → deserialize_array() for each column
        ↓
[pre_filter()] → Validate topology & geometry
        ↓
[Training Loop]
```

---

## 2. Detailed Data Structure

### Complete Parquet Row Example

```python
{
    # Serialized geometry (bytes stored in parquet)
    'face_points_normalized': <bytes>,      # shape when deserialized: (12, 10, 10, 3)
    'edge_points_normalized': <bytes>,      # shape when deserialized: (23, 10, 3)

    # Serialized bounding boxes (bytes stored in parquet)
    'face_bbox_world': <bytes>,             # shape when deserialized: (12, 6)
    'edge_bbox_world': <bytes>,             # shape when deserialized: (23, 6)

    # Serialized topology (bytes stored in parquet)
    'face_edge_incidence': <bytes>,         # shape when deserialized: (12, 23)

    # Metadata for filtering (stored as native columns)
    'num_faces_after_splitting': 12,        # int
    'scaled_unique': True,                   # bool

    # Optional geometry constraint info
    'constraint_faces': <bytes>,            # Only if load_geom=true
}
```

---

## 3. Serialization Details

### How Arrays Are Serialized

```python
# Serialization Process (STEP → Parquet)
import io
import numpy as np

def serialize_array(array: np.ndarray) -> bytes:
    """
    Uses numpy.save() format (NPZ binary format)
    - Preserves dtype information
    - Preserves shape information
    - All in one binary blob
    """
    memfile = io.BytesIO()
    np.save(memfile, array)     # Writes numpy format header + data
    serialized = memfile.getvalue()  # Returns bytes
    return serialized

# Example:
face_grid = np.random.randn(12, 10, 10, 3).astype(np.float32)
serialized = serialize_array(face_grid)
# serialized is bytes with numpy header + float32 data

# Deserialization Process (Parquet → Training)
def deserialize_array(serialized: bytes) -> np.ndarray:
    """
    Reverses the serialization - numpy.load() automatically
    handles the header and data restoration
    """
    memfile = io.BytesIO()
    memfile.write(serialized)
    memfile.seek(0)
    array = np.load(memfile)    # Reads numpy format, returns ndarray
    return array
```

### Why This Approach?

1. **Type preservation**: float32 doesn't become float64
2. **Shape preservation**: (12, 10, 10, 3) is automatically restored
3. **Efficient**: Binary format, no JSON parsing overhead
4. **Compatible**: Works with any numpy array

### Storage in Parquet

```
Parquet Row (Arrow Table):
┌─────────────────────────────────────┐
│ face_points_normalized: binary       │ ← 4.8 MB (12×10×10×3 float32)
├─────────────────────────────────────┤
│ edge_points_normalized: binary       │ ← 2.8 KB (23×10×3 float32)
├─────────────────────────────────────┤
│ face_bbox_world: binary              │ ← 288 bytes (12×6 float32)
├─────────────────────────────────────┤
│ edge_bbox_world: binary              │ ← 552 bytes (23×6 float32)
├─────────────────────────────────────┤
│ face_edge_incidence: binary          │ ← 276 bytes (12×23 bool)
├─────────────────────────────────────┤
│ num_faces_after_splitting: int64     │ ← 8 bytes
├─────────────────────────────────────┤
│ scaled_unique: bool                  │ ← 1 byte
└─────────────────────────────────────┘
Total per sample: ~5-10 MB
```

---

## 4. Face-Edge Incidence Matrix Details

### Structure and Meaning

```python
# Example: 3 faces, 4 edges
face_edge_incidence = np.array([
    [1, 0, 1, 0],  # Face 0: touches edges 0, 2
    [1, 1, 0, 0],  # Face 1: touches edges 0, 1
    [0, 1, 1, 1],  # Face 2: touches edges 1, 2, 3
])
# Shape: (num_faces, num_edges)

# Manifold constraint (must be satisfied):
# Each edge must touch EXACTLY 2 faces
np.sum(face_edge_incidence, axis=0) == 2
# Output: [True, True, True, True]  ✓ Valid
```

### From Face-Edge Incidence to Edge-Face Incidence

```python
def convert_face_edge_adj_to_index_arrays(face_edge_adj: np.ndarray) -> np.ndarray:
    """
    Converts (num_faces, num_edges) boolean matrix
    to (num_edges, 2) index array showing which faces touch each edge
    """
    num_faces = face_edge_adj.shape[0]
    num_edges = face_edge_adj.shape[1]

    edge_face_incidence = []
    for face_attached_to_edge_flags in np.transpose(face_edge_adj):
        # Get indices of faces touching this edge
        face_indices = np.where(face_attached_to_edge_flags)[0]

        if face_indices.shape[0] < 2:
            # Pad with -1 for open edges (only 1 adjacent face)
            face_indices = np.concatenate([face_indices, np.array([-1])])

        edge_face_incidence.append(face_indices[:2])  # Keep only first 2

    return np.stack(edge_face_incidence)  # Shape: (num_edges, 2)

# Example:
# If face_edge_adj = [[1, 0, 1, 0],
#                     [1, 1, 0, 0],
#                     [0, 1, 1, 1]]
# Then edge_face_incidence = [[0, 1],   # Edge 0 touches faces 0, 1
#                              [1, 2],   # Edge 1 touches faces 1, 2
#                              [0, 2],   # Edge 2 touches faces 0, 2
#                              [2, -1]]  # Edge 3 touches face 2 (open edge)
```

---

## 5. Coordinate System: Normalized vs World

### The Problem

Raw CAD models have **different scales and positions**:
- Model A: vertices from 0 to 1000 mm
- Model B: vertices from 0 to 10 mm
- Direct training → biased toward larger objects

### The Solution: Dual Representation

```
NORMALIZED SPACE                    WORLD SPACE
(stored in parquet)                 (stored in parquet)

face_points_normalized              face_bbox_world
┌─────────────────────────┐        ┌──────────────────────┐
│ All values in [-1, 1]   │        │ [min_x, min_y, min_z │
│ relative to bbox center │        │  max_x, max_y, max_z]│
│                         │        │                      │
│ Enables:                │  +     │ Enables:             │
│ - Scale-invariant       │        │ - Absolute coords    │
│   training              │        │ - Reconstruction     │
│ - Size diversity        │        │ - Precise geometry   │
└─────────────────────────┘        └──────────────────────┘
```

### Denormalization Formula

```
# Given:
# - normalized point: p_norm ∈ [-1, 1]³
# - face bounding box: bbox = [min_x, min_y, min_z, max_x, max_y, max_z]

# Calculate:
center = (bbox[0:3] + bbox[3:6]) / 2  # Midpoint
size = bbox[3:6] - bbox[0:3]          # Diagonal vector

# Denormalize:
p_world = p_norm * (size / 2) + center

# Example:
# bbox = [0, 0, 0, 100, 100, 100]
# center = [50, 50, 50]
# size = [100, 100, 100]
# p_norm = [0, 0, 0]  (center of normalized box)
# p_world = [0, 0, 0] * 50 + [50, 50, 50] = [50, 50, 50] ✓
# p_norm = [1, 1, 1]  (corner of normalized box)
# p_world = [1, 1, 1] * 50 + [50, 50, 50] = [100, 100, 100] ✓
```

---

## 6. UV-Grid Structure

### Face UV-Grids

```python
# Face is parameterized as (u, v) surface
# Shape: (num_faces, num_u, num_v, 3)

face_points_normalized[i]  # U-V grid for face i
# has shape (num_u, num_v, 3)
#
# Example: 10×10 grid on face i
# face_points_normalized[i, :, :, :] =
# [[[-0.5, -0.5, 0.2], [-0.4, -0.5, 0.15], ..., [0.5, -0.5, 0.2]],   # v=0
#  [[-0.5, -0.4, 0.1], [-0.4, -0.4, 0.14], ..., [0.5, -0.4, 0.1]],   # v=1
#  ...
#  [[-0.5,  0.5, 0.0], [-0.4,  0.5, 0.05], ..., [0.5,  0.5, 0.0]]]   # v=9
# Each row is u-direction, each column is v-direction

# Standard resolution: 10×10 points per face
# Provides ~100 points per surface for geometry learning
```

### Edge UV-Grids

```python
# Edge is parameterized as (u) curve
# Shape: (num_edges, num_u, 3)

edge_points_normalized[j]  # U-curve for edge j
# has shape (num_u, 3)
#
# Example: 10 points on edge j
# edge_points_normalized[j, :, :] =
# [[-0.8, -0.2, 0.0],   # u=0
#  [-0.6, -0.1, 0.0],   # u=1
#  [-0.4,  0.0, 0.0],   # u=2
#  ...
#  [ 0.8,  0.2, 0.0]]   # u=9

# Standard resolution: 10 points per edge
# Provides boundary curves for surface stitching
```

---

## 7. Validation & Filtering

### Pre-Filter Checklist (as implemented in AutoBrep)

```python
def pre_filter(row):
    """All checks that must pass for a sample to be used in training"""

    # [1] Extract topology
    face_edge_adj = deserialize_array(row["face_edge_incidence"])

    # [2] Check: Not empty
    if len(face_edge_adj) == 0:
        return False  # "Empty face-edge incidence"

    # [3] Check: Valid dimension
    if len(face_edge_adj.shape) != 2:
        return False  # "Invalid adjacency matrix dimension"

    # [4] Check: Every face has at least one adjacent edge
    if np.any(np.all(~face_edge_adj, axis=1)):
        return False  # "Face with no adjacent edges"

    # [5] Check: Manifold constraint (each edge touches exactly 2 faces)
    edge_face_count = np.sum(face_edge_adj, axis=0)
    if np.any(edge_face_count != 2):
        return False  # "Non-manifold solid (edge touches != 2 faces)"

    # [6] Check: Too many edges
    if face_edge_adj.shape[1] > MAX_EDGES:
        return False  # "Too many edges for max sequence length"

    # [7] Check: Tiny faces (with tolerance for quantization)
    TOL = 1 / (2 ** (BIT_DEPTH - 1))  # Usually 1/512 ≈ 0.002
    face_bbox = deserialize_array(row["face_bbox_world"])  # (num_faces, 6)
    face_sizes = np.abs(face_bbox[:, 3:6] - face_bbox[:, 0:3])
    if np.any(np.all(face_sizes < TOL, axis=-1)):
        return False  # "Face smaller than quantization tolerance"

    # [8] Check: Tiny edges
    edge_bbox = deserialize_array(row["edge_bbox_world"])  # (num_edges, 6)
    edge_sizes = np.abs(edge_bbox[:, 3:6] - edge_bbox[:, 0:3])
    if np.any(np.all(edge_sizes < TOL, axis=-1)):
        return False  # "Edge smaller than quantization tolerance"

    return True  # Passed all checks!
```

---

## 8. Training Data Loading Pipeline

### Reading from Parquet at Training Time

```python
class ParquetRowIterable(IterableDataset):
    """Streams rows from parquet files"""

    def _iter_rows(self):
        # 1. Open parquet dataset
        dataset = ds.dataset(self.paths, format="parquet")

        # 2. Query with filter + column selection
        scanner = dataset.scanner(
            columns=self.columns,  # Only load these columns
            filter=self.filter_expr,  # e.g., num_faces_after_splitting >= 2
            batch_size=4096  # Read in batches
        )

        # 3. Stream batches
        for record_batch in scanner.to_batches():
            # Convert batch to row dicts
            for idx in range(record_batch.num_rows):
                row = {k: record_batch.column(k)[idx].as_py()
                       for k in record_batch.schema.names}

                # 4. Deserialize arrays
                row = self.unpickle(row)
                # Now row['face_points_normalized'] is np.ndarray, not bytes

                # 5. Apply post-processing & augmentation
                row = self.map_func(row, aug=True)

                yield row
```

### Example Row Flow

```
Parquet disk:
  face_points_normalized: b'\x93NUMPY\x01\x00...'  (bytes)

↓ dataset.scanner().to_batches()

Arrow batch:
  face_points_normalized: <ArrowArray of bytes>

↓ unpickle(row)

Python dict:
  face_points_normalized: np.ndarray(shape=(12,10,10,3), dtype=float32)

↓ map_func(row, aug=True)

Augmented dict:
  face_points_normalized: np.array(...) + noise/rotation
  ...

↓ Training loop consumes
```

---

## 9. Assembly Handling Strategies

### Strategy Used: **Largest Solid by Face Count**

```python
cad_solids = load_step(step_file)  # May return multiple solids

if len(cad_solids) > 1:
    # Assembly detected - pick the largest solid
    solid = max(cad_solids, key=lambda s: len(list(s.faces())))
else:
    solid = cad_solids[0]
```

### Why This Strategy?

1. **Simple**: Easy to implement
2. **Biased toward geometry**: Larger solids often contain more interesting geometry
3. **Deterministic**: No randomness or hyperparameters
4. **Data loss**: But loses assembly structure information

### Alternative Strategies (Not Used)

```python
# Strategy 1: Merge all solids
merged_solid = union_of_solids(cad_solids)
# Problem: Union operation can cause artifacts

# Strategy 2: Create multi-object representation
multi_geometry = create_multi_solid_representation(cad_solids)
# Problem: Requires changes to entire training pipeline

# Strategy 3: Process each solid separately
for solid in cad_solids:
    create_sample(solid)
# Problem: May overrepresent assemblies; increases dataset size

# Strategy 4: Skip assemblies entirely
if len(cad_solids) == 1:
    create_sample(cad_solids[0])
else:
    skip_file()
# Problem: Data loss from multi-solid STEP files
```

---

## 10. Column Selection & Dataset Directory Structure

### Recommended Parquet Directory Layout

```
data_root/
├── train/
│   ├── part_0000.parquet     # One parquet file per split
│   ├── part_0001.parquet     # For parallel reading
│   ├── part_0002.parquet
│   └── ...
└── val/
    ├── part_0000.parquet
    ├── part_0001.parquet
    └── ...
```

### Columns Queried During Training

```python
# Required columns (always loaded)
required_cols = [
    "face_points_normalized",
    "edge_points_normalized",
    "face_bbox_world",
    "edge_bbox_world",
    "face_edge_incidence",
]

# Filter columns (checked in pre_filter, must load from disk)
filter_cols = [
    "num_faces_after_splitting",  # Integer for face count filtering
    "scaled_unique",               # Boolean for data quality flag
]

# Optional columns (conditional on hyperparameters)
optional_cols = [
    "constraint_faces",  # Only if load_geom=True
]

# PyArrow pushdown filter example
filter_expr = (
    pc.field("num_faces_after_splitting") >= 2) &
    pc.field("num_faces_after_splitting") <= 30
)
# ↑ This filter is applied AT THE STORAGE LEVEL
# ↓ Only matching rows are deserialized from disk
```

---

## 11. Performance Characteristics

### Typical Parquet File Size

```
One STEP model → One parquet row
├─ face_points_normalized:    ~4.8 MB (if 12 faces × 10×10 grid)
├─ edge_points_normalized:    ~3 KB   (if 23 edges × 10 points)
├─ face_bbox_world:           ~384 B  (if 12 faces)
├─ edge_bbox_world:           ~552 B  (if 23 edges)
├─ face_edge_incidence:       ~276 B  (if 12×23 boolean matrix)
├─ num_faces_after_splitting: ~8 B
├─ scaled_unique:             ~1 B
└─ TOTAL per row:            ~5 MB

Dataset of 1M samples:
├─ Raw disk size:  ~5 TB (uncompressed)
├─ With snappy:    ~3-4 TB (compressed, default for parquet)
└─ Training time:  Streaming → ~1-2 weeks on 8×A100 GPUs
```

### PyArrow Pushdown Efficiency

```python
# Without filter (reads ALL rows, deserializes ALL arrays)
dataset.scanner(columns=[...])
# Reads: 5 MB/row × 1M rows = 5 TB from disk

# With filter (reads SOME rows, deserializes SOME arrays)
dataset.scanner(
    columns=[...],
    filter=pc.field("num_faces_after_splitting") <= 30
)
# Reads: 5 MB/row × 800k rows ≈ 4 TB from disk
# ↑ Depends on data distribution
```

---

## Summary Table

| Aspect | Details |
|--------|---------|
| **Format** | Apache Parquet (PyArrow) |
| **Serialization** | numpy.save() binary format |
| **Face Points** | (num_faces, 10, 10, 3) float32, normalized [-1,1] |
| **Edge Points** | (num_edges, 10, 3) float32, normalized [-1,1] |
| **Bounding Boxes** | (num_faces/edges, 6) float32, world coordinates |
| **Topology** | (num_faces, num_edges) boolean matrix |
| **Assembly Handling** | Use largest solid by face count |
| **Validation** | Pre-filter checks manifold, size, quantization |
| **Typical Size** | ~5 MB per model (highly variable) |
| **Typical Dataset** | 1M deduped ABC models = 5TB uncompressed |
