# Reconstructions are kept, and aligned by a stored transform

Status: accepted (2026-11-26), for 0.4.0.

Until 0.3.0 a Project had one reconstruction. Running photogrammetry again
overwrote it, and georeferencing wrote its similarity transform into every
product in place (`photogrammetry/project_transform.py`), after moving the
COLMAP model to `sparse_prealigned/`. From 0.4.0 a Project holds Surveys,
each with any number of Reconstructions:

1. **Nothing is overwritten.** Running any step makes a new Reconstruction.
   It shares the earlier steps' products with the one it came from instead
   of copying them. A product file is removed only when the last
   Reconstruction that uses it is deleted.
2. **An alignment is a transform kept with the Reconstruction**, never
   applied to its files. Its products stay in their own SfM frame. The
   viewer, export, GCP picking and comparisons between Surveys apply
   `X' = s · R @ X + t` as they read. Fitting it again, from GCPs or EXIF
   GPS, replaces only the transform.

## Considered Options

- **Align by making a new, transformed Reconstruction**: every reader would
  stay simple, but each alignment would copy the dense cloud and mesh
  (gigabytes), and every re-fit with one more GCP would copy them again.
- **Keep transforming in place, as in 0.3.0**: this contradicts (1). It
  also needs the pre-alignment backup and the "already georeferenced"
  refusal, which exist only because in-place aligning can't be undone.

## Consequences

- MVS and meshing run in the SfM frame, so the products a Reconstruction
  shares with its parent never need transforming.
- Every reader of geometry goes through one function that applies the
  Reconstruction's transform. Reading the arrays directly is a bug.
- The `job.georef.already` refusal and `sparse_prealigned/` go away.
  Reprojecting to another CRS changes the Project's frame, not the files.
- Opening a 0.3.0 Project that was georeferenced in place keeps its
  products as they are, with an identity transform and the frame recorded
  in the manifest.
