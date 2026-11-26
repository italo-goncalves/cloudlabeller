# Roadmap

*This file decides what comes next and in what order. It holds no design:
the design lives in [DESIGN.md](../DESIGN.md) and the decisions in
[adr/](adr/). **Now** is the next release, with what "done" means. **Next**
lists the themes after it, in order, each with the reason it comes where it
does. **Later** is unordered. **Dropped** records what was taken off, and
why, so it isn't proposed again. An item moves by editing this file, and
leaves the file when it is built.*

Last released: **0.3.0** (2026-11-26), the Desktop app. The Qt window is
replaced by a Tauri app ([desktop.md](desktop.md)) over a JSON-RPC Engine
([engine.md](engine.md)); it shares geokit with GeoScape, installs per user,
updates itself, and sets up its Python Environment unattended, in WSL on the
GPU when it can ([ADR-0001](adr/0001-shared-permissive-kit-from-geoscape.md)
to [ADR-0004](adr/0004-the-tauri-core-starts-every-job.md)).

---

## Now

Not chosen yet: by default, the first theme in **Next**.

---

## Next

1. **Georeferencing Tier 2: ground control points picked on the cloud.**
   - Pick points on the dense cloud and enter their surveyed coordinates (a
     CSV import that knows its CRS).
   - Refit the similarity with the existing RANSAC fit.
   - Report a residual for each point, and support check points.
   - Re-align from the backed-up pre-alignment model.

   *Why here:* this needs point picking on the cloud, which 0.3.0 builds in
   the kit. Building it in Qt first would mean building it twice.
2. **Georeferencing Tier 3: the full treatment.**
   - Mark control points on the photos and triangulate them.
   - Bundle adjustment constrained by the control points (may need pycolmap
     work).
   - DJI RTK/PPK coordinates, weighted by each photo's accuracy.
   - Orthophoto and DEM export.

   *Why here:* it builds on Tier 2's control points.

## Later

- **Active learning**: the model proposes which uncertain photos to label
  next.
- **Out-of-core clouds**: COPC streaming once clouds outgrow the viewer's
  point budget. This needs a permissive COPC writer, such as copclib or PDAL.
- **Assisted 2D labelling**: polygon, superpixel or click-to-segment
  (SAM-style).
- **macOS and Linux installers**, when someone asks for them.
- **TensorFlow beyond 2.19.** This touches the locked U-Net, so it is a
  separate decision.

## Dropped

- **The portable zip** (941 MiB): replaced in 0.3.0 by an installer of tens
  of MB that downloads the Environment on first run (ADR-0002).
- **Restyling the Qt app** with GeoScape's tokens: it yields no installer and
  nothing shared with GeoScape (ADR-0002).
