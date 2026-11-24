# UI, viewer and Environment code is shared with GeoScape through a permissive kit

CloudLabeller adopts GeoScape's UI system, but the two can't share code
directly: GeoScape is closed and commercial, and only accepts
permissive-licensed dependencies, while CloudLabeller is public under GPL-3. So
the generic pieces are moved out of GeoScape into a third public repository,
`rockanalytica/geokit`, under Apache-2.0 with RockAnalytica as copyright
holder, and both apps depend on it at a pinned version. Its first contents:

- design tokens and brand slots;
- the viewer layers that depend only on three.js (point cloud, mesh, value
  shader);
- the Rust Environment manager and GPU Lease, taking the list of packages to
  install as a setting.

## Considered Options

- **CloudLabeller as an app inside GeoScape's monorepo**: free reuse, but
  CloudLabeller would stop being public GPL software with a Zenodo DOI.
- **Copying the code into CloudLabeller**: the two copies drift apart, which
  is exactly the repetition this decision exists to avoid.

## Consequences

- GeoScape is extracted first. It switches to the kit and releases before
  CloudLabeller takes anything from it, so there is one source of truth.
- Code flows one way only. Apache-2.0 code may go into GPL-3 CloudLabeller.
  Code written in CloudLabeller never goes back into the kit or GeoScape.
  Anything both apps need, such as camera frusta, single-point picking and
  lasso selection, is written in the kit.
- Everything in the kit is public. What GeoScape is willing to publish is
  decided one module at a time.
- The kit names no domain from either app (no Drillholes, no Labels). It
  passes GeoScape's licence gate.
