// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

// The imperative half of the 3D viewer: one renderer, one camera and its
// controls, geokit's layer host and the layers in it. `Viewer3D` decides what
// to show; this draws it, and only when something changed (never in a loop).
//
// Everything in the scene is recentred on `origin`, the middle of the cloud or
// mesh shown, so float32 keeps its precision on georeferenced data. Whatever
// speaks to the Engine speaks in world coordinates, so `worldViewProjection`
// puts the origin back.

import {
  type Frustum,
  FrustumLayer,
  type FrustumView,
  type CloudView as GkCloudView,
  LassoPath,
  LayerHost,
  MeshLayer,
  type MeshView,
  PointCloudLayer,
} from "@geokit/viewer";
import {
  AmbientLight,
  BufferGeometry,
  Color,
  DirectionalLight,
  Float32BufferAttribute,
  LinearSRGBColorSpace,
  Matrix4,
  MOUSE,
  PerspectiveCamera,
  Points,
  PointsMaterial,
  Scene,
  Vector3,
  WebGLRenderer,
} from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { CloudName, CloudView, MeshData } from "../engine/api";
import {
  type Bounds,
  boundsOf,
  centreOf,
  classificationFor,
  diagonalOf,
  homePose,
  type Rgb,
  recentre,
  singleNodeSource,
  type Vec3,
} from "./adapt";

/** The one node a cloud is shown as. */
export const NODE = "0-0-0-0";

/** What is on screen: a cloud or the mesh, and which versions of the Project it was read at. */
export interface Geometry {
  kind: "cloud" | "mesh";
  /** The cloud's name; null for the mesh. */
  cloud: CloudName | null;
  /** Each shown point's full-cloud index; null when all are shown, in order. */
  index: Int32Array | null;
  /** Points (or vertices) drawn. */
  shown: number;
  /** Points (or vertices) in all. */
  total: number;
  labels: Int32Array | null;
  /** The mesh's photographic colours. */
  colors: Uint8Array | null;
  /** In world coordinates. */
  bounds: Bounds;
  geomVersion: number;
  labelsVersion: number;
}

const BACKGROUND = new Color().setRGB(0.085, 0.095, 0.115, LinearSRGBColorSpace);

export class Rig {
  readonly canvas: HTMLCanvasElement;
  readonly scene = new Scene();
  readonly camera = new PerspectiveCamera(40, 1, 0.01, 1000);
  readonly renderer: WebGLRenderer;
  readonly host: LayerHost;
  readonly lasso = new LassoPath();
  controls: OrbitControls;

  geometry: Geometry | null = null;
  /** The world point that is the scene's zero. */
  origin: Vec3 = [0, 0, 0];
  /** The Project (by root) whose geometry the camera was last sent home for. */
  homedFor: string | null = null;
  /** Called with whatever a layer fails with. */
  onError: (e: unknown) => void = () => {};
  /** CSS pixels. */
  width = 1;
  height = 1;
  /** How far a frustum reaches now, for flying to one. */
  frustumReach = 1;

  private cloud: PointCloudLayer | null = null;
  private mesh: MeshLayer | null = null;
  private frusta: FrustumLayer | null = null;
  private gcpPoints: Points | null = null;
  private gcpList: { xyz: Vec3; active: boolean }[] = [];
  private classification: Float32Array | null = null;
  private look: GkCloudView = {
    colourBy: "classification",
    colourRange: [0, 1],
    filterRange: [-Infinity, Infinity],
    classMask: [],
    size: 2,
    budget: 1,
    blend: 0.6,
    ownColourClass: 0,
  };
  private meshView: MeshView = {
    colour: "#bbbbbb",
    opacity: 1,
    mode: "smooth",
    vertexColours: null,
  };
  private lassoOn = false;
  private queued = false;
  private disposed = false;
  private readonly ratio: number;

  constructor(parent: HTMLElement) {
    this.ratio = Math.min(2, window.devicePixelRatio || 1);
    this.renderer = new WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(this.ratio);
    // The cloud's shaders write colours as they are; the mesh follows them
    this.renderer.outputColorSpace = LinearSRGBColorSpace;
    this.renderer.setClearColor(BACKGROUND);
    this.canvas = this.renderer.domElement;
    this.canvas.style.touchAction = "none";
    this.canvas.style.display = "block";
    parent.appendChild(this.canvas);

    this.scene.add(this.camera);
    this.scene.add(new AmbientLight(0xffffff, 0.75 * Math.PI));
    const head = new DirectionalLight(0xffffff, 0.45 * Math.PI);
    head.position.set(0, 0, 1);
    head.target.position.set(0, 0, -1);
    this.camera.add(head, head.target);
    this.scene.add(this.lasso.object);

    this.camera.up.set(0, 0, 1);
    this.host = new LayerHost(this.renderer, this.scene);
    this.controls = this.makeControls(new Vector3());
  }

  // -- drawing, on demand ----------------------------------------------------

  /** Draw once, at the next frame. */
  invalidate(): void {
    if (this.queued || this.disposed) return;
    this.queued = true;
    requestAnimationFrame(this.frame);
  }

  private readonly frame = () => {
    this.queued = false;
    if (this.disposed) return;
    this.cloud?.apply(this.look, this.camera, this.height);
    this.renderer.render(this.scene, this.camera);
  };

  resize(width: number, height: number): void {
    this.width = Math.max(1, Math.floor(width));
    this.height = Math.max(1, Math.floor(height));
    this.renderer.setSize(this.width, this.height);
    this.camera.aspect = this.width / this.height;
    this.camera.updateProjectionMatrix();
    this.invalidate();
  }

  // -- camera ----------------------------------------------------------------

  private makeControls(target: Vector3): OrbitControls {
    this.controls?.dispose();
    const c = new OrbitControls(this.camera, this.canvas);
    c.enableDamping = false;
    c.screenSpacePanning = true;
    c.mouseButtons = {
      LEFT: this.lassoOn ? null : MOUSE.ROTATE,
      MIDDLE: MOUSE.DOLLY,
      RIGHT: MOUSE.PAN,
    };
    c.target.copy(target);
    c.addEventListener("change", () => this.invalidate());
    c.update();
    return c;
  }

  /** Put the camera at `position` looking at `target`, `up` upright. Positions are world coordinates. */
  setView(position: Vec3, target: Vec3, up: Vec3): void {
    const o = this.origin;
    const turned = this.camera.up.distanceToSquared(new Vector3(...up)) > 1e-12;
    this.camera.up.set(...up);
    this.camera.position.set(position[0] - o[0], position[1] - o[1], position[2] - o[2]);
    // OrbitControls fixes the up it sees when it is made
    if (turned)
      this.controls = this.makeControls(
        new Vector3(target[0] - o[0], target[1] - o[1], target[2] - o[2]),
      );
    else {
      this.controls.target.set(target[0] - o[0], target[1] - o[1], target[2] - o[2]);
      this.controls.update();
    }
    this.invalidate();
  }

  /** Where the camera is, in world coordinates. */
  private fit(diagonal: number): void {
    this.camera.near = Math.max(diagonal / 5000, 1e-6);
    this.camera.far = Math.max(diagonal * 60, 1);
    this.camera.updateProjectionMatrix();
  }

  /** The home view of what is shown: from the south-east, elevated, `up` upright. */
  home(up: Vec3): void {
    const g = this.geometry;
    if (!g) return;
    const h = homePose(g.bounds, up);
    this.camera.fov = 40;
    this.fit(h.diagonal);
    this.setView(h.position, h.target, h.up);
  }

  /** Arm or disarm the lasso: left drag stops turning the view. */
  setLasso(on: boolean): void {
    this.lassoOn = on;
    this.controls.mouseButtons = {
      LEFT: on ? null : MOUSE.ROTATE,
      MIDDLE: MOUSE.DOLLY,
      RIGHT: MOUSE.PAN,
    };
    this.canvas.style.cursor = on ? "crosshair" : "";
    if (!on) this.lasso.set([], this.width, this.height);
    this.invalidate();
  }

  /** The projection times the view, world to clip space, column-major, for the Engine's `cloud.select`. */
  worldViewProjection(): number[] {
    this.camera.updateMatrixWorld();
    const m = new Matrix4().multiplyMatrices(
      this.camera.projectionMatrix,
      this.camera.matrixWorldInverse,
    );
    // the scene is the world minus the origin
    m.multiply(new Matrix4().makeTranslation(-this.origin[0], -this.origin[1], -this.origin[2]));
    return Array.from(m.elements);
  }

  // -- geometry --------------------------------------------------------------

  /** Move the scene's zero, keeping what the camera sees where it is. */
  private reorigin(next: Vec3): void {
    const first = this.geometry === null;
    const d = new Vector3(
      this.origin[0] - next[0],
      this.origin[1] - next[1],
      this.origin[2] - next[2],
    );
    this.origin = next;
    this.setFrusta(null, null);
    this.setGcps(this.gcpList);
    if (first) return;
    this.camera.position.add(d);
    this.controls.target.add(d);
    this.controls.update();
  }

  private dropGeometry(): void {
    for (const l of [this.cloud, this.mesh]) {
      if (!l) continue;
      this.host.remove(l);
      l.dispose();
    }
    this.cloud = null;
    this.mesh = null;
    this.classification = null;
  }

  /** Show nothing; the next geometry is a first one again. */
  clearGeometry(): void {
    this.dropGeometry();
    this.setFrusta(null, null);
    this.geometry = null;
    this.homedFor = null;
    this.invalidate();
  }

  setCloud(
    cloud: CloudName,
    v: CloudView,
    labels: Int32Array,
    geomVersion: number,
    labelsVersion: number,
  ): void {
    const bounds = boundsOf(v.xyz);
    if (!bounds) {
      this.clearGeometry();
      return;
    }
    this.dropGeometry();
    this.reorigin(centreOf(bounds));
    const shown = Math.floor(v.xyz.length / 3);
    const classification = classificationFor(labels, v.index);
    const layer = new PointCloudLayer(
      singleNodeSource(v.xyz, v.rgb, classification, this.origin),
      this.host.ids,
    );
    layer.onChange = () => this.invalidate();
    layer.onError = (_key, e) => this.onError(e);
    this.host.add(layer);
    this.cloud = layer;
    this.classification = classification;
    this.look = { ...this.look, budget: shown + 1 };
    this.geometry = {
      kind: "cloud",
      cloud,
      index: v.index,
      shown,
      total: v.n_points,
      labels,
      colors: null,
      bounds,
      geomVersion,
      labelsVersion,
    };
    this.fit(diagonalOf(bounds));
    this.invalidate();
  }

  setMesh(
    m: MeshData,
    labels: Int32Array | null,
    geomVersion: number,
    labelsVersion: number,
  ): void {
    const bounds = boundsOf(m.vertices);
    if (!bounds) {
      this.clearGeometry();
      return;
    }
    this.dropGeometry();
    this.reorigin(centreOf(bounds));
    const layer = new MeshLayer(
      { vertices: recentre(m.vertices, this.origin), faces: m.faces },
      { ...this.meshView, vertexColours: null },
      this.host.ids,
    );
    this.host.add(layer);
    this.mesh = layer;
    const n = Math.floor(m.vertices.length / 3);
    this.geometry = {
      kind: "mesh",
      cloud: null,
      index: null,
      shown: n,
      total: n,
      labels,
      colors: m.colors,
      bounds,
      geomVersion,
      labelsVersion,
    };
    this.fit(diagonalOf(bounds));
    this.invalidate();
  }

  /** New labels for what is shown: the cloud's classes change in place, the mesh needs its colours again. */
  setLabels(labels: Int32Array, labelsVersion: number): void {
    const g = this.geometry;
    if (!g) return;
    g.labels = labels;
    g.labelsVersion = labelsVersion;
    if (this.cloud && this.classification) {
      const next = classificationFor(labels, g.index);
      if (next.length === this.classification.length) {
        this.classification.set(next);
        this.cloud.setClassification(NODE, this.classification);
      }
    }
    this.invalidate();
  }

  // -- looks -----------------------------------------------------------------

  /** How the cloud is drawn; `size` is in CSS pixels. */
  setCloudLook(look: Partial<GkCloudView>): void {
    const { size, ...rest } = look;
    this.look = {
      ...this.look,
      ...rest,
      ...(size === undefined ? {} : { size: size * this.ratio }),
    };
    this.invalidate();
  }

  /** Mark points of the cloud shown (positions among those shown), or none. */
  setMarks(shown: Uint32Array | null, colour: Rgb): void {
    this.look = { ...this.look, markColour: colour };
    if (!this.cloud) return;
    if (shown && shown.length > 0) this.cloud.mark(NODE, shown);
    else this.cloud.clearMarks();
    this.invalidate();
  }

  setMeshColours(rgb: Float32Array): void {
    this.meshView = { ...this.meshView, vertexColours: rgb };
    this.mesh?.apply(this.meshView);
    this.invalidate();
  }

  // -- GCPs ------------------------------------------------------------------

  /** Where the GCPs are (world coordinates), drawn over everything; the
   *  active one stands out. */
  setGcps(list: { xyz: Vec3; active: boolean }[]): void {
    this.gcpList = list;
    if (this.gcpPoints) {
      this.scene.remove(this.gcpPoints);
      this.gcpPoints.geometry.dispose();
      (this.gcpPoints.material as PointsMaterial).dispose();
      this.gcpPoints = null;
    }
    if (list.length > 0) {
      const pos = new Float32Array(list.length * 3);
      const col = new Float32Array(list.length * 3);
      list.forEach((g, i) => {
        for (let k = 0; k < 3; k++) pos[i * 3 + k] = (g.xyz[k] ?? 0) - (this.origin[k] ?? 0);
        const c = g.active ? [1, 0.83, 0.24] : [1, 0.48, 0.45];
        col.set(c, i * 3);
      });
      const geometry = new BufferGeometry();
      geometry.setAttribute("position", new Float32BufferAttribute(pos, 3));
      geometry.setAttribute("color", new Float32BufferAttribute(col, 3));
      const material = new PointsMaterial({
        size: 11 * this.ratio,
        sizeAttenuation: false,
        vertexColors: true,
        depthTest: false,
        transparent: true,
      });
      this.gcpPoints = new Points(geometry, material);
      this.gcpPoints.renderOrder = 10;
      this.scene.add(this.gcpPoints);
    }
    this.invalidate();
  }

  // -- cameras ---------------------------------------------------------------

  /** Replace the frusta (null or none: remove them). */
  setFrusta(frusta: Frustum[] | null, view: FrustumView | null): void {
    if (this.frusta) {
      this.host.remove(this.frusta);
      this.frusta.dispose();
      this.frusta = null;
    }
    if (frusta && frusta.length > 0 && view) {
      this.frusta = this.host.add(new FrustumLayer(frusta, view, this.host.ids));
      this.frustumReach = view.size;
    }
    this.invalidate();
  }

  get hasFrusta(): boolean {
    return this.frusta !== null;
  }

  applyFrusta(view: FrustumView): void {
    if (!this.frusta) return;
    this.frustumReach = view.size;
    this.frusta.apply({ ...view, dotSize: (view.dotSize ?? 6) * this.ratio });
    this.invalidate();
  }

  /** The frustum under a pixel (CSS px from the canvas's top left), or null. Only frusta are asked. */
  pickFrustum(x: number, y: number): number | null {
    const frusta = this.frusta;
    if (!frusta) return null;
    const aside = [this.cloud, this.mesh].filter(
      (l): l is PointCloudLayer | MeshLayer => l !== null,
    );
    for (const l of aside) this.host.remove(l);
    try {
      const hit = this.host.pickAt(x, y, this.camera);
      if (hit && hit.layer === frusta && hit.target.kind === "element") return hit.target.index;
      return null;
    } finally {
      for (const l of aside) this.host.add(l);
    }
  }

  /** Fly (at once) to where frustum `i` looked from: its position, direction, up and field of view. */
  flyToFrustum(i: number): void {
    const pose = this.frusta?.viewFrom(i);
    if (!pose) return;
    const o = this.origin;
    const at = new Vector3(...pose.position);
    const dir = new Vector3(...pose.target).sub(at).normalize();
    const reach = Math.max(this.frustumReach * 2, 1e-3);
    const target = at.clone().addScaledVector(dir, reach);
    this.camera.fov = pose.fov;
    this.camera.updateProjectionMatrix();
    this.setView(
      [at.x + o[0], at.y + o[1], at.z + o[2]],
      [target.x + o[0], target.y + o[1], target.z + o[2]],
      pose.up,
    );
  }

  dispose(): void {
    this.disposed = true;
    this.controls.dispose();
    this.dropGeometry();
    this.setFrusta(null, null);
    this.setGcps([]);
    this.host.dispose();
    this.lasso.dispose();
    this.renderer.dispose();
    this.renderer.forceContextLoss();
    this.canvas.remove();
  }
}
