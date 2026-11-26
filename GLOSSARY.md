# CloudLabeller

Photogrammetric reconstruction of outcrops and samples, with labels painted on
photos or on the 3D product and carried between them, and a U-Net that
propagates labels to unlabelled photos.

## Running work

**Environment**:
The Python installation the app runs its work in, either native on Windows or
inside a WSL distribution; exactly one is active at a time.
_Avoid_: venv, runtime, backend

**Engine**:
The Python side of an open Project, living in the active Environment, that
answers the window's requests and plans its Jobs (the Tauri core starts
them); one per open Project.
_Avoid_: backend, server, kernel

**Array file**:
A large array (a cloud, a mask, label ids) passed between the window and
the Engine as raw bytes in the Project's `.arrays/` folder, named in the
message instead of sent through it.
_Avoid_: buffer, blob, exchange file

**Job**:
One long-running piece of work the user starts and can cancel: a
reconstruction step, georeferencing, training, a prediction batch.
_Avoid_: task, process, run

**GPU Job**:
A Job that wants the GPU: COLMAP feature extraction, matching and
patch-match stereo, and CNN training or prediction.

**GPU Lease**:
The single right to use the GPU, held by at most one GPU Job at a time; other
GPU Jobs wait for it, shown as "waiting for GPU".
_Avoid_: GPU lock, mutex

**Workspace**:
A scratch copy of a Project's inputs on the Environment's own disk, where a
Job does its heavy reading and writing before its products are copied back
to the Project.
_Avoid_: scratch, staging area, cache

## Viewing and labelling

**Lasso**:
A loop drawn on the 3D view that takes every point of the cloud whose
projection falls inside it, hidden or not, through the whole cloud rather
than only the points on screen. What it took can be given the active class
or, on the sparse cloud, deleted.
_Avoid_: selection tool, brush (that paints a photo)

**Frustum**:
A solved photo's camera drawn in the 3D view: a wireframe pyramid from where
it stood, coloured by the photo's status. Clicking one shows its photo;
double-clicking one looks through it.
_Avoid_: camera gizmo

**Detail level**:
How much of a thing is used. For reconstruction it is the photo size COLMAP
works at (High, Medium, Low, or a pixel cap). For viewing it is how many of
a cloud's points reach the screen (Low, Medium, High, Full); the rest of the
cloud is still labelled and exported.
_Avoid_: LOD, quality (MVS's own option)

