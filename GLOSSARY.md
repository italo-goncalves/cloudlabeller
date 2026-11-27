# CloudLabeller

Photogrammetric reconstruction of outcrops and samples, with labels painted on
photos or on the 3D product and carried between them, and a U-Net that
propagates labels to unlabelled photos.

## Projects and Surveys

**Project**:
A folder holding everything about one site: its Surveys, one list of
Classes shared by all of them, and the models trained on them.
_Avoid_: workspace (a Job's scratch copy), job

**Survey**:
One acquisition of a Project's site: the photos taken together on one flight
or field day, with their own camera and settings, reconstructed on their
own. It carries a name, a timestamp (by default the median of its photos'
EXIF times), and optionally an operator, a drone or camera model, and notes.
Every Project has at least one.
_Avoid_: flight, campaign, dataset, epoch

**Photo**:
One picture in a Survey, known by its Survey and its file name; two Surveys
may each have a photo of the same name.
_Avoid_: frame, shot

**Reconstruction**:
One run of photogrammetry for a Survey and what it produced: sparse cloud
and cameras, and when run that far, dense cloud and mesh. Running any step
again makes a new Reconstruction that shares the earlier steps' products
with the one it came from; nothing is overwritten. A Survey has one
**active** Reconstruction, the one shown, labelled and exported.
_Avoid_: version, model (a trained network), run

**Current Survey**:
The Survey the window is showing and editing; one at a time.

**Alignment**:
The scale, rotation and shift that carry a Reconstruction from its own frame
into the Project's frame, fitted to its GCPs or, short of three GCPs, to
its photos' EXIF GPS. It is kept with the Reconstruction, never applied to
its files.
_Avoid_: georeferencing (the act of fitting it), registration

**Project frame**:
The one coordinate frame every Survey is aligned into: a local ENU origin
or a projected CRS. It comes from the GCPs' surveyed coordinates when any
has them, otherwise from a chosen **reference Survey**.
_Avoid_: world, global frame

**GCP**:
A ground point that can be recognised in the field and in the photos,
belonging to the Project, with a name and optionally surveyed coordinates
(in a stated CRS, with an accuracy). A *control* GCP is used to fit
Alignments; a *check* GCP is only measured against them. A GCP without
coordinates still ties Surveys to each other.
_Avoid_: tie point, marker, target

**Mark**:
Where a GCP was identified: a pixel on a Photo (belonging to the Survey,
and located in 3D by triangulating two or more Marks with a
Reconstruction's cameras) or one of the points of a Reconstruction's cloud,
taken as the point nearest to where the user clicked. A cloud Mark wins over
triangulated ones.
_Avoid_: observation, pick, measurement

**Class**:
One kind of thing a label can say, with a name, a colour and an id from 0
to N-1; one list per Project. Unlabelled is not a Class.
_Avoid_: category, label (the assignment, not the kind)

**Mask**:
A Photo's labels, one Class (or unlabelled) per pixel, painted or imported
by the user. It belongs to the Survey, so every Reconstruction of that
Survey can use it; a Reconstruction's cloud and mesh labels belong to the
Reconstruction.
_Avoid_: annotation, segmentation

**Tile**:
A fixed-size square of photo pixels the network trains and predicts on, cut
from a Photo on an overlapping grid and shrunk by the model's factor before
it reaches the network; photos of any size or shape yield Tiles of the same
size. Predictions of overlapping Tiles are blended into one per Photo.
_Avoid_: patch, crop, chip, window

**Normal map**:
The dense cloud's surface normals drawn into a Photo: each pixel holds the
normal of the point it sees, shown in false colour (red east, green north,
blue up). Only a Photo with a camera in a Survey with a dense cloud has one.
It can also be given to the network beside the Photo's colours, an
experimental option.
_Avoid_: normal image, slope map

**Uncertainty**:
How unsure a prediction is at each pixel of a Photo (the entropy of its
Class probabilities), kept with the Photo's prediction.
_Avoid_: confidence (its opposite), error

## Comparing Surveys

**Reference Survey** and **compared Survey**:
The two Surveys of a comparison: differences are measured from the
reference's surface to the compared one's. (A reference Survey may also
anchor the Project frame when no GCP has coordinates.)

**Level of detection**:
The smallest difference between two Surveys that is told apart from their
Alignments' error; smaller differences are shown as no change and left out
of volumes.
_Avoid_: threshold, tolerance

**DEM**:
A north-up grid of the height of the topmost surface of a Reconstruction,
exported as a GeoTIFF; cells with nothing beneath them are left empty.
_Avoid_: DTM, DSM, height map

**Class map**:
A DEM's companion grid holding, per cell, the Class of the topmost surface.
_Avoid_: label raster

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

