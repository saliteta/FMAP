# Explicit Geometry Register (EGR)

Status: design, not implemented. Part of FG-SfM ([README](../README.md)); this doc is the plan for linking
reconstruction islands through a graph of distinctive 3D landmarks described by VGGT tokens.

## 1. Idea

A large aerial block is reconstructed as **islands**: connected groups of images that VGGT batches have
registered into one frame. Today islands are linked image to image, using GPS (BEV k-nearest neighbours) and an
exhaustive fallback. That works with RTK, struggles with consumer GPS and has nothing to go on without GPS.

The images are redundant; the **geometry they see is not**. A distinctive building, a toll plaza, a road junction
or a tunnel portal is one object however many images see it. EGR keeps an explicit **register** of such
objects: each entry has a 3D extent in its island's frame, a descriptor taken from VGGT's own tokens, and the
list of images that see it. Islands are linked by matching register entries, a search over a few hundred
objects instead of hundreds of images or hundreds of thousands of points.

## 2. Implicit vs explicit registers

VGGT-Ω ([arXiv 2605.15195](https://arxiv.org/abs/2605.15195)) adds learnable **register tokens** (scene
tokens) to every frame. In its register-attention layers only the registers attend across frames; the
following frame-wise blocks pass the aggregated scene information back into the image tokens. The registers are
an **implicit** shared memory: they work inside one forward pass, and cross attention decides what they hold.

EGR makes that memory **explicit** and lifts it above the batch:

| | VGGT-Ω registers (implicit) | Explicit geometry register (ours) |
|---|---|---|
| What an entry is | a learned token per frame, no fixed meaning | one physical object: building, road structure, landmark |
| Where it lives | inside one forward pass (one batch) | persistent, across all batches and islands |
| Geometry | none explicit (geometry is decoded from image tokens) | 3D extent in the island frame + observing images |
| How it is used | cross attention reads and writes it | explicit search and matching on the island / 3D graph level |
| Scale limit | frames per pass (≈ 500 on an 80 GB A100) | unbounded: only the register grows |
| Inspectable | no | yes: every match is an object you can look at |

VGGT-Ω shows that cross-frame scene information can be concentrated in a few tokens per frame, and its tokens
remain the best candidate descriptor (§5). What it does not show is whether those tokens describe *objects*,
stay consistent *across batches*, or can be used to *match* separately reconstructed parts. EGR depends on
exactly that, so it is the first thing to measure (§6).

## 3. The register

Each entry (one landmark object):

| Field | Content |
|---|---|
| `extent` | centroid and footprint / oriented box, in the island frame (scale from the island's Sim(3)) |
| `descriptor` | VGGT(-Ω) tokens pooled over the object's pixels (§5), one per observing batch, plus their mean |
| `views` | images that see it, with the object's pixel region in each |
| `uniqueness` | how far its descriptor is from every other entry (rarity) × how many batches see it (recurrence) |
| `island` | the island (frame) it belongs to |

Selection rule: keep **distinctive and recurring** objects. Rows of identical villas or uniform forest are
exactly what to leave out; they match everywhere and pin down nothing.

## 4. Two graphs and island linking

- **Image graph** (exists today, `SceneGraph`): images, batches, measured overlap, islands = frames.
- **Register graph** (new): nodes are register entries; each is linked to the images that see it. Objects are
  not linked to each other; they connect islands through their images.

Linking a new island:
1. Build its register entries from the batches that formed it.
2. Match them against the register (descriptor nearest neighbour + uniqueness gate).
3. Verify geometrically: ≥ 3 consistent matches → Sim(3) from the extents (coarse, metres). With 1–2 matches,
   use them only to pick partner images.
4. Refine as today: one VGGT batch of the island's images plus the images that see the matched objects,
   then the usual Sim(3) from shared cameras and the overlap gates.

Rule: **an island joins the graph only through shared register entries.** If it truly overlaps the main graph
but shares none, the register (object selection or descriptor) failed, not the island. That makes failures
diagnosable.

What it replaces: the BEV ring search and the exhaustive fallback in `link_bev`. GPS becomes an optional
extra gate. This is the missing piece for the **no-GPS** regime of the SZTU plan.

## 5. Descriptor and object discovery

Descriptor candidates, compared in §6:
- VGGT-1B aggregator patch tokens, per layer (frame-attention vs global-attention layers), pooled over the object mask;
- VGGT-Ω patch tokens after register attention, pooled the same way;
- VGGT-Ω registers of the frames that see the object (scene-level; likely too coarse alone).

Not DINO patch features alone: they are per-image and low level. The aim is a descriptor of the object as
seen jointly by several views.

Object discovery options, simplest first:
1. class-agnostic segmentation (e.g. SAM) per image, lifted to 3D with VGGT depth and merged across views;
2. unsupervised clustering of VGGT tokens + 3D positions into instances (VGGT only, no extra model);
3. open-vocabulary detection ("building", "road", "bridge") for semantic control over which classes count.

## 6. Stage EGR-0: feasibility (first experiment)

The crux: a token depends on which other frames share its batch. **Is the pooled token of the same object
consistent across different batches and viewpoints?**

Protocol on HAV (Bentley AT as ground truth for which images see which object):
1. Select ~20 distinctive objects (buildings, road structures) and ~100 distractors, segmented once.
2. Run VGGT-1B and VGGT-Ω on batches with different compositions: nadir only, oblique only, mixed,
   different strips, different batch sizes.
3. Pool each object's tokens per batch, per layer and per descriptor type (§5).
4. Retrieval: query an object's descriptor from batch A against all objects from batch B.

Metrics: top-1 / top-5 retrieval accuracy and mAP, split by nadir ↔ nadir, oblique ↔ oblique and
nadir ↔ oblique; margin between the true match and the best distractor.

Go / no-go: top-1 ≥ 0.8 across batches including nadir ↔ oblique for some layer/descriptor → build EGR-1.
Otherwise we know which representation fails, and whether VGGT tokens need help (e.g. a small learned head).

## 7. Stages

| Stage | Content | Done when |
|---|---|---|
| EGR-0 | token consistency study (§6) | go / no-go with the best layer and descriptor |
| EGR-1 | object discovery + register building per island | register of HAV islands, inspected in the viewer |
| EGR-2 | island linking via the register (§4), behind a `--link-mode register` switch | HAV links without GPS, same final accuracy |
| EGR-3 | SZTU (1500 images): RTK → consumer GPS → no GPS | no-GPS run registers all images, poses within the RTK run's accuracy |

## 8. Risks and open questions

- **Batch dependence of tokens** (§6). The main risk; everything else depends on it.
- **Repetitive scenes**: suburbs and farmland may have few distinctive objects. The uniqueness score must
  say so, and the fallback (image retrieval, sequence order) must stay.
- **Coarse geometry**: object extents give metre-level alignment only; precise registration stays with the
  VGGT batch and the global BA.
- **VGGT-Ω availability**: weights on Hugging Face are gated (access approval); check the license before
  commercial use. Memory: 13.4 GB at 100 frames and 43 GB at 500 frames on an A100, so a 24 GB GPU holds
  roughly 200 frames per pass.

## 9. Related work

| Work | Unit of linking | Descriptor | Alignment from the links? |
|---|---|---|---|
| VGGT-Ω (2026) | frames inside one pass | learned register tokens | implicit, within the batch only |
| G3AR (2026, multi-sequence aerial) | image chunks, maximum spanning tree | verified image proximity | yes, Sim(3) from shared images |
| VGGT-Long / VGGT-SLAM | images (loop closure) | SALAD or VGGT DINO tokens as a global image descriptor | yes, Sim(3) / homography |
| Semantic SLAM with VGGT (2511.16282) | objects | external detector (YOLO), 3D IoU | no, objects are tracked only |
| SG-Reg (2025) | objects in a scene graph | semantic label + shape + topology | yes |
| Maps from Motion (3DV 2025) | semantic objects | 2D object layout graph | yes, 2D maps |

As far as we found, nobody links islands during graph building through object landmarks described by VGGT
tokens. SG-Reg is the closest in spirit but uses semantic labels on pre-built maps.

References: [VGGT-Ω paper](https://arxiv.org/abs/2605.15195), [VGGT-Ω code](https://github.com/facebookresearch/vggt-omega),
[G3AR](https://arxiv.org/pdf/2609.16603), [VGGT-SLAM](https://arxiv.org/pdf/2505.12549),
[Semantic SLAM with VGGT](https://arxiv.org/html/2511.16282v1), [SG-Reg](https://arxiv.org/pdf/2504.14440),
[Maps from Motion](https://arxiv.org/html/2411.12620v2).
