---
title: "Camera Control and 4D Generation: From Pans to Consistent 3D Scenes"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "How a per-pixel camera ray map turns a video model into a controllable camera, and how that same model becomes a 3D/4D prior — the concrete bridge from generating clips to rendering navigable, dynamic scenes."
tags:
  [
    "video-generation",
    "diffusion-models",
    "camera-control",
    "4d-generation",
    "gaussian-splatting",
    "novel-view-synthesis",
    "world-models",
    "generative-ai",
    "deep-learning",
    "diffusers",
  ]
category: "machine-learning"
subcategory: "Video Generation"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/camera-control-and-4d-generation-1.png"
---

You asked your video model for a slow orbit around a parked motorcycle. The prompt said "slow orbit, cinematic, the camera circles the bike." What came back was a clip where the *bike* spun on a turntable while the camera sat still, the chrome smeared into a different shape on every revolution, and by the last frame the headlight had quietly migrated to the other side. The model gave you motion. It did not give you a *camera*. And it certainly did not give you a single, stable, three-dimensional motorcycle that you were merely looking at from different angles — it re-invented the bike on every frame, because nothing in its training ever forced one underlying 3D object to exist.

That gap is the subject of this post. A plain text-to-video model treats the camera the way it treats everything else: as something to vaguely gesture at in pixels. The word "orbit" is a weak suggestion competing with a thousand other tokens. But there is a different way to drive the camera — you hand the model the actual camera *geometry*: the position and orientation of the lens for every frame, encoded not as words but as a dense per-pixel signal the denoiser can read directly. Do that, and "orbit" stops being a suggestion and becomes a metric instruction the model tracks to within a degree or two. That is **explicit camera control**, and it is the first half of this post.

![Graph showing camera extrinsics and intrinsics turning into per-pixel rays, then a Plucker ray map added to the noisy latent before the spacetime DiT produces a pose-controlled clip](/imgs/blogs/camera-control-and-4d-generation-1.png)

The second half is where it gets interesting. Once you can specify the camera, you can ask the model to render the *same scene* from many viewpoints. And the moment you want those viewpoints to agree — for the motorcycle to be one consistent 3D object seen from the left and the right, not two different bikes — you have walked straight into the **3D consistency problem**, and from there into **4D generation**: dynamic, moving 3D scenes you can both navigate and watch evolve over time. By the end of this post you will be able to build a Plücker ray-map camera embedding from raw extrinsics and inject it into a video DiT in PyTorch; explain precisely why pure 2D video diffusion only *approximates* a single shared 3D scene; and sketch the score-distillation loop that turns a camera-controlled video model into a *prior* that optimizes a dynamic 3D Gaussian representation — the [DreamGaussian4D / SV4D](https://sv4d.github.io) line. This is the rung of the ladder where video generation stops being "make a clip" and starts being "build a renderable, navigable, dynamic scene," which is exactly the bridge to [video models as world models](/blog/machine-learning/video-generation/video-models-as-world-models).

It all sits on the spine of this series: **video is spatial generation times temporal coherence under a brutal compute budget.** Camera control is a new conditioning port on the same denoiser we have been building all along — a refinement of the [text, image, motion, and camera conditioning](/blog/machine-learning/video-generation/conditioning-video-text-image-motion-camera) menu, where we introduced camera pose as one knob among many. Here we open that knob all the way up, follow it into 3D, and find out where the geometry holds and where it quietly breaks. If you have not read [why video generation is hard](/blog/machine-learning/video-generation/why-video-generation-is-hard), the one-line frame is: a video model is a 2D pattern matcher we are pushing to behave like a 3D renderer, and this post is about exactly how far that push goes.

## 1. The core idea: a camera is a function from a scene to a frame

Start with the cleanest possible picture of what a camera *is*, because everything downstream depends on getting this right. A camera is a function that takes a 3D scene and a viewpoint and returns a 2D image. The viewpoint has two parts. The **extrinsics** say where the camera is and which way it points: a rotation $R \in \mathbb{R}^{3\times3}$ and a translation $t \in \mathbb{R}^3$, usually packed into a $4\times4$ matrix that maps world coordinates into the camera's own coordinate frame. The **intrinsics** say how that camera turns 3D directions into pixels: the focal lengths $f_x, f_y$ and the principal point $c_x, c_y$, packed into a $3\times3$ matrix $K$. Together, $(K, R, t)$ are the complete description of "where the lens is and how it sees."

Now here is the problem in one sentence. **A plain video model has no $(K, R, t)$ anywhere in it.** It was trained on clips with no camera annotations, so it learned the *statistics* of how cameras tend to move — dolly-ins are common, wild teleports are rare — but it never learned to obey a *specific* camera path, because it was never given one as input. When you write "orbit" in the prompt, you are asking a model that has only ever seen the *effects* of cameras to produce a *particular cause*, through the narrowest possible channel: a few text tokens fighting for attention against the subject, the lighting, the style, and everything else.

The fix is direct: stop describing the camera and start *handing the model its geometry*, frame by frame, as a signal dense enough to act on. That signal is a per-pixel encoding of the camera's rays. Instead of one $4\times4$ matrix per frame buried in a text prompt, you give the denoiser a tensor the same height and width as its latent, where every pixel carries the 3D ray that pixel looks along. The model no longer has to guess the camera; it can read it.

The reason this works at all is a property we lean on constantly in this series: the denoiser is a function over a spatial latent grid, and it accepts *additional channels* on that grid gracefully. We already exploited this for the start frame in image-to-video (a channel-concatenated latent) and for structure tracks in ControlNet-style conditioning (a depth or pose sequence). A camera ray map is the same trick, aimed at geometry: a few extra channels, perfectly aligned with the latent, telling each location which direction in space it is staring down. The denoiser learns to make the pixel it generates *consistent with* the ray it was told it represents.

Why is a *ray* the right thing to hand the model, as opposed to, say, the raw camera matrix? Because the ray is the quantity that actually determines what a pixel sees. The image-formation equation says a pixel $(u, v)$ shows whatever 3D point lies along its viewing ray, at the first opaque surface it hits. So the ray *is* the question the pixel is asking of the world: "what is along this line of sight?" Two consecutive frames of a moving camera differ precisely because their rays differ — the rays rotate and translate, and surfaces that were occluded come into view while others slide behind nearer ones. **Parallax — the thing that makes a camera move look three-dimensional rather than like a panning photograph — is exactly the differential of the ray field across frames.** A near surface, whose ray angle changes a lot per unit of camera translation, sweeps across the frame quickly; a far surface, whose ray angle barely changes, drifts slowly. When you hand the model the per-frame ray map, you are handing it the raw material of parallax directly. It does not have to infer "how fast should this surface move given the camera motion"; the rate is encoded in how the rays change, and the model has learned, from data, to render the surface displacement that those changing rays imply. That is the deep reason dense pose conditioning grips so much harder than a prompt word: the prompt says *that* the camera moves; the ray map says *how the world should reproject* as it does.

This also tells you what the model is *not* getting for free. The ray map specifies viewing directions, but it carries no depth — it does not say how far along each ray the first surface sits. Depth is what the model must still invent, and it is where the approximation lives: given the rays, the model produces *a* plausible depth-consistent rendering, but not necessarily *the* depth-consistent rendering that a true 3D scene would dictate. Hold that thought; it is the seed of every consistency limit in the second half of this post.

#### Worked example: how weak is "orbit" as a control?

Take our running clip — a 5-second, 720p, 24 fps shot, which after the VAE's $4\times8\times8$ compression is roughly a $13 \times 90 \times 160 \times 16$ latent. The text prompt enters through cross-attention as, say, 32 T5 tokens, of which maybe *two* ("slow orbit") carry the camera instruction. Those two tokens must influence about 3.0 million latent scalars, and they do it softly, by nudging attention weights. A Plücker ray map, by contrast, supplies $90 \times 160 \times 6 \approx 86{,}000$ numbers *per latent frame*, each one a hard geometric fact about that pixel's viewing direction, added directly to the signal the denoiser reads. The information content of the control jumps by four orders of magnitude, and — crucially — it is spatially *aligned* with what it constrains. Measured on camera-control benchmarks, this is the difference between a rotation error of roughly 10–12 degrees (text-only) and 1–2 degrees (Plücker pose conditioning), as reported by the [CameraCtrl](https://hehao13.github.io/projects-CameraCtrl/) line of work. Anchoring the geometry per pixel removes far more error than its token count would suggest, because camera pose is a *low-dimensional* cause with *high-dimensional* effects — and the ray map exposes exactly that low-dimensional cause.

## 2. From extrinsics to a Plücker ray map

Let us build the camera embedding from scratch, because the construction *is* the intuition. We want, for each pixel $(u, v)$ in each frame, the 3D ray that the camera shoots through that pixel: an origin and a direction. Then we encode that ray in a form that is convenient for a neural network.

![Stack of layers turning a 4x4 camera pose into a ray origin, a per-pixel unit direction, a moment vector, and finally a six-channel Plucker embedding injected into the DiT](/imgs/blogs/camera-control-and-4d-generation-2.png)

Start with a pixel $(u, v)$. Using the intrinsics, the direction it looks along *in camera coordinates* is $K^{-1} [u, v, 1]^\top$, normalized to unit length. Call that $d_{\text{cam}}$. To express it in world coordinates, rotate by the camera-to-world rotation: $d = R^\top d_{\text{cam}}$ (using the convention that $R$ maps world to camera, so its transpose maps back). The ray's origin is the camera center in world coordinates, $o = -R^\top t$ — the same for every pixel of a given frame. So each pixel now has a ray $(o, d)$.

You could feed $(o, d)$ directly — that is six numbers per pixel and it works. But the field has largely converged on **Plücker coordinates**, which encode a line in a way that is independent of where along the line you put the origin, and that turns out to train better. The Plücker representation of the ray is the pair $(d, m)$ where $m = o \times d$ is the *moment* (the cross product of the origin and direction). The direction $d$ tells you which way the line points; the moment $m$ tells you the line's position in space without privileging any particular point on it. Stack them and you get a **6-channel value per pixel**: $(d_x, d_y, d_z, m_x, m_y, m_z)$.

Why Plücker and not raw $(o, d)$? Two reasons that matter in practice. First, translation-equivariance: the moment $m = o \times d$ is invariant to sliding the origin $o$ along the ray (because $o + \lambda d$ gives $m = (o + \lambda d)\times d = o \times d$, since $d \times d = 0$). The network sees the *line*, not an arbitrary point on it, which is exactly the geometric quantity that should drive a pixel's appearance. Second, the values are well-scaled and smooth across the image plane, so they play nicely as an additive or concatenated channel without a learned normalization fighting the latent statistics. The result is a tensor of shape $T \times H \times W \times 6$ that lives on the same grid as the video and can be folded into the denoiser the way any other per-pixel conditioning is.

Here is the construction in PyTorch — readable, runnable, and the exact thing you would adapt to your own model. It takes per-frame extrinsics and intrinsics and returns the Plücker ray map.

```python
import torch

def plucker_ray_map(R, t, K, H, W):
    """Build a (T, 6, H, W) Plucker camera embedding.

    R: (T, 3, 3) world-to-camera rotations
    t: (T, 3)    world-to-camera translations
    K: (3, 3)    pinhole intrinsics (shared across frames here)
    Returns: (T, 6, H, W) tensor of (direction, moment) per pixel.
    """
    T = R.shape[0]
    device = R.device

    # Pixel grid -> homogeneous image coordinates (u, v, 1).
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    pix = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1)   # (H, W, 3)

    # Camera-space ray directions: K^{-1} [u, v, 1]^T, then normalize.
    Kinv = torch.inverse(K)                                    # (3, 3)
    d_cam = pix @ Kinv.T                                       # (H, W, 3)
    d_cam = d_cam / d_cam.norm(dim=-1, keepdim=True)

    # camera-to-world: R is world->camera, so R^T maps camera->world.
    Rt = R.transpose(1, 2)                                     # (T, 3, 3)
    # World-space directions per frame: (T, H, W, 3).
    d_world = torch.einsum("tij,hwj->thwi", Rt, d_cam)

    # Camera center in world coordinates: o = -R^T t, same for all pixels.
    o = -torch.einsum("tij,tj->ti", Rt, t)                    # (T, 3)
    o = o[:, None, None, :].expand(T, H, W, 3)               # broadcast

    # Plucker moment m = o x d.
    m = torch.cross(o, d_world, dim=-1)                       # (T, H, W, 3)

    plucker = torch.cat([d_world, m], dim=-1)                 # (T, H, W, 6)
    return plucker.permute(0, 3, 1, 2).contiguous()          # (T, 6, H, W)
```

A few things to note before we wire this into a model. The map is built at the *latent* resolution you intend to inject it at, not the pixel resolution — if your DiT operates on a $90 \times 160$ latent grid, you build the ray map at $90 \times 160$ with intrinsics rescaled to match. The intrinsics here are shared across frames (a fixed lens), which is the common case; per-frame zoom would make $K$ a $(T, 3, 3)$ tensor and changes nothing structural. And the construction is fully differentiable — which, foreshadowing the second half, is exactly what lets a 4D pipeline backpropagate appearance gradients through the camera.

## 3. Injecting pose into a video DiT

Having a $T \times 6 \times H \times W$ Plücker map is half the job; the other half is getting the denoiser to actually use it. There are two dominant injection strategies, and the choice mirrors the conditioning ports we catalogued earlier in the series.

The first is **channel concatenation into the latent**: project the 6-channel ray map up to the latent's channel count (or to a small added budget) with a lightweight conv, and add or concatenate it to the noisy latent before the first DiT block. This is what [CameraCtrl](https://hehao13.github.io/projects-CameraCtrl/) does — a dedicated *pose encoder* (a small ResNet-ish stack) ingests the Plücker map and produces a multi-scale feature that is added into the temporal attention layers of the backbone. The backbone itself stays frozen or lightly tuned; only the pose encoder learns. This plug-in property is the whole appeal: you do not retrain a 5B-parameter video model, you train a small adapter that teaches the existing model to read a camera channel.

The second is **conditioning the attention** more directly — adding the pose features as keys/values the video tokens attend to, or modulating the AdaLN scale-shift from a pooled pose embedding. This grips harder but couples more tightly to the architecture. In practice the concatenation/additive route dominates the open ecosystem because it is the least invasive.

Here is a faithful sketch of the additive-injection approach, the kind of `nn.Module` you would drop alongside a DiT's temporal attention. It takes the Plücker map, encodes it, and returns a per-token bias that is added to the hidden states.

```python
import torch
import torch.nn as nn

class CameraPoseEncoder(nn.Module):
    """Turn a (T, 6, H, W) Plucker map into an additive per-token feature
    aligned with a DiT's latent tokens (CameraCtrl-style plug-in)."""

    def __init__(self, hidden_dim, patch=(1, 2, 2)):
        super().__init__()
        pt, ph, pw = patch
        # Patchify the ray map the SAME way the DiT patchifies the latent,
        # so the pose tokens line up one-to-one with the video tokens.
        self.proj = nn.Conv3d(
            6, hidden_dim,
            kernel_size=(pt, ph, pw),
            stride=(pt, ph, pw),
        )
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Zero-init the last layer so the adapter starts as a no-op
        # and the pretrained backbone is undisturbed at step 0.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, plucker):
        # plucker: (B, 6, T, H, W) -> tokens (B, L, hidden)
        feat = self.proj(plucker)                  # (B, C, T', H', W')
        B, C, Tp, Hp, Wp = feat.shape
        feat = feat.flatten(2).transpose(1, 2)     # (B, L, C)
        return self.mlp(feat)                      # additive bias per token


# Inside the DiT forward, after patchifying the noisy latent to `x`:
#   pose_bias = camera_encoder(plucker_map)        # (B, L, hidden)
#   x = x + pose_bias                              # inject pose, then attn
```

Two design choices in that code are doing real work. First, **the pose map is patchified with the same kernel and stride as the latent**, so pose token $i$ corresponds to exactly the same spacetime region as video token $i$. Misalign these and the model gets a camera ray for the wrong patch — a subtle bug that shows up as the camera "lagging" the content. Second, **the final projection is zero-initialized.** This is the standard trick (from ControlNet and from the adapter literature) that lets you bolt a new conditioning branch onto a pretrained model without destroying it: at step 0 the bias is exactly zero, the backbone behaves identically to its uncontrolled self, and the camera signal *grows in* during fine-tuning rather than shocking the network. Skip the zero-init and your first thousand steps are spent recovering the quality you just smashed.

This is the entire reason camera control became a *plug-in* rather than a *new model*. You freeze the expensive backbone, train a few-million-parameter pose encoder on clips with known camera paths (synthetic data with ground-truth poses, or real footage run through structure-from-motion to recover $(R, t)$), and you get metric camera control on top of any compatible video DiT. The same pattern works for SVD, AnimateDiff, and the open DiT models — the pose encoder is small, the data is the bottleneck, and the backbone never moves.

From the *caller's* side this stays pleasantly simple: you build a trajectory, turn it into the pose map, and pass it alongside the usual generation arguments. Here is how that looks against a camera-controlled image-to-video pipeline, with a helper that synthesizes a clean orbit trajectory so you can see the whole path from "I want an orbit" to a rendered clip:

```python
import torch
from diffusers.utils import export_to_video, load_image

def orbit_trajectory(n_frames, radius=2.0, height=0.0, sweep_deg=45.0):
    """Generate (R, t) extrinsics for a smooth orbit around the origin."""
    Rs, ts = [], []
    for i in range(n_frames):
        ang = torch.deg2rad(torch.tensor(sweep_deg * i / (n_frames - 1)))
        # Camera position on a circle, looking at the origin.
        cam_pos = torch.tensor([radius * torch.sin(ang), height,
                                radius * torch.cos(ang)])
        fwd = -cam_pos / cam_pos.norm()                  # look-at origin
        up = torch.tensor([0.0, 1.0, 0.0])
        right = torch.cross(up, fwd); right /= right.norm()
        true_up = torch.cross(fwd, right)
        R = torch.stack([right, true_up, fwd])           # world->camera
        Rs.append(R)
        ts.append(-R @ cam_pos)                          # t = -R * center
    return torch.stack(Rs), torch.stack(ts)

# Build the trajectory and its Plucker map at LATENT resolution.
num_frames = 49
R, t = orbit_trajectory(num_frames, sweep_deg=45.0)
K = torch.tensor([[200., 0., 80.], [0., 200., 45.], [0., 0., 1.]])  # latent-scale
pose_map = plucker_ray_map(R, t, K, H=90, W=160)         # (T, 6, 90, 160)

pipe = load_camera_controlled_i2v("cogvideox-5b-camctrl")  # backbone + pose encoder
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

frames = pipe(
    image=load_image("motorcycle_hero_frame.png"),
    prompt="a parked motorcycle, cinematic, slow orbit to the right",
    camera_pose=pose_map,            # the only camera-control argument
    num_frames=num_frames,
    num_inference_steps=50,
    guidance_scale=6.0,
).frames[0]
export_to_video(frames, "orbit.mp4", fps=8)
```

The only camera-specific argument is `camera_pose=pose_map`; everything else is the ordinary I2V call we built in the conditioning post. Two practitioner notes hide in this snippet. First, the intrinsics `K` are at *latent* scale — focal lengths and principal point divided by the VAE's spatial downsample (here $8\times$), because the pose map must match the latent grid the encoder patchifies, not the output pixels. Get this wrong and the camera's field of view will be off, producing a too-wide or too-narrow orbit that *looks* like the model ignoring you. Second, the orbit's `sweep_deg` is the honesty dial: 45 degrees is comfortably inside what the model can fake from a single start frame; push toward 180 and you are back in the hallucination regime of §4, where you need multi-view machinery, not a single conditioned clip.

#### Worked example: the cost of the pose branch

A CameraCtrl-style pose encoder on a CogVideoX-class backbone adds on the order of 10–20M parameters — under 0.5% of a 5B model. Training it requires clips with known poses; the common recipe uses [RealEstate10K](https://google.github.io/realestate10k/), roughly 10 million frames from real-estate walkthroughs where camera trajectories were recovered by structure-from-motion, plus synthetic renders for clean ground truth. The marginal inference cost is one forward pass through the small encoder per step — negligible against the DiT itself — plus the memory for the $T \times 6 \times H \times W$ map, which at $13 \times 6 \times 90 \times 160$ in fp16 is about 4 MB, a rounding error next to the multi-gigabyte activation footprint of the backbone. The lesson: precise camera control is *cheap to add and cheap to run*; what is expensive is the labeled-trajectory data to train the adapter on. This is the opposite of the usual deep-learning cost profile, and it is why camera control shipped into production models fast while 4D generation did not.

### How the methods differ: encoding density versus injection point

The three families of camera control that shipped — MotionCtrl, CameraCtrl, and trajectory/drag control — are best understood along two axes: *how densely* the pose is encoded, and *where* it enters the network. Those two choices, not the brand name, determine how tightly the camera tracks and how cleanly the method composes with everything else.

![Matrix comparing MotionCtrl, CameraCtrl, and trajectory drag across pose encoding, injection point, camera tracking, and whether the base model is reused](/imgs/blogs/camera-control-and-4d-generation-4.png)

**Encoding density** is the first axis. MotionCtrl encodes the camera *coarsely*, as a per-frame $RT$ vector (the flattened rotation and translation) handed to a small camera-motion module. That is on the order of a dozen numbers per frame — enough to say "rotate this much, translate that way," but blind to *which pixel* looks *where*. CameraCtrl encodes the camera *densely*, as the full Plücker ray map — six numbers per pixel per frame — so the constraint is spatially resolved everywhere. Trajectory/drag methods (DragNUWA, Tora) sit in a third category: they encode *object*-level 2D paths rather than a whole-camera pose, which is a different control entirely (it moves things *in* the scene, not the lens). The denser the encoding, the tighter the grip — which is exactly why CameraCtrl's dense Plücker map reaches the ~1–2° rotation error that the coarser $RT$ encoding only approaches under moderate motion.

**Injection point** is the second axis. MotionCtrl injects its camera-motion features into a fine-tuned temporal module — the backbone moves a little. CameraCtrl injects through a *separate, frozen-backbone* pose encoder added to the temporal-attention layers — the backbone does not move at all, which is what makes it a true plug-in that transfers across models. Trajectory methods add a dedicated branch that is fine-tuned end-to-end. The trade is familiar from the broader conditioning story: a separate frozen-backbone adapter (CameraCtrl) is the cheapest to add and the most portable; a fine-tuned module (MotionCtrl, trajectory drag) can grip slightly harder on its specific backbone but does not transfer and risks disturbing the base model's quality.

The practical reading of the comparison: **use dense Plücker pose through a frozen-backbone adapter when you need precise, portable camera control**; reach for the coarser $RT$ encoding when you specifically want to *decouple* camera motion from object motion and can tolerate a looser grip; and reach for trajectory/drag when the thing you actually want to move is an *object in the scene*, not the camera. These are not competitors so much as different tools for different intents — and, because each enters through a different door, they can stack, exactly as the conditioning ports of the previous post compose.

## 4. Why this gives geometric control (and where it stops)

It is worth being precise about *why* a per-pixel ray map yields geometric control, because the same reasoning tells us exactly where the control runs out.

![Before-and-after comparison showing a text-only camera prompt that wanders with large rotation error versus a Plucker pose channel that tracks the requested orbit to within a couple of degrees](/imgs/blogs/camera-control-and-4d-generation-3.png)

The denoiser is trained, on every clip, to make the generated pixel at location $(u, v, \text{frame } f)$ *consistent with* the ray $(d, m)$ it was handed for that location. Over millions of examples, it learns the joint distribution of (ray map, video) — meaning it learns what the world looks like *from* a given set of viewpoints. At inference, you fix the ray map to your desired trajectory, and the model samples the video most consistent with being seen from those viewpoints. Because the ray map is dense and aligned, this constraint bites at every pixel of every frame, which is why the camera tracks tightly: the rotation between frame $f$ and frame $f+1$ is *encoded directly* in how the ray directions rotate, and the model has learned to translate that rotation into the correct parallax, occlusion, and perspective change.

That is genuine geometric control over the *camera*. But notice the precise claim: the model has learned to produce video *consistent with* a viewpoint sequence. It has **not** been forced to commit to a single, fixed 3D scene that those viewpoints all look at. There is no explicit 3D representation inside the denoiser — no point cloud, no mesh, no radiance field. There is a 2D network that has gotten very good at *faking* what a moving camera would see. Most of the time the fake is excellent, because the training data taught it real parallax. But the consistency is *emergent and approximate*, not *enforced and exact*. Push the camera far enough — a 90-degree orbit, a viewpoint the training distribution rarely contained, a scene with thin or repetitive structure — and the fake leaks: the back of the object, which the model never had to keep consistent with the front, comes out as a plausible hallucination rather than the true other side.

This is the central honest caveat of the whole field, and it is worth stating without hedging: **camera-controlled video diffusion gives you a controllable camera over an approximately-consistent scene, not a true renderer over an exact one.** The pose channel tells the model where to look; it does not give the model a 3D thing to look *at*. For a smooth pan or a modest orbit, the difference is invisible and the output is gorgeous. For a full turntable or a navigable scene, the difference is everything — and closing it is precisely what drives the move to multi-view and 4D, where we *add back* an explicit 3D representation and use the video model only as a teacher.

## 5. The multi-view consistency condition

To make the gap rigorous, consider the simplest version of the problem: generate two views of the same scene from two different camera poses, and ask them to *agree*. What does agreement even mean, mathematically?

![Graph showing one shared 3D scene producing two views linked by an epipolar or cross-view attention constraint, branching to a consistent pair with low reprojection error or to ghosting when no shared 3D exists](/imgs/blogs/camera-control-and-4d-generation-5.png)

Two views are geometrically consistent if and only if there exists a single 3D scene — a set of 3D points with colors — that reprojects to *both* images under their respective camera matrices. Formally, for a 3D point $X$ visible in both views with cameras $P_a = K[R_a | t_a]$ and $P_b = K[R_b | t_b]$, its image locations $x_a \sim P_a X$ and $x_b \sim P_b X$ are not free: they are tied by the **epipolar constraint** $x_b^\top F\, x_a = 0$, where $F$ is the fundamental matrix determined entirely by the two camera poses. The epipolar constraint says a point seen at $x_a$ in view A must lie *on a specific line* in view B. It is the algebraic shadow of "both images come from one 3D world."

A pair of images that satisfies the epipolar constraint everywhere — meaning every surface point in view A reprojects to the right place in view B — is multi-view consistent by construction; you could triangulate a single 3D scene from it. A pair that violates it is showing you two *different* worlds wearing the same prompt. And here is the problem with pure 2D video diffusion: **nothing in the per-frame denoising objective enforces the epipolar constraint.** The model denoises each latent toward the data distribution; it was never told that frame A and frame B must be reprojections of one scene. It approximates the constraint because consistent video is *more common in the training data* than inconsistent video, but approximation is not enforcement. The telltale failures — a texture that flickers between views, a feature that ghosts or doubles, geometry that subtly "breathes" as the camera moves — are all epipolar violations the model had no mechanism to prevent.

The methods that *do* push toward true multi-view consistency add a mechanism. [MVDream](https://mv-dream.github.io) and the multi-view video models couple the views inside the network with **cross-view attention**: when denoising view B, the tokens are allowed to attend to view A's tokens, so the network can copy and align shared structure rather than re-inventing it. [SV3D](https://sv3d.github.io) (Stable Video 3D) reframes the orbit *as a video* — the camera circling an object is just a clip with a known pose per frame — and leans on the video model's native temporal attention to keep the orbit coherent, then conditions on the camera so the views land at the right angles. These approaches get *much* closer to a single shared 3D scene, because cross-view attention is a soft, learned stand-in for the epipolar constraint. But "much closer" is still not "exact." The seams show up as minor reprojection error, and when you actually fuse the views into a 3D representation, those errors become floaters and blur. Which is the cue for the final move: stop hoping the 2D model is consistent, and instead *optimize an explicit 3D representation* that is consistent by construction, using the video model only to supply appearance.

#### Worked example: how small a reprojection error becomes a visible floater

It is easy to underestimate how little inconsistency it takes to ruin a fused 3D asset, so put numbers on it. Suppose you generate two views of the motorcycle 30 degrees apart, and the model places a chrome highlight on the tank that is consistent to within a 3-pixel reprojection error in a 1024-pixel-wide image — a sub-0.3% error, invisible if you flip between the two frames by eye. Now triangulate that point into 3D. The triangulation depth uncertainty scales roughly as $\delta z \approx z \cdot \delta_{\text{px}} / (f \cdot \sin\theta)$, where $z$ is the point's distance, $f$ the focal length in pixels, $\theta$ the angular baseline between views, and $\delta_{\text{px}}$ the reprojection error. For a modest 30-degree baseline, a 3-pixel error can translate into a depth error of a few percent of the object's size — which, splatted back, is a Gaussian floating a visible distance off the true surface. Multiply that across thousands of surface points, each with its own small independent error, and you get the characteristic "fuzz" of naively-fused multi-view diffusion: a haze of floaters around the true geometry. This is the quantitative reason the field stopped fusing views directly and moved to *optimization* (SDS into an explicit representation): the optimization averages thousands of noisy views into one geometry, suppressing the per-view error that direct triangulation amplifies. Small 2D inconsistencies are not small in 3D — and that single fact is what makes the explicit-representation detour worth its cost.

## 6. Video diffusion as a 3D/4D prior

Here is the conceptual pivot that makes 4D generation possible, and it is borrowed wholesale from the image world — specifically from the [score-prior / SDS idea](/blog/machine-learning/image-generation/diffusion-from-first-principles) that powered DreamFusion's text-to-3D. The idea is to flip the roles. Instead of asking the diffusion model to *output* a consistent multi-view video directly (which it can only approximate), you maintain an explicit, *inherently consistent* 3D representation, render it, and use the diffusion model as a *critic* that scores how realistic the renders are. You then optimize the 3D representation to make its renders score well. Consistency is free, because the representation is 3D by construction; realism is supplied by the prior.

![Stack of the 4D distillation loop: render a deformable-Gaussian scene at given poses and times, add noise, score with a camera-controlled video diffusion prior, take the SDS gradient, and update the Gaussians over thousands of steps](/imgs/blogs/camera-control-and-4d-generation-6.png)

The explicit representation, in the 2024–2025 generation of methods, is almost always **3D Gaussian Splatting** (3DGS) for static scenes and a **deformable / dynamic Gaussian** field for moving ones. A 3DGS scene is a cloud of anisotropic Gaussians — each with a position, a covariance (shape), an opacity, and a view-dependent color — that you can *splat* (rasterize) to any camera pose extremely fast and *differentiably*. The differentiability is the whole point: you can backpropagate an appearance loss from a rendered image all the way to the Gaussians' parameters. For 4D, you add a time axis: either the Gaussians' positions are functions of $t$ (a deformation field warps a canonical set of Gaussians over time), or you carry per-time attributes. Now rendering takes a camera pose *and* a time, and returns the frame a camera at that pose would see at that instant.

The scoring mechanism is **Score Distillation Sampling (SDS)**. We covered the score-based view of diffusion in the image series, so the one-line version: a diffusion model trained with the standard objective learns the score $\nabla_x \log p(x)$ of the data distribution at every noise level — it knows, for a noisy image, which direction makes it more realistic. SDS exploits this as a loss for *anything that renders to an image*. You take your rendered frame $x = g(\theta)$ (where $\theta$ are the Gaussian parameters), add noise to it at a random level $\sigma$, ask the diffusion model to denoise it, and use the *difference* between the model's prediction and the noise you added as a gradient on $\theta$. Concretely, the SDS gradient is

$$
\nabla_\theta \mathcal{L}_{\text{SDS}} = \mathbb{E}_{\sigma, \epsilon}\Big[ w(\sigma)\,\big(\hat{\epsilon}_\phi(x_\sigma; \sigma, c) - \epsilon\big)\, \frac{\partial x}{\partial \theta} \Big],
$$

where $\hat{\epsilon}_\phi$ is the diffusion model's noise prediction, $c$ is the conditioning (the text prompt and, critically here, the *camera pose*), $\epsilon$ is the noise that was added, and $w(\sigma)$ weights the noise levels. Read it as: "render the scene, ask the diffusion prior how to make this render more realistic, and nudge the 3D parameters in that direction." Repeat across random poses, random times, and random noise levels for a few thousand iterations, and the Gaussian field converges to a scene whose renders look right from *every* viewpoint and *every* time — consistent by construction, realistic by the prior.

The decisive upgrade for 4D over the original image-prior text-to-3D is **which model supplies the score.** DreamFusion used a 2D image diffusion model, which knew nothing about temporal coherence or multi-view consistency, so it suffered the infamous "Janus problem" (multi-faced objects) and could not do motion at all. The 4D line swaps in a *multi-view* or *camera-controlled video* diffusion prior — exactly the models we built in the first half of this post. Because that prior already understands both viewpoint changes and temporal evolution, the SDS gradients it produces pull the Gaussians toward something *jointly* consistent across space and time. This is the [DreamGaussian4D](https://jiawei-ren.github.io/projects/dreamgaussian4d/) and [SV4D](https://sv4d.github.io) recipe in one paragraph: a dynamic Gaussian representation, supervised by a camera-aware video diffusion model through SDS, optimized until its renders match what the video prior expects from every pose and moment.

Here is a conceptual training loop that makes the structure concrete. It is deliberately a *sketch* — the real implementations have many details around densification, view sampling, and reference-view anchoring — but it captures the load-bearing logic.

```python
import torch

def distill_4d(gaussians, video_prior, cameras, times, prompt,
               steps=8000, lr=1e-3):
    """Optimize a dynamic-3D Gaussian rep with a video diffusion prior.

    gaussians:   deformable 3D-Gaussian scene with .render(pose, t)
    video_prior: a camera-controlled video diffusion model (frozen)
    cameras:     sampler of (R, t, K) viewpoints to supervise from
    times:       sampler of timestamps in [0, 1]
    """
    opt = torch.optim.Adam(gaussians.parameters(), lr=lr)

    for step in range(steps):
        # 1. Sample a viewpoint and a short window of times.
        R, t, K = cameras.sample()
        t_window = times.sample_window()            # e.g. 8 consecutive times

        # 2. Render the dynamic scene -> a short clip from this camera.
        clip = torch.stack([
            gaussians.render(R, t, K, time=tt) for tt in t_window
        ])                                          # (F, 3, H, W), differentiable

        # 3. Score the render with the video prior (SDS).
        sigma = sample_noise_level()
        noise = torch.randn_like(clip)
        noisy = clip + sigma * noise
        with torch.no_grad():                       # prior is frozen
            eps_pred = video_prior(
                noisy, sigma,
                camera_pose=plucker_ray_map(R, t, K, H, W),  # pose conditioning
                prompt=prompt,
            )
        # SDS gradient: (prediction - true noise) flows into the render.
        grad = sds_weight(sigma) * (eps_pred - noise)

        # 4. Backprop the SDS gradient through the differentiable render.
        opt.zero_grad()
        clip.backward(gradient=grad)                # d(clip)/d(gaussians)
        opt.step()

        if step % 500 == 0:
            gaussians.densify_and_prune()           # 3DGS housekeeping
    return gaussians
```

The shape of the loop is the takeaway: **render → score with a camera-controlled video prior → push the gradient back into the 3D parameters → repeat.** The video model never produces the final asset directly; it is a frozen teacher whose only job is to say "more realistic this way" at every pose and time. The 3D representation, being explicit, carries the consistency the 2D model could only approximate. This is the cleanest answer to the multi-view problem we posed in §5: rather than coax a 2D model into being consistent, you make consistency structural and borrow the 2D model's realism.

### The deformation field: what makes a Gaussian scene "4D"

It is worth being concrete about the one piece that separates 3D from 4D, because it is where most of the difficulty lives. A static 3DGS scene is a fixed set of Gaussians $\{\mu_i, \Sigma_i, \alpha_i, c_i\}$ — mean position, covariance, opacity, color. To make it *dynamic*, you do not store a separate cloud per timestep (that would be enormous and would not enforce that the moving object stays the *same* object). Instead you keep one **canonical** Gaussian set and learn a **deformation field** $\Delta(\mu, t)$ — typically a small MLP — that, given a canonical Gaussian's position and a time $t$, predicts how that Gaussian moves, rotates, and scales at that instant:

$$
\mu_i(t) = \mu_i + \Delta_\mu(\mu_i, t), \qquad
\Sigma_i(t) = \Delta_R(\mu_i, t)\, \Sigma_i\, \Delta_R(\mu_i, t)^\top .
$$

The canonical set carries the object's *identity* (it is the same Gaussians at every time); the deformation field carries the *motion*. This factorization is exactly why 4D Gaussians keep an object coherent as it moves — the appearance is shared across time by construction, and only the motion is per-frame. It is the 3D-native version of the same "what is given versus what is invented" split we used for conditioning: the canonical Gaussians are *given* once, the deformation is *invented* per timestep under the video prior's supervision.

```python
import torch
import torch.nn as nn

class DeformableGaussians(nn.Module):
    """Canonical 3D Gaussians + a small deformation MLP over time."""

    def __init__(self, n_points, hidden=128):
        super().__init__()
        # Canonical (time-invariant) Gaussian parameters carry identity.
        self.mu = nn.Parameter(torch.randn(n_points, 3) * 0.1)   # positions
        self.scale = nn.Parameter(torch.zeros(n_points, 3))      # log-scale
        self.rot = nn.Parameter(torch.zeros(n_points, 4))        # quaternion
        self.opacity = nn.Parameter(torch.zeros(n_points, 1))
        self.color = nn.Parameter(torch.rand(n_points, 3))

        # Deformation field: (position, time) -> motion at that instant.
        self.deform = nn.Sequential(
            nn.Linear(3 + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 3 + 4),                # delta position + delta rot
        )
        nn.init.zeros_(self.deform[-1].weight)       # start as the static scene
        nn.init.zeros_(self.deform[-1].bias)

    def at_time(self, t):
        """Return the deformed Gaussians at time t in [0, 1]."""
        tcol = torch.full((self.mu.shape[0], 1), float(t), device=self.mu.device)
        delta = self.deform(torch.cat([self.mu, tcol], dim=-1))
        d_mu, d_rot = delta[:, :3], delta[:, 3:]
        return {
            "mu": self.mu + d_mu,                    # moved positions
            "rot": self.rot + d_rot,                 # rotated covariance
            "scale": self.scale, "opacity": self.opacity, "color": self.color,
        }

    def render(self, R, t_cam, K, time):
        gauss = self.at_time(time)
        return gaussian_splat(gauss, R, t_cam, K)    # differentiable rasterizer
```

Notice the zero-initialized deformation head again — the same trick as the pose encoder. At step 0 the scene is purely static (the deformation is a no-op), and motion *grows in* under the video prior's SDS gradients rather than fighting a half-formed geometry. The deformation field is also where the honest limits of 4D originate: a small MLP can represent smooth, moderate motion well, but it struggles to represent large, fast, or topology-changing motion (a splash, a tearing cloth), which is one reason today's 4D outputs shimmer on fine fast-moving detail. The representation is doing real work, but it is a *smooth* prior over motion, and reality is not always smooth.

## 7. Results: the capability ladder, honestly

Now the measured picture, because the spine of this series is that every claim earns a number. The capability ladder runs from *control a camera over a single clip* up to *generate a full dynamic 4D scene*, and at every rung two things degrade together: geometric consistency moves from explicitly enforced toward only approximately learned, and engineering maturity falls from "ships in production" toward "fragile research."

![Matrix mapping four capabilities to a representative method, its consistency level, and its maturity, showing consistency and maturity both degrading from camera control down to dynamic 4D](/imgs/blogs/camera-control-and-4d-generation-7.png)

The table below is the same picture in numbers you can act on. Treat the figures as representative orders of magnitude from the public literature, not lab-certified constants — I flag the soft ones.

| Capability | Representative method | What it produces | Consistency | Typical cost | Maturity |
| --- | --- | --- | --- | --- | --- |
| Camera-trajectory control | CameraCtrl / MotionCtrl | one pose-controlled clip | explicit pose, ~1–2° RotErr | + small adapter, ~same as base inference | production-ready |
| Multi-view (orbit) | SV3D / MVDream | N consistent views | near-consistent, minor seams | ~tens of seconds to a few min | research-mature |
| Static-3D lift | SV3D + SDS → 3DGS | a static 3D asset | good static, view-dependent | minutes per asset on an A100 | usable |
| Dynamic 4D | SV4D / DreamGaussian4D | a moving 3D scene | approximate, temporal flicker | tens of minutes to hours per scene | early research |

A few of these numbers deserve their footnotes. The **~1–2° rotation error** for camera control comes from the CameraCtrl evaluation protocol, which recovers the realized camera path from the generated video via structure-from-motion and compares it to the requested path; it is a real, repeatable metric, and the text-only baseline it improves on sits around 10–12°. The **minutes-per-asset** for static-3D lift is the SDS-distillation wall — you are running thousands of optimization steps, each a render plus a diffusion forward pass — and it is the single biggest reason 3D-from-image is not yet interactive. The **tens-of-minutes-to-hours** for dynamic 4D is the same wall plus a time axis: you are now distilling consistency across both viewpoints *and* frames, which multiplies the supervision the optimizer needs. And "approximate, temporal flicker" for 4D is the honest state of the art — the dynamic Gaussian fields produced today are impressive but show visible shimmer in fine detail and struggle with large, fast motion, because the video prior's own temporal coherence is itself only approximate and those errors compound through distillation.

#### Worked example: the cost of lifting one image to a 4D asset

Walk a single image — say, a frame of our motorcycle — up the ladder and watch the cost compound. **Step 1, multi-view:** run SV3D to generate 21 orbit views; on an A100 80GB this is roughly one video-diffusion forward sequence, on the order of tens of seconds. **Step 2, static lift:** distill those views into a 3D Gaussian scene via SDS — a few thousand optimization steps, each a splat render plus a diffusion score, landing around 5–10 minutes for a clean static asset. **Step 3, animate to 4D:** introduce the deformation field and distill against a *video* prior across poses and times — now each optimization step supervises a short clip, the step count climbs, and you are looking at tens of minutes to a couple of hours for a single dynamic scene, plus the peak VRAM to hold both the Gaussian field and the video-prior activations (which is why 4D work clusters on 80GB cards). The headline: **the jump from a static 3D asset to a dynamic 4D one is roughly an order of magnitude more compute**, and it buys you motion that is still visibly imperfect. That trade — large cost for approximate dynamics — is the honest frontier as of 2026, and it is why 4D generation lives in research pipelines and high-end content tools rather than in your real-time app.

## 8. Case studies: the real methods and what they actually deliver

Let us ground all of this in named systems, because the architecture-to-result mapping is where the field's real state lives. It helps to see them in order, because the field walked the exact ladder this post describes: coarse camera control first, then dense pose, then the image-to-3D lift, then dynamic 4D — each step reusing the previous one's video prior as its engine.

![Timeline tracing camera control from MotionCtrl's RT embedding through CameraCtrl's Plucker pose to SV3D, DreamGaussian4D, and SV4D's dynamic 4D generation](/imgs/blogs/camera-control-and-4d-generation-8.png)

The progression on the timeline is not just chronological — it is *causal*. MotionCtrl (2023) established that a video model could be steered by an explicit camera signal at all. CameraCtrl (2024) made that signal dense (Plücker) and the adapter portable, which is what let the same idea spread across backbones. SV3D (2024) realized that a camera-controlled video model is already a multi-view generator and used it to produce clean orbits — the input that makes static-3D distillation converge. DreamGaussian4D and SV4D (2024) then plugged those camera-aware video priors into an explicit dynamic-Gaussian representation through SDS, crossing from "looks 3D" to "is a navigable 3D/4D asset." SV4D 2.0 (2025) tightened the dynamics. Read top to bottom, the timeline is the argument of this entire post: give a video model the camera's geometry, and you can climb all the way from a pan to a renderable, moving scene.

**CameraCtrl (Plücker pose as a plug-in).** The CameraCtrl line established the recipe most camera control now follows: encode the camera trajectory as a per-frame Plücker ray map, feed it through a dedicated pose encoder, and inject the resulting features into a frozen (or lightly tuned) video backbone's temporal layers. Trained largely on RealEstate10K with SfM-recovered poses, it delivers the ~1–2° rotation tracking quoted above and, critically, *transfers* — the same pose-encoder idea has been adapted onto SVD, AnimateDiff, and open DiT backbones. The lesson it cemented: camera control is a small adapter and a data problem, not a new foundation model. This is the technique you reach for when you have a base video model you like and you need a controllable camera on top of it.

**MotionCtrl (decoupled camera and object motion).** MotionCtrl is worth contrasting because it encodes the camera more coarsely — as a sequence of $RT$ (rotation-translation) vectors fed to a camera-motion module — and separately handles *object* trajectories via a 2D drag signal. The coarser camera encoding tracks well for moderate motion but is looser at extremes than a dense Plücker map; the payoff is that it cleanly *separates* "move the camera" from "move this object," which is exactly the decomposition a director wants. The contrast with CameraCtrl is the recurring trade of this post: a denser pose encoding (Plücker) grips harder; a coarser one (RT vectors) is simpler and composes more easily with object-level control.

**SV3D and the orbit-as-video trick.** Stable Video 3D's key insight is that *a camera orbiting a static object is just a video with a known pose per frame.* So you do not need a bespoke multi-view architecture — you take a strong video diffusion model (SVD), condition it on the orbit's per-frame poses, and let its native temporal attention enforce coherence around the loop. The output is a set of consistent novel views good enough to distill into a static 3D asset. SV3D is the cleanest demonstration that **a video model is already a latent multi-view model** — the temporal axis and the viewpoint axis are, for a static scene, the same axis. This is the conceptual hinge of the whole post: control the camera, and a video model *becomes* a 3D-view generator.

**DreamGaussian4D and SV4D (the 4D frontier).** These are the dynamic-scene methods. DreamGaussian4D lifts a single video (or image) to a deformable 3D Gaussian field by distilling against image and video diffusion priors, optimizing a canonical Gaussian set plus a time-varying deformation. SV4D (and the 2025 SV4D 2.0) trains a *multi-view video* diffusion model directly — generating multiple consistent viewpoints *and* multiple timesteps in one model — and uses it as the prior to optimize a 4D representation, which tightens the joint space-time consistency relative to stitching separate 3D and video priors. They represent the genuine state of the art in image/video-to-4D, and they are also where the honesty caveats bite hardest: outputs are compelling for moderate motion and modest orbits, and visibly degrade — flicker, blur, geometry breathing — under fast motion, large viewpoint swings, or fine repetitive texture. As of 2026 this is a research-grade capability producing real but imperfect assets at minutes-to-hours per scene.

The thread connecting all four: **every step up the ladder reuses the video model as the engine, and adds an explicit-geometry mechanism to convert the video model's approximate consistency into something more structural** — a pose channel for control, cross-view attention for multi-view, an explicit Gaussian field for 3D/4D. The video model never stops being central; it just gets wrapped in more geometry as the ambition grows.

## 9. The stress tests: where the geometry breaks

A technique is only understood once you know how it fails. Here are the failure modes I would probe before betting a pipeline on any of this, posed as the engineering questions they really are.

**What happens past a modest orbit?** Camera control is excellent for pans, dollies, and orbits up to perhaps 30–60 degrees, because the training data is dense there and the model has real parallax to imitate. Push to a full 180–360° turntable and you ask the model to render the back of an object it only ever saw the front of within a single clip. With *pure* camera-controlled video diffusion (no explicit 3D), the far side is a plausible hallucination — often subtly wrong, sometimes a different object entirely. This is precisely why full turntables go through SV3D-style multi-view-then-distill rather than a single conditioned clip: you need the cross-view mechanism (or an explicit 3D rep) to *force* the back to be consistent with the front.

**What happens when motion is large between frames?** Both camera control and 4D distillation lean on the video model's temporal coherence, which is itself strongest for small inter-frame motion. Large motion — a fast pan combined with a fast-moving subject — stresses the prior's coherence, and in the 4D case those coherence errors compound through thousands of SDS steps into visible flicker and ghosting in the Gaussian field. The mitigation is the same one the long-video work uses: smaller motion per step, more frames, and reference-view anchoring so drift has something to snap back to. We treat the temporal side of this directly in [long video and autoregressive rollout](/blog/machine-learning/video-generation/long-video-and-autoregressive-rollout); the 4D case inherits its error-accumulation behavior wholesale.

**What happens when the scene has thin or repetitive structure?** Epipolar consistency is hard to nail on thin structures (a bike's spokes, a chain-link fence) and repetitive texture (brick, foliage), because the model can satisfy the appearance objective with a *different* valid-looking structure in each view — the constraint is locally ambiguous. These are the surfaces where multi-view methods seam and 4D fields shimmer. If your asset is mostly thin or repetitive geometry, set expectations accordingly; the distillation will fight you.

**What happens when the distillation cost is the wall?** For 4D, the dominant cost is not the video model's single forward pass — it is the *thousands* of them inside the SDS loop, each paired with a differentiable render. When a single asset takes an hour, the bottleneck is iteration count, and the research frontier is squarely aimed at it: better initializations (start from an SV3D static lift rather than noise), few-step distilled priors (so each SDS score is cheaper), and direct 4D-generation models that skip per-scene optimization. Until those mature, budget 4D as an offline, batch capability, not an interactive one. The serving-side reality of this — VRAM walls, cost-per-asset — connects to the broader [efficient inference](/blog/machine-learning/edge-ai) story, but the short version is: 4D today is a render-farm job, not a request handler.

**What happens when the SDS prior and the reference disagree?** A failure mode specific to distillation. When you lift a *specific* input image to 4D, you want the result to match that exact image at the reference view *and* look plausible everywhere else. But the SDS gradient pulls toward the diffusion prior's idea of "plausible," which is a distribution, not your specific image — so without a counterweight, the optimization can drift away from your reference, subtly changing the very thing you anchored on. The fix every serious pipeline uses is a **reference-view reconstruction loss**: at the input's known pose and time, add a direct pixel (and perceptual) loss between the render and the actual input image, and weight it heavily. The SDS prior then governs the *unseen* views and times while the reconstruction loss nails the *seen* one. Get the balance wrong in either direction and you see it: too little reference weight and the asset stops resembling your input; too much and the unseen views go blurry and mean-reverted because the prior could not do its job. Tuning that one weight is a surprising fraction of the practical work in getting a clean 4D asset.

**What happens when you actually need metric geometry?** A subtle one. Even when the renders look consistent, the recovered 3D is not *metrically* faithful — scale is ambiguous from monocular priors, and the geometry can be locally plausible but globally warped. If your downstream task needs accurate 3D (measurement, robotics, simulation that respects real distances), a generated 4D asset is a *visual* approximation, not a surveyed one. This is the line between "looks 3D" and "is 3D," and it is the deepest version of this post's caveat — the same line that separates a video model from a true [world model](/blog/machine-learning/video-generation/video-models-as-world-models).

## 10. When to reach for this (and when not to)

A decisive guide, because the cost-and-maturity spread across these techniques is enormous and the wrong choice wastes hours.

**Reach for camera-trajectory control** whenever you need a *specified* camera move — an orbit, a dolly, a precise reveal — and you have or can recover the poses. It is cheap (a small adapter), fast (near base inference cost), and mature. If your only goal is "the camera should circle the subject," a Plücker-pose-conditioned clip is the right tool and the rest of this post is overkill. Use CameraCtrl-style dense pose for tight tracking; use MotionCtrl-style coarse $RT$ when you also need to decouple object motion and can tolerate looser camera grip.

**Do not reach for full camera control when text suffices.** For a gentle, unspecified drift where "slow push-in" in the prompt gets you 90% of the way, the pose adapter is unnecessary machinery. Camera conditioning earns its keep when the path must be *precise* — repeatable across runs, matched to a plate, or tracing an exact geometric shape. Loose cinematic motion does not need it.

**Reach for multi-view (SV3D-style)** when you need several *consistent* views of one object — for a product turntable, or as the front end of a static-3D lift. This is research-mature and produces genuinely usable orbits. It is the right step *before* any 3D distillation, because clean multi-view inputs are what make the distillation converge.

**Reach for static-3D lift** when you need an actual navigable 3D asset (a 3DGS scene you can render from arbitrary new viewpoints in real time after the fact) and you can afford minutes of offline optimization per asset. The payoff is a real, reusable 3D thing; the cost is the SDS wall. Worth it for content pipelines, asset creation, and anything where you render the result many times.

**Reach for dynamic 4D only when you genuinely need motion in 3D** — a moving character or scene you can both navigate and watch evolve — and you accept research-grade quality at tens-of-minutes-to-hours per scene. As of 2026, 4D is the bleeding edge: spectacular demos, real flicker, real cost. If you can get away with a 2D camera-controlled clip (you are only ever going to view it from the path you generated), do that instead — you do not need a 4D asset to *watch* a video, only to *navigate* one.

**Do not confuse "looks 3D" with "is 3D."** If the downstream consumer is a human watching a clip, a camera-controlled 2D video is almost always the better trade — cheaper, sharper, mature. The explicit-3D ladder earns its cost only when you need to *re-render* from new, unplanned viewpoints or *interact* with the scene. That distinction — watch versus navigate — is the cleanest decision rule in this whole space.

To make that rule operational, here is the decision I actually walk through on a real task. **Question one: will any viewpoint that was not generated up front ever be needed?** If no — the camera path is known in advance and fixed — stop at camera-controlled 2D video. You never need a 3D representation to play back a path you already chose; you only need one to answer viewpoints you have not chosen yet. **Question two: does the content move?** If the scene is static and you do need free viewpoints (a product you will spin interactively, an environment you will fly around), lift to a static 3DGS asset — it is mature enough and the offline minutes pay for unlimited later renders. **Question three: does it move *and* need free viewpoints?** Only then do you pay for 4D, and only then do the tens-of-minutes-to-hours and the visible flicker become a cost worth accepting. The number of real tasks that genuinely require the top rung is smaller than the demos suggest — most "3D video" needs are satisfied two rungs down by a well-controlled camera over a planned path. Reserve 4D for the cases that are unambiguously *interactive and dynamic*, and you will spend your compute where it actually buys you something.

One more practitioner caution worth stating plainly: **do not start a 4D distillation from noise if you can start from a static lift.** The single biggest determinant of whether a 4D optimization converges cleanly is its initialization. Generate clean multi-view orbits with SV3D, distill a *static* 3DGS asset first, then introduce the deformation field and animate against the video prior. Starting the deformable optimization from an already-coherent static geometry turns a brittle, hours-long fight into a much shorter, more reliable refinement — because the optimizer is no longer discovering geometry and motion simultaneously, it is adding motion to geometry it already trusts. Skipping the static stage to "save a step" is the most common way a 4D run ends in a flickering mush.

## 11. Why this matters: the bridge to world models

Step back from the mechanics and the reason this rung of the ladder matters comes into focus. A plain video model generates a clip you watch passively. A *camera-controlled* video model generates a clip you can *direct* — you decide where the lens goes. A *3D/4D-consistent* model generates a scene you can *navigate* — you can choose a viewpoint that was never explicitly generated and get a coherent answer. Each step adds agency over the output, and the last step crosses a meaningful line: a navigable, dynamic scene is no longer a recording, it is something closer to a *simulation* — a place you can move through and that changes over time independent of any single fixed camera path.

That is exactly the connection to world models. A world model is a generative model you can *act in*: you supply a viewpoint or an action, and it produces the next state of the world consistent with everything before. Camera control is the most basic action — "move the lens here" — and the consistency machinery of multi-view and 4D is precisely what makes the world *stay the same world* as you move through it. The line we will pick up in [video models as world models](/blog/machine-learning/video-generation/video-models-as-world-models) is the natural continuation: replace the camera-pose action with a richer action space (move, interact, change), keep the consistency requirement, and you have the beginnings of a learned simulator. Camera control and 4D generation are the concrete, shipped-today first steps of that program — the point where a video model stops being a video model and starts becoming a renderer of worlds.

It also matters mundanely and immediately: controllable cinematography (orbits, reveals, matched moves) is a real production need that camera conditioning solves *now*, and 3D/4D asset creation from a single image or video is a content-pipeline capability that, even at research-grade quality, is already faster than modeling a dynamic scene by hand. The frontier is glamorous; the present-day utility is real. Both flow from the same idea this post is built on — give the model the camera's geometry, and a 2D generator starts behaving like a 3D one.

The practical end-to-end choices — which camera-control method to bolt onto which backbone, when to lift to 3D, how to budget a 4D job — come together in the series [capstone playbook](/blog/machine-learning/video-generation/building-with-video-generation-the-playbook). Everything here is one branch of that larger decision tree.

## 12. Key takeaways

- **A camera is $(K, R, t)$, and a plain video model has none of it.** Text words like "orbit" are a four-orders-of-magnitude weaker control than a per-pixel ray map, which is why text-only camera tracking sits near 10–12° error and Plücker-pose conditioning reaches 1–2°.
- **Encode pose as a Plücker ray map and inject it per token.** Build a $T \times 6 \times H \times W$ tensor of $(\text{direction}, \text{moment})$, patchify it the same way the latent is patchified, and add it through a zero-initialized adapter so the frozen backbone is undisturbed at step 0.
- **Camera control is a cheap adapter and a data problem.** A ~10–20M-parameter pose encoder over a frozen backbone gives metric control; the real cost is labeled-trajectory data (RealEstate10K + synthetic), not compute or parameters.
- **Pure 2D video diffusion only *approximates* a single shared 3D scene.** Nothing in the per-frame objective enforces the epipolar (multi-view) constraint; cross-view attention is a learned, soft stand-in that gets close but seams at extremes.
- **For true consistency, make the 3D explicit and use the video model as a prior.** SDS-distill a 3D-Gaussian (static) or deformable-Gaussian (4D) field against a camera-controlled video diffusion model: render → score → push the gradient into the geometry → repeat.
- **A video model is already a latent multi-view model.** For a static object, the temporal axis and the viewpoint axis are the same axis — SV3D's orbit-as-video trick is the cleanest proof.
- **Consistency and maturity degrade together up the ladder.** Camera control ships in production; multi-view is research-mature; static-3D lift is usable at minutes/asset; dynamic 4D is early research at tens-of-minutes-to-hours/scene with visible flicker.
- **"Looks 3D" is not "is 3D."** Generated geometry is visually plausible but not metrically faithful; reach for explicit 3D/4D only when you must *navigate* or *re-render* the scene, not merely *watch* it.
- **This is the bridge to world models.** Camera control is the simplest action; multi-view/4D consistency is what keeps the world one world as you move — the concrete first steps toward a navigable, dynamic, learned simulator.

## 13. Further reading

- **He et al., "CameraCtrl: Enabling Camera Control for Text-to-Video Generation" (2024)** — the Plücker-pose-encoder plug-in recipe and the camera-tracking evaluation protocol used throughout this post.
- **Wang et al., "MotionCtrl: A Unified and Flexible Motion Controller for Video Generation" (2024)** — coarse $RT$ camera encoding plus decoupled object-trajectory drag control.
- **Voleti et al., "SV3D: Novel Multi-view Synthesis and 3D Generation from a Single Image using Latent Video Diffusion" (2024)** — the orbit-as-video insight and the multi-view-to-static-3D pipeline.
- **Shi et al., "MVDream: Multi-view Diffusion for 3D Generation" (2023)** — cross-view attention as a soft multi-view-consistency mechanism.
- **Ren et al., "DreamGaussian4D: Generative 4D Gaussian Splatting" (2024)** and **Xie et al., "SV4D: Dynamic 3D Content Generation with Multi-Frame and Multi-View Consistency" (2024–2025)** — the image/video-to-4D distillation frontier.
- **Poole et al., "DreamFusion: Text-to-3D using 2D Diffusion" (2022)** — the original Score Distillation Sampling formulation this whole 3D/4D line generalizes.
- **Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (2023)** — the explicit, differentiable representation that the distillation loops optimize.
- Within this series: [why video generation is hard](/blog/machine-learning/video-generation/why-video-generation-is-hard), [conditioning video: text, image, motion, camera](/blog/machine-learning/video-generation/conditioning-video-text-image-motion-camera), [video diffusion transformers](/blog/machine-learning/video-generation/video-diffusion-transformers), [long video and autoregressive rollout](/blog/machine-learning/video-generation/long-video-and-autoregressive-rollout), [video models as world models](/blog/machine-learning/video-generation/video-models-as-world-models), and the [capstone playbook](/blog/machine-learning/video-generation/building-with-video-generation-the-playbook); and out to the image series for the [score-prior / SDS foundation](/blog/machine-learning/image-generation/diffusion-from-first-principles).
