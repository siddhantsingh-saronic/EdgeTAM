import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2


# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. EdgeTAM is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

from sam2.build_sam import build_sam2_video_predictor

checkpoint = "../checkpoints/edgetam.pt"
model_cfg = "edgetam.yaml"

predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def create_mask_video(video_segments, frame_names, video_dir, output_path, fps=30):
    """
    Create a video combining original frames with segmentation masks overlay.
    
    Args:
        video_segments: Dictionary containing per-frame segmentation results
        frame_names: List of frame filenames
        video_dir: Directory containing the original frames
        output_path: Path for the output video file
        fps: Frames per second for the output video
    """
    # Get the first frame to determine video dimensions
    first_frame = cv2.imread(os.path.join(video_dir, frame_names[0]))
    height, width, _ = first_frame.shape
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process each frame
    for frame_idx in range(len(frame_names)):
        # Load original frame
        frame_path = os.path.join(video_dir, frame_names[frame_idx])
        frame = cv2.imread(frame_path)
        
        # If we have segmentation results for this frame, overlay them
        if frame_idx in video_segments:
            # Create overlay for all objects in this frame
            overlay = np.zeros_like(frame, dtype=np.float32)
            
            for obj_id, mask in video_segments[frame_idx].items():
                # Convert mask to 3-channel and apply color
                mask_3d = np.stack([mask.squeeze()] * 3, axis=-1)
                
                # Use different colors for different objects
                cmap = plt.get_cmap("tab10")
                color = np.array(cmap(obj_id)[:3]) * 255
                
                # Apply colored mask
                colored_mask = mask_3d * color.reshape(1, 1, 3)
                overlay += colored_mask
            
            # Blend original frame with overlay
            alpha = 0.4  # Transparency of the mask overlay
            frame = frame.astype(np.float32)
            blended = cv2.addWeighted(frame, 1-alpha, overlay.astype(np.float32), alpha, 0)
            frame = blended.astype(np.uint8)
        
        # Write frame to video
        out.write(frame)
    
    # Release video writer
    out.release()
    print(f"Video saved to: {output_path}")

# ffmpeg -i /Users/siddsingh/Documents/horizon_clips/second_half/second_half.mp4 -q:v 2 -start_number 0 /Users/siddsingh/Documents/horizon_clips/second_half/frames/'%05d.jpg'

# `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
video_dir = "/Users/siddsingh/Documents/horizon_clips/second_half/first_500_frames/"

# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# take a look the first video frame
frame_idx = 0
plt.figure(figsize=(9, 6))
plt.title(f"frame {frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[frame_idx])))

inference_state = predictor.init_state(video_path=video_dir)

# predictor.reset_state(inference_state)

ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 4  # give a unique id to each object we interact with (it can be any integers)

point = [902.3181818181819, 674.5000000000001, 1188.681818181818, 806.3181818181819]

# Let's add a box at (x_min, y_min, x_max, y_max) = (300, 0, 500, 400) to get started
box = np.array(point, dtype=np.float32)
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    box=box,
)

plt.figure(figsize=(9, 6))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_box(box, plt.gca())
show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

# run propagation throughout the video and collect the results in a dict
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# render the segmentation results every few frames
vis_frame_stride = 30
plt.close("all")
for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
    plt.figure(figsize=(6, 4))
    plt.title(f"frame {out_frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)

# Create video with mask overlay
output_video_path = "segmented_video.mp4"
create_mask_video(video_segments, frame_names, video_dir, output_video_path, fps=30)


