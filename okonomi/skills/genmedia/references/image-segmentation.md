# Image Segmentation for Animation

Extract individual regions from generated images to create animated compositions. Use segmentation when you want to animate parts of a static image independently while preserving the original composition.

## When to Use

- Animate individual elements of a generated image (e.g., mirror facets in a disco ball)
- Create composition-preserving animations from static images
- Extract object masks for selective editing or effects
- Identify and isolate regions for independent animation control

## Segmentation Methods

### 1. SAM2 Auto-Segment (fal.ai)

**Best for:** Quick automatic segmentation of distinct objects

```python
result = genmedia.run(
    "fal-ai/sam2/auto-segment",
    image_url="<your-image-url>",
    return_mask_image=True,
    mask_image_format="png"
)

# Returns individual_masks array with separate PNG masks
mask_urls = [mask["url"] for mask in result["individual_masks"]]
```

**Pros:**
- Fully automatic — no prompts or coordinates needed
- Returns clean binary masks
- Works well for distinct objects with clear boundaries

**Cons:**
- May over-segment or under-segment complex scenes
- Limited control over segmentation granularity
- Typically finds 20-50 segments

### 2. Felzenszwalb Graph-Based (scikit-image)

**Best for:** Fine-grained segmentation based on color and spatial similarity

```python
from skimage import segmentation, color
from skimage.measure import regionprops
from PIL import Image
import numpy as np

img = Image.open("image.png")
img_array = np.array(img)

# Segment using Felzenszwalb algorithm
segments = segmentation.felzenszwalb(
    img_array,
    scale=120,        # Higher = fewer, larger segments
    sigma=0.8,        # Gaussian smoothing
    min_size=400      # Minimum segment size in pixels
)

# Extract region properties
props = regionprops(segments)

facets = []
for prop in props:
    if 400 < prop.area < 25000:  # Filter by size
        minr, minc, maxr, maxc = prop.bbox
        cy, cx = prop.centroid
        
        facets.append({
            "center": [int(cx), int(cy)],
            "bbox": [int(minc), int(minr), int(maxc - minc), int(maxr - minr)],
            "area": float(prop.area)
        })
```

**Pros:**
- Excellent for fine detail (can find 300+ segments)
- Tunable parameters for granularity control
- Respects color gradients and texture boundaries
- Works offline (no API calls)

**Cons:**
- Requires scikit-image installation
- More code than SAM2
- Need to tune parameters per image

**Parameter tuning:**
- `scale`: 80-150 typical (lower = more segments)
- `sigma`: 0.5-1.0 (blur before segmentation)
- `min_size`: 200-1000 (filter tiny regions)

### 3. SLIC Superpixels (scikit-image)

**Best for:** Regular grid-like segmentation

```python
from skimage import segmentation

segments = segmentation.slic(
    img_array,
    n_segments=150,   # Target number
    compactness=10,   # Higher = more square-shaped
    start_label=1
)
```

**Pros:**
- Predictable, uniform segments
- Good for regular patterns
- Fast

**Cons:**
- Ignores object boundaries
- Less accurate for natural shapes

## Processing Masks to Coordinates

Once you have segmentation masks, extract bounding boxes and centers:

```python
from PIL import Image
import numpy as np

# For each mask
mask_img = Image.open("mask.png").convert('L')
mask_array = np.array(mask_img)

# Find non-zero pixels
ys, xs = np.where(mask_array > 128)

if len(xs) > 0:
    # Bounding box
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    
    # Center
    x_center = int(xs.mean())
    y_center = int(ys.mean())
    
    # Area
    area = float(len(xs))
    
    facet = {
        "center": [x_center, y_center],
        "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
        "area": area
    }
```

## Animation with p5.js

Use extracted coordinates to animate image regions:

```javascript
const FACETS = /* JSON array of facet coordinates */;

let img;
let facetGraphics = [];

function preload() {
  img = loadImage('original-image.png');
}

function setup() {
  createCanvas(1376, 768);
  
  // Create graphics buffer for each facet
  for (let facet of FACETS) {
    let bbox = facet.bbox;
    let g = createGraphics(bbox[2], bbox[3]);
    
    // Copy facet region from original
    g.copy(img, bbox[0], bbox[1], bbox[2], bbox[3],
           0, 0, bbox[2], bbox[3]);
    
    facetGraphics.push({
      graphics: g,
      center: facet.center,
      phase: random(TWO_PI),
      speed: random(1.5, 3.0)
    });
  }
}

function draw() {
  background(0);
  let time = millis() * 0.001;
  
  for (let fg of facetGraphics) {
    push();
    translate(fg.center[0], fg.center[1]);
    
    // Subtle scale pulsing
    let scale = 1.0 + sin(time * fg.speed + fg.phase) * 0.06;
    scale(scale);
    
    // Brightness shimmer
    let brightness = 0.75 + sin(time * 2.0 + fg.phase) * 0.25;
    tint(255, brightness * 255);
    
    imageMode(CENTER);
    image(fg.graphics, 0, 0);
    pop();
  }
}
```

## Workflow Summary

1. **Generate image** using genmedia (e.g., photorealistic disco ball interior)
2. **Choose segmentation method:**
   - SAM2 for quick automatic segmentation (20-50 segments)
   - Felzenszwalb for fine detail (300+ segments) — recommended for complex scenes
   - SLIC for uniform grid patterns
3. **Extract coordinates** from masks or labeled regions
4. **Create p5.js sketch** that copies each region into separate graphics buffers
5. **Animate independently** with subtle transforms (scale, brightness, rotation)
6. **Preserve composition** — keep facets in original positions, minimal transforms

## Example: Disco Ball Interior

```python
# 1. Generate image
disco_img = genmedia.run(
    "krea/v2/medium/turbo/text-to-image",
    prompt="interior disco ball mirror facets...",
    aspect_ratio="16:9"
)

# 2. Segment with Felzenszwalb (best for fine detail)
from skimage import segmentation
from PIL import Image
import numpy as np

img = Image.open(disco_img["images"][0]["url"])
segments = segmentation.felzenszwalb(
    np.array(img),
    scale=120,
    sigma=0.8,
    min_size=400
)

# Extract ~390 facets
# Create p5.js animation with individual shimmer per facet
```

**Result:** Each of 390 mirror facets animates independently while preserving the photorealistic composition.

## Tips

- **Start with SAM2** for prototyping — fastest workflow
- **Use Felzenszwalb** when you need fine segmentation granularity
- **Tune parameters** on a test image before batch processing
- **Filter by area** to remove noise (tiny segments) and background (huge segments)
- **Preserve composition** — animate in-place rather than moving facets around
- **Subtle transforms** work best: 3-6% scale pulsing, brightness shimmer, minimal rotation
