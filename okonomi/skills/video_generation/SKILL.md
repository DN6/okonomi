---
name: video-generation
description: "Video generation prompting and production pipeline. Creates AI-generated videos from still images, frame pairs, or text concepts — with intentional camera work, cinematic style, and post-production assembly. Covers: prompt crafting for Veo 3.1 / Kling / Seedance, camera direction, lighting, scene production workflows, character consistency across scenes, video stitching, crossfades, and audio layering. Trigger when users request: video from images, scene animation, cinematic sequences, video transitions, motion from stills, video editing/assembly, or video prompting advice."
metadata:
  tags: [video, animation, cinematography, veo, kling, post-production, ffmpeg]
  related_skills: [concept-research]
---

# Video Generation Pipeline

## Creative Standard

Video is a time-based medium — every frame must serve the motion and the story. Before generating a single clip, decide what the camera reveals, how the subject moves, and what the viewer should feel.

**Descriptive, not instructional.** Video prompts describe a scene unfolding — never tell the model what to do.

Bad: "Make the water move and add some wind"
Good: "A serene lake with gentle ripples spreading across the surface, soft wind rustling through willow branches"

**One motion per clip.** Each generation should have a single, clear camera movement. Combining dolly + pan + zoom produces confused, unnatural results.

**Lighting sells realism.** Lighting direction matters more than subject detail. Always specify it.

## Modes

| Mode | Input | Output | Tools |
|------|-------|--------|-------|
| **Image-to-video** | Single source image | Animated clip with motion | `video.image_to_video()` |
| **Frame transition** | First + last frame images | Video bridging two states | `video.first_last_frame_to_video()` |
| **Scene production** | Concept → hero frame → variants | Multi-scene storyboard with animated clips | `image_generation.*` → `video.*` |
| **Assembly** | Multiple clips | Stitched final video with transitions and audio | `run()` with ffmpeg |

## Prompt Anatomy

Build prompts by layering seven elements. Not all are required, but specificity improves results.

| Element | What It Controls | Reference |
|---------|-----------------|-----------|
| **Shot framing & camera motion** | How the camera frames and moves | `references/camera-and-framing.md` |
| **Visual style** | Medium, film stock, aesthetic | `references/visual-style.md` § Film Stocks |
| **Lighting** | Illumination, mood, atmosphere | `references/visual-style.md` § Lighting |
| **Character descriptions** | Physical traits, clothing, expression | — |
| **Location & environment** | Setting, time, weather, textures | — |
| **Action & motion** | What unfolds in the scene | — |
| **Audio & sound** (Veo 3.1) | Dialogue, ambient, SFX, music | — |

### Character Descriptions
Specific beats generic. Layer physical traits, clothing, expression, and motion.

Weak: "A man walks through a city"
Strong: "A tall man in his thirties with a close-cropped beard, wearing a rumpled linen suit, walks briskly through narrow cobblestone streets, glancing over his shoulder"

### Location & Environment
Build the world with sensory detail — textures, atmosphere, scale.

Weak: "A jazz club"
Strong: "A dimly lit basement jazz club, blue cigarette smoke hanging in spotlights, exposed brick walls lined with framed black-and-white photographs, a small stage with a worn upright piano"

### Action & Motion
Describe what happens as a continuous unfolding, not a command.

Weak: "The bird flies away"
Strong: "A heron launches from the riverbank, wings spreading wide, water droplets falling from its legs as it climbs into a pale morning sky"

### Audio & Sound (Veo 3.1)
Use separate sentences to describe audio. Veo 3.1 generates synchronized sound.

- **Dialogue**: "The man in the red hat says: Where is the rabbit?"
- **Ambient**: "The hum of distant traffic, crickets chirping"
- **SFX**: "Footsteps echoing on marble, a glass clinking"
- **Music**: "A slow jazz piano melody playing softly in the background"

## Prompt Templates

### Image-to-Video
```
[Camera movement], [lighting conditions]. [Subject description] [action/motion].
[Environmental details in motion]. [Atmospheric elements].
```

Example: "Slow dolly forward, warm golden hour lighting. A woman in a flowing white dress stands at the edge of a cliff, her hair and dress billowing in a strong coastal wind. Waves crash against rocks below, sending spray into the amber light. Seabirds glide past in the distance."

### First-Last Frame Transition
```
[Opening state description]. [Transition action — what changes and how].
[Ending state description]. [Continuous elements that persist throughout].
```

Example: "A closed flower bud in soft morning light. Petals slowly unfurl in timelapse, dewdrops catching prismatic light as they roll off opening petals. The fully bloomed flower sways gently, a bee approaching from the right. Soft bokeh background throughout."

### Cinematic Narrative
```
[Visual style and medium]. [Establishing shot/framing].
[Character introduction with physical detail]. [Action sequence].
[Environmental reaction]. [Emotional beat/resolution].
[Sound design notes].
```

## Production Workflow

The most effective approach: **build scenes as stills, then animate**.

### Step 1: Generate the Hero Frame
Create the primary scene with `image_generation.text_to_image()`. Refine composition, lighting, and character design until right.

### Step 2: Build Scene Variants
Use `image_generation.edit_image()` and `image_generation.reference_image()` to create multiple scenes from the hero frame:
- **Edit** to change elements — swap backgrounds, alter time of day, change expressions, add/remove objects
- **Reference** to maintain character/style consistency — use the hero as style or character reference while placing the subject in new compositions

This gives a storyboard of consistent frames without regenerating from scratch.

### Step 3: Animate Individual Scenes
Bring each scene to life with `video.image_to_video()` — add camera motion, environmental animation, and character movement.

### Step 4: Bridge Scenes with Transitions
Use `video.first_last_frame_to_video()` to create smooth transitions. Feed scene A as first frame, scene B as last frame, with a prompt describing the transition.

### Step 5: Stitch & Finish
Combine clips into a final video with ffmpeg. Add crossfades, audio, color grading. See `references/post-production.md` for the full ffmpeg toolkit.

### Example
```python
# 1. Hero frame
hero = image_generation.text_to_image(prompt="A detective in a trenchcoat standing under a streetlamp, rain-soaked noir city, 1940s")

# 2. Scene variants
scene_office = image_generation.edit_image(prompt="Same detective sitting at a cluttered desk in a dim office, whiskey glass, case files", image=hero["images"][0])
scene_alley = image_generation.reference_image(prompt="The detective crouching behind a dumpster in a dark alley, gun drawn, rain pouring", images=[hero["images"][0]])

# 3. Animate each scene
vid_street = video.image_to_video(prompt="Camera slowly dollies forward, rain falling through the cone of lamplight, the detective's coat rippling in the wind", image=hero["images"][0])
vid_office = video.image_to_video(prompt="Static shot, cigarette smoke curling upward, the detective turns a page, clock ticking on the wall", image=scene_office["images"][0])

# 4. Transition
transition = video.first_last_frame_to_video(first_frame=hero["images"][0], last_frame=scene_office["images"][0], prompt="The detective turns and walks through a doorway, the rain-soaked street dissolving into a warm dim office interior")
```

Then stitch with ffmpeg — see `references/post-production.md`.

## Common Mistakes

- **Instructional language**: "Make it zoom in" → "Camera slowly zooms into..."
- **Vague descriptions**: "A cool scene" → describe exactly what makes it cool
- **Conflicting camera movements**: One primary motion per clip
- **Missing lighting**: Always specify — it drives mood more than subject detail
- **No atmosphere**: Add particles — dust, fog, rain, floating embers
- **Overloaded prompts**: Each clip should be one coherent moment
- **Skipping stills**: Always refine the frame before animating it
- **Expensive re-runs**: If a generation fails on JSON parsing, `print(result)` to inspect — don't regenerate

## Style Quick Reference

| Intent | Prompt Fragment |
|--------|----------------|
| Dreamy/ethereal | "Soft focus, diffused backlight, slow motion, pastel color palette" |
| Gritty/realistic | "Handheld camera, desaturated colors, harsh available light, film grain" |
| Epic/grand | "Wide aerial shot, dramatic orchestral tone, golden hour, vast landscape" |
| Intimate/personal | "Tight close-up, shallow depth of field, warm practical lighting, quiet" |
| Suspenseful | "Low angle, deep shadows, slow dolly forward, muted color palette" |
| Joyful/energetic | "Bright saturated colors, dynamic tracking shot, fast motion, lens flare" |
| Nostalgic | "Super 8 film texture, warm color grading, slightly overexposed, soft vignette" |
| Futuristic | "Clean digital, cool blue-teal palette, volumetric lighting, reflective surfaces" |

For detailed lookup tables: `references/camera-and-framing.md`, `references/visual-style.md`, `references/post-production.md`.
