# Okonomi

Okonomi is an Agent meant for experimenting with creative image creation.

## Features

- **Creative Strategy Tools**: Leverages multiple creative thinking techniques to generate unique prompts
- **Multi-modal Image Generation**: Support for text-to-image, image-to-image, image editing, and style transfer
- **LoRA Integration**: Dynamic loading and combination of LoRA models for enhanced control
- **Memory System**: ChromaDB-based memory for learning from past generations
- **Automatic Evaluation**: Built-in evaluation system for prompt and image quality
- **MCP Tools**: Modular tool architecture for extensibility

## Installation

### Prerequisites

- Python 3.11 or higher
- uv package manager (recommended) or pip

### Clone the repository

```shell
git clone https://github.com/dn6/okonomi.git
cd okonomi
uv pip install -e .
```

## Configuration

### Required Environment Variables

Create a `.env` file in your project root with the following variables or set them manually in your environment.

```bash
# Hugging Face API Token (Required to use LLMs)
# Get from: https://huggingface.co/settings/tokens
HF_TOKEN=your_huggingface_token

# Fal.ai API Key (Required for Image Generation)
# Get from: https://fal.ai/dashboard/keys
FAL_KEY=your_fal_api_key

# ChromaDB API Key (Optional)
# Only required for cloud-hosted ChromaDB. Leave unset for local storage
CHROMA_API_KEY=your_chroma_api_key
```

### Optional Environment Variables

```bash
# Enable observability/tracing (Optional)
LOG_CREATE_AGENT_TRACES=false

# Custom configuration file path (Optional)
OKONOMI_CONFIG=/path/to/custom/config.json
```

### Configuring Memory

Generated images are available through [FAL's](https://fal.ai) CDN. You can also save the generated images to a Hugging Face dataset by 
creating a [dataset repository](https://huggingface.co/new-dataset) and setting the `image_repo_id` to your repo id in the `config.json` file.


## Usage

### Interactive Mode

```bash
# Start interactive session
okonomi

# You'll be prompted to enter your image generation request
```

### Direct Command

```bash
# Generate an image with a direct command
okonomi "Create a cyberpunk city at sunset"

# Or using the --task flag
okonomi --task "Design a futuristic logo"
```

### Command-Line Options

```bash
# Disable automatic evaluation (enable manual review of generated images)
okonomi --eval "Generate a landscape painting"

# Disable automatic planning (enable plan approval prompts)
okonomi --plan "Create a character portrait"

# Use custom configuration file
okonomi --config custom-config.json "Generate artwork"
```

## Configuration File

The default configuration is in `config.json`. Key settings include:

- **Agent Settings**: Model selection, temperature, planning interval
- **Inference Providers**: HuggingFace and Fal.ai model configurations
- **Memory Settings**: ChromaDB collections for storing evaluations and LoRAs
- **Tool Settings**: MCP tool configurations and callbacks

### Custom Configuration

You can create a custom configuration file and specify it:

```bash
okonomi --config my-config.json "Generate image"
```

Or set a default via environment variable:

```bash
export OKONOMI_CONFIG=/path/to/my-config.json
okonomi "Generate image"
```

## MCP Tools

Okonomi includes several MCP (Model Context Protocol) tools:

- **prompt.py**: Creative prompt generation and image description
- **image.py**: Image generation tools (text-to-image, image-to-image, editing)
- **memory.py**: Memory management and LoRA discovery
- **strategies.py**: Creative strategy selection

## Troubleshooting

## Examples

### Basic Text-to-Image

```bash
okonomi "A serene mountain landscape at dawn with mist"
```

### With Creative Strategy

```bash
okonomi "Transform the concept of time into a visual metaphor"
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues, questions, or suggestions, please open an issue on GitHub.
