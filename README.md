# Okonomi

Okonomi is an Agent meant for experimenting with creative image creation.

## Features

- **Code Execution**: A persistent Python REPL whose variables survive across turns and sessions
- **Session Memory**: Sessions are indexed and searchable — recall past conversations, prompts and outputs
- **Skills**: Progressively-disclosed instruction sets (`ffmpeg`, `p5js`, `research`, `yt-dlp`) loaded on demand
- **Background Jobs**: Long-running code, shell commands and sub-agents dispatch to the background
- **Tools**: Shell, file editing, web search and fetch

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
# OpenRouter API Key (Required — all model calls route through OpenRouter)
# Get from: https://openrouter.ai/keys
OPENROUTER_API_KEY=your_openrouter_api_key
```

### Optional Environment Variables

```bash
# Tavily API Key — required for web_search / web_fetch
# Get from: https://app.tavily.com
TAVILY_API_KEY=your_tavily_api_key

# Storage root for sessions, research, skills and favourites
# Default: ~/.okonomi
OKONOMI_HOME=/path/to/okonomi/home

# Skip the shell-command safety checks in the `bash` tool
OKONOMI_YOLO=false
```

## Usage

### Interactive Mode

```bash
# Start interactive session
okonomi

# You'll be prompted to enter your image generation request
```

### One-off Task

```bash
okonomi --task "Design a futuristic logo"
```

### Command-Line Options

```bash
# Resume from the latest session
okonomi --resume

# Load a specific session by ID
okonomi --session <id>

# Show executed code and full tool output
okonomi --verbose

# Run as a Discord bot (requires DISCORD_BOT_TOKEN)
okonomi --discord
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues, questions, or suggestions, please open an issue on GitHub.
