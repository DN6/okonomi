"""Discord bot interface for Okonomi.

Runs the Okonomi agent as a Discord bot, routing messages to the agent
and sending back text responses with image/video attachments.

Launch via: okonomi --discord
"""

import asyncio
import io
import logging
import os
from pathlib import Path

import discord
import httpx
from discord import app_commands

from ..agent import OkonomiAgent
from ..config import load_env_or_die


logger = logging.getLogger(__name__)

DISCORD_MAX_MESSAGE_LENGTH = 2000
DISCORD_MAX_FILE_SIZE = 25 * 1024 * 1024


class AgentSession:
    """Wraps an OkonomiAgent for a single Discord conversation context."""

    def __init__(self):
        self.agent = OkonomiAgent()
        self.lock = asyncio.Lock()
        self._prev_output_count = 0

    async def run(self, message: str, reset: bool = False) -> str:
        async with self.lock:
            self._prev_output_count = len(self.agent.state.artifacts)
            return await self.agent.run_async(message, reset=reset)

    def new_outputs(self) -> list:
        return self.agent.state.artifacts[self._prev_output_count :]


class OkonomiBot(discord.Client):
    """Discord bot that routes messages to OkonomiAgent instances."""

    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.dm_messages = True
        intents.guild_messages = True
        super().__init__(intents=intents)

        self.tree = app_commands.CommandTree(self)
        self._sessions: dict[str, AgentSession] = {}
        self._allowed_users: set[int] = set()
        self._setup_allowed_users()
        self._setup_commands()

    def _setup_allowed_users(self) -> None:
        allowed = os.getenv("DISCORD_ALLOWED_USERS", "")
        for uid in allowed.split(","):
            uid = uid.strip()
            if uid.isdigit():
                self._allowed_users.add(int(uid))

    def _is_allowed(self, user_id: int) -> bool:
        if not self._allowed_users:
            return True
        return user_id in self._allowed_users

    def _session_key(self, message: discord.Message) -> str:
        if isinstance(message.channel, discord.DMChannel):
            return f"dm:{message.author.id}"
        thread_id = message.channel.id if isinstance(message.channel, discord.Thread) else None
        if thread_id:
            return f"guild:{message.guild.id}:thread:{thread_id}"
        return f"guild:{message.guild.id}:channel:{message.channel.id}:user:{message.author.id}"

    def _interaction_key(self, interaction: discord.Interaction) -> str:
        channel = interaction.channel
        if isinstance(channel, discord.DMChannel):
            return f"dm:{interaction.user.id}"
        if isinstance(channel, discord.Thread):
            return f"guild:{interaction.guild_id}:thread:{channel.id}"
        return f"guild:{interaction.guild_id}:channel:{channel.id}:user:{interaction.user.id}"

    def _get_session(self, key: str) -> AgentSession:
        if key not in self._sessions:
            self._sessions[key] = AgentSession()
        return self._sessions[key]

    def _setup_commands(self) -> None:
        @self.tree.command(name="new", description="Start a new conversation")
        async def cmd_new(interaction: discord.Interaction):
            if not self._is_allowed(interaction.user.id):
                await interaction.response.send_message("Not authorized.", ephemeral=True)
                return
            key = self._interaction_key(interaction)
            if key in self._sessions:
                try:
                    self._sessions[key].agent.save_session()
                except Exception:
                    pass
                del self._sessions[key]
            await interaction.response.send_message("Started a new conversation.")

        @self.tree.command(name="status", description="Show current session info")
        async def cmd_status(interaction: discord.Interaction):
            if not self._is_allowed(interaction.user.id):
                await interaction.response.send_message("Not authorized.", ephemeral=True)
                return
            key = self._interaction_key(interaction)
            session = self._sessions.get(key)
            if not session:
                await interaction.response.send_message("No active session.", ephemeral=True)
                return
            agent = session.agent
            usage = agent.session.tokens
            lines = [
                f"**Session:** `{agent.session.session_id[:8]}`",
                f"**Task:** {agent.session.task_input or 'N/A'}",
                f"**Artifacts:** {len(agent.state.artifacts)}",
                f"**Tokens:** {usage.total_tokens:,} ({usage.requests} requests)",
            ]
            await interaction.response.send_message("\n".join(lines), ephemeral=True)

        @self.tree.command(name="help", description="Show available commands")
        async def cmd_help(interaction: discord.Interaction):
            help_text = (
                "**Okonomi Bot Commands**\n"
                "`/new` — Start a new conversation\n"
                "`/status` — Show current session info\n"
                "`/help` — Show this message\n\n"
                "Send a message to start creating. In servers, @mention the bot."
            )
            await interaction.response.send_message(help_text, ephemeral=True)

    async def setup_hook(self) -> None:
        await self.tree.sync()
        logger.info("Slash commands synced")

    async def on_ready(self) -> None:
        logger.info(f"Logged in as {self.user} (ID: {self.user.id})")

    async def on_message(self, message: discord.Message) -> None:
        if message.author == self.user:
            return
        if message.author.bot:
            return
        if not self._is_allowed(message.author.id):
            return

        if not self._should_respond(message):
            return

        content = message.content
        if self.user:
            content = content.replace(f"<@{self.user.id}>", "").replace(f"<@!{self.user.id}>", "").strip()

        image_urls = []
        for attachment in message.attachments:
            if attachment.content_type and attachment.content_type.startswith("image/"):
                image_urls.append(attachment.url)

        if image_urls:
            urls_text = "\n".join(f"[Attached image: {url}]" for url in image_urls)
            content = f"{content}\n{urls_text}" if content else urls_text

        if not content:
            return

        key = self._session_key(message)
        session = self._get_session(key)

        async with message.channel.typing():
            try:
                response = await session.run(content)
            except Exception as e:
                logger.error(f"Agent error: {e}", exc_info=True)
                await message.reply(f"Error: {e}", mention_author=False)
                return

        files = await self._collect_media(session)
        await self._send_response(message, response, files)

        try:
            session.agent.save_session()
        except Exception as e:
            logger.warning(f"Failed to save session: {e}")

    def _should_respond(self, message: discord.Message) -> bool:
        if isinstance(message.channel, discord.DMChannel):
            return True
        if self.user in message.mentions:
            return True
        if (
            message.reference
            and message.reference.resolved
            and isinstance(message.reference.resolved, discord.Message)
            and message.reference.resolved.author == self.user
        ):
            return True
        return False

    async def _collect_media(self, session: AgentSession) -> list[discord.File]:
        files = []
        for artifact in session.new_outputs():
            if artifact.artifact_type not in ("image", "video", "audio", "3d"):
                continue
            path = artifact.path
            url = artifact.url

            if path and Path(path).exists() and Path(path).stat().st_size < DISCORD_MAX_FILE_SIZE:
                try:
                    files.append(discord.File(path, filename=Path(path).name))
                except Exception:
                    pass
            elif url:
                try:
                    data = await self._download_url(url)
                    if data and len(data) < DISCORD_MAX_FILE_SIZE:
                        ext = ".jpg" if artifact.artifact_type == "image" else ".mp4"
                        filename = f"{artifact.artifact_type}_{len(files)}{ext}"
                        files.append(discord.File(io.BytesIO(data), filename=filename))
                except Exception:
                    pass

        return files

    async def _download_url(self, url: str) -> bytes | None:
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(url, follow_redirects=True)
                response.raise_for_status()
                return response.content
        except Exception:
            return None

    async def _send_response(
        self,
        original: discord.Message,
        text: str,
        files: list[discord.File],
    ) -> None:
        chunks = _chunk_message(text)

        if not chunks and not files:
            return

        first_chunk = chunks[0] if chunks else ""
        try:
            await original.reply(
                first_chunk,
                files=files if files else None,
                mention_author=False,
            )
        except discord.HTTPException:
            await original.channel.send(first_chunk, files=files if files else None)

        for chunk in chunks[1:]:
            try:
                await original.channel.send(chunk)
            except discord.HTTPException as e:
                logger.warning(f"Failed to send chunk: {e}")
                break


def _chunk_message(text: str) -> list[str]:
    if not text:
        return []
    if len(text) <= DISCORD_MAX_MESSAGE_LENGTH:
        return [text]

    chunks = []
    remaining = text
    in_code_block = False
    code_lang = ""

    while remaining:
        if len(remaining) <= DISCORD_MAX_MESSAGE_LENGTH:
            chunks.append(remaining)
            break

        limit = DISCORD_MAX_MESSAGE_LENGTH
        if in_code_block:
            limit -= 10

        split_at = remaining.rfind("\n", 0, limit)
        if split_at < limit // 2:
            split_at = remaining.rfind(" ", 0, limit)
        if split_at < limit // 2:
            split_at = limit

        chunk = remaining[:split_at]
        remaining = remaining[split_at:].lstrip("\n")

        fences = chunk.count("```")
        if fences % 2 == 1:
            if in_code_block:
                in_code_block = False
            else:
                in_code_block = True
                fence_pos = chunk.rfind("```")
                after = chunk[fence_pos + 3 : fence_pos + 20]
                code_lang = after.split("\n")[0].strip()

        if in_code_block:
            chunk += "\n```"
            remaining = f"```{code_lang}\n{remaining}"

        chunks.append(chunk)

    return chunks


def run_discord_bot() -> None:
    """Entry point for launching the Discord bot."""
    token = os.getenv("DISCORD_BOT_TOKEN")
    if not token:
        raise EnvironmentError("DISCORD_BOT_TOKEN not set. Set it in your .env file or environment.")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    load_env_or_die()
    bot = OkonomiBot()

    logger.info("Starting Okonomi Discord bot...")
    bot.run(token, log_handler=None)
