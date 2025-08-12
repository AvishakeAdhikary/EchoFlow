# EchoFlow
A production-ready voice-to-voice AI communication system featuring a WebSocket server and a modern GUI client. The system supports real-time voice processing, text-based interaction, session management, and tool integration using an OpenAI-compatible API, Whisper for speech-to-text, and Kokoro TTS for text-to-speech.

## Features

### Server
- **WebSocket Server**: Handles multiple client connections with auto-reconnection and session persistence.
- **Voice Processing**: Integrates Whisper for speech-to-text and Kokoro TTS for text-to-speech.
- **Session Management**: Stores conversation history and session data in an SQLite database.
- **Model Context Protocol (MCP)**: Supports tool calls (e.g., time, calculator, weather).
- **Error Handling**: Robust error recovery with comprehensive logging.
- **OpenAI-Compatible API**: Configurable for integration with custom or third-party AI endpoints.

### Client
- **Modern GUI**: Dark-themed Tkinter interface with real-time audio visualization.
- **Real-Time Voice Recording/Playback**: Uses PyAudio for audio input/output with device selection.
- **WebSocket Client**: Auto-reconnects to the server with message queuing.
- **Conversation History**: Displays and manages chat history.
- **Text and Voice Input**: Supports both text and voice-based interactions.

## Showcase Video

Watch a short 10-15 second demonstration of the Voice AI System in action:

https://github.com/user-attachments/assets/acc122e4-1f60-4a0c-8a4c-0ec1751bc107

*Note*: The video showcases real-time voice input, server processing, and AI response playback through the GUI.

## Prerequisites

- **Python**: Version 3.8 or higher
- **[uv](https://docs.astral.sh/uv/)**: For dependency management and virtual environments
- **Audio Devices**: Microphone and speakers/headphones for voice input/output
- **Internet Connection**: Required for API calls if using external AI services
- **Operating System**: Compatible with Windows, macOS, or Linux

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/voice-ai-system.git
cd voice-ai-system
```

### 2. Install uv
Install `uv` if not already installed, following the [official uv installation guide](https://docs.astral.sh/uv/getting-started/installation/):
```bash
pip install uv
```

Verify installation:
```bash
uv --version
```

### 3. Set Up the Virtual Environment
Create a virtual environment and install dependencies using `uv`:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

The `pyproject.toml` defines dependencies, and `uv.lock` ensures reproducible builds. The `uv sync` command installs all required packages, including:
- `openai`
- `transformers`
- `torch`
- `torchaudio`
- `kokoro-tts`
- `pyaudio`
- `numpy`
- `matplotlib`
- `websockets`
- `aiohttp`

To export a `requirements.txt` for reference:
```bash
uv pip freeze > requirements.txt
```

### 4. Configure API Key
Edit `server.py` to set the OpenAI-compatible API endpoint and key:
```python
self.openai_client = AsyncOpenAI(
    base_url="http://localhost:8000/v1",  # Replace with your API endpoint
    api_key="your-api-key"               # Replace with your API key
)
```

*Note*: For local development, use `http://localhost:8000/v1` or your API provider's endpoint. Ensure the API server is running and accessible.

### 5. Configure Server URL
The client connects to the server at `ws://localhost:8765` by default. Update the server URL in `client.py` if deploying to a different host:
```python
self.server_url = "ws://localhost:8765"  # Update with your server address
```

For production, replace `localhost` with your server's public IP or domain, and ensure the server is accessible (e.g., port 8765 is open).

## Project Structure

```
voice-ai-system/
├── server.py           # WebSocket server with voice processing and API integration
├── client.py           # Tkinter GUI client for voice and text interaction
├── pyproject.toml      # Dependency configuration for uv
├── uv.lock             # Lock file for reproducible builds
├── voice_server.log    # Server log file (generated on run)
├── voice_ai.db         # SQLite database (generated on run)
├── README.md           # Project documentation
└── showcase.mp4        # Showcase video (10-15 seconds)
```

## Usage

### Running the Server
1. Activate the virtual environment:
   ```bash
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Start the server:
   ```bash
   uv run python server.py
   ```

   The server runs on `ws://localhost:8765` by default and logs to `voice_server.log`. The SQLite database (`voice_ai.db`) is created automatically.

### Running the Client
1. In a separate terminal, activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

2. Start the client:
   ```bash
   uv run python client.py
   ```

3. In the GUI:
   - **Server URL**: Enter `ws://localhost:8765` (or your server's address).
   - **Audio Devices**: Select input (microphone) and output (speakers) devices from the dropdowns.
   - **Connect**: Click **Connect** to establish a WebSocket connection.
   - **Voice Input**: Click **Start Recording** to send voice messages.
   - **Text Input**: Type in the text box and click **Send Text** (or press `Ctrl+Enter`).
   - **History**: View conversation history and audio visualizations in real-time.

### Example Interaction
1. Start the server and client.
2. Connect the client to `ws://localhost:8765`.
3. Record a voice message (e.g., "What's the current time?").
4. The server transcribes the audio, processes it via the API, and responds with text and audio.
5. The client displays the response and plays the audio output.

## Production Deployment

### Server
- **Host**: Deploy on a server with a public IP or domain. Update `HOST` in `server.py`:
  ```python
  HOST = "0.0.0.0"  # Or your server's IP/domain
  ```
- **Port**: Ensure port 8765 is open in your firewall/security group.
- **SSL/TLS**: For secure connections, use `wss://` with a reverse proxy (e.g., Nginx) and SSL certificates.
- **Process Management**: Use a process manager like `systemd` or `pm2` to keep the server running:
  ```bash
  uv run python server.py &
  ```
- **Logging**: Monitor `voice_server.log` for errors and performance.
- **Database**: Back up `voice_ai.db` regularly to prevent data loss.
- **Scaling**: For high traffic, consider load balancing or sharding the SQLite database.

### Client
- **Distribution**: Package the client using `PyInstaller` for standalone executables:
  ```bash
  uv run pyinstaller --onefile client.py
  ```
- **Server URL**: Update `self.server_url` in `client.py` to point to the production server.
- **Audio Devices**: Ensure users have compatible audio devices. Provide fallback options for text-only input.

### Dependencies
- Use `uv.lock` for consistent dependency versions across environments.
- Regularly update dependencies:
  ```bash
  uv sync --upgrade
  ```

### Security
- Secure the API key in `server.py` using environment variables:
  ```python
  import os
  self.openai_client = AsyncOpenAI(
      base_url=os.getenv("API_BASE_URL", "http://localhost:8000/v1"),
      api_key=os.getenv("API_KEY", "your-api-key")
  )
  ```
  Set environment variables before running:
  ```bash
  export API_BASE_URL="http://your-api-endpoint/v1"
  export API_KEY="your-api-key"
  ```
- Sanitize user inputs to prevent injection attacks.
- Enable HTTPS/WSS for production WebSocket connections.

## Configuration

### Server Configuration
- **Host/Port**: Modify in `server.py` (`main()` function):
  ```python
  HOST = "0.0.0.0"
  PORT = 8765
  ```
- **Database**: SQLite database (`voice_ai.db`) is created automatically. For production, consider a more robust database like PostgreSQL.
- **Model**: Change the default model in `server.py` (`Session` class):
  ```python
  model_name="google/medgemma-27b-it"
  ```
- **Voice Processing**: Adjust the STT model in `VoiceProcessor`:
  ```python
  VoiceProcessor(stt_model="openai/whisper-small")
  ```

### Client Configuration
- **Server URL**: Set in the GUI or modify in `client.py`:
  ```python
  self.server_url = "ws://localhost:8765"
  ```
- **Audio Settings**: Adjust `sample_rate`, `channels`, or `chunk_size` in `AudioManager` for performance or compatibility.

## Development

### Adding Tools
Extend `ModelContextProtocol` in `server.py` to add new tool functions:
```python
async def new_tool(self, **args) -> str:
    return "Result of new tool"
self.available_tools["new_tool"] = self.new_tool
```

### Customizing the GUI
Modify `setup_styles` and `create_widgets` in `VoiceAIClientGUI` to customize the appearance or add features like custom themes or buttons.

### Running Tests
Add tests to a `tests/` directory and run with `uv`:
```bash
uv run pytest
```

Ensure tests cover WebSocket communication, audio processing, and API integration.

## Troubleshooting

- **Dependency Issues**: Verify `uv.lock` is up-to-date:
  ```bash
  uv sync
  ```
- **Audio Device Errors**: Check device availability in the GUI dropdowns. Ensure PyAudio is configured correctly.
- **Connection Issues**: Verify the server is running and the URL (`ws://localhost:8765`) is correct. Check firewall settings.
- **API Errors**: Confirm the API endpoint and key in `server.py`. Test connectivity to the API server.
- **Performance**: For heavy loads, optimize `VoiceProcessor` by using a lighter STT model (e.g., `openai/whisper-tiny`).

## Contributing

1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit changes:
   ```bash
   git commit -m "Add your feature"
   ```
4. Push to the branch:
   ```bash
   git push origin feature/your-feature
   ```
5. Open a pull request with a detailed description.

Follow the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/).

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- Built with [uv](https://docs.astral.sh/uv/) for dependency management.
- Uses [Whisper](https://github.com/openai/whisper) for speech-to-text.
- Uses [Kokoro TTS](https://github.com/hexgrad/kokoro) for text-to-speech.
- Powered by [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/) and [Matplotlib](https://matplotlib.org/) for audio handling and visualization.
