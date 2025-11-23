# Deepgram Live Transcription with Nova-3

A real-time speech-to-text transcription application using Deepgram's Nova-3 model. This project displays live transcriptions on a web page.

## Features

- 🎤 Real-time live transcription using Deepgram Nova-3 model
- 🎙️ Microphone input - speak directly into your microphone
- 🌐 Web-based interface to view transcriptions
- 💬 Real-time updates via WebSocket
- 🎨 Modern, responsive UI

## Prerequisites

- Node.js (v14 or higher)
- npm or yarn
- Deepgram API key ([Get one here](https://console.deepgram.com/))

## Installation

1. **Clone or navigate to the project directory**

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Set up your environment variables:**
   - Copy `.env.example` to `.env`
   - Add your Deepgram API key:
     ```
     DEEPGRAM_API_KEY=your-actual-api-key-here
     ```

## Usage

1. **Start the server:**
   ```bash
   npm start
   ```

2. **Open your browser:**
   Navigate to `http://localhost:3000`

3. **Start transcription:**
   - Click "Start Transcription"
   - Allow microphone access when your browser prompts you
   - Start speaking into your microphone

4. **View live transcriptions:**
   - Transcriptions will appear in real-time in the transcription area
   - Final transcripts are shown in regular text
   - Interim (temporary) transcripts are shown in italic gray
   - Click "Stop" when you're done

## Project Structure

```
.
├── server.js          # Node.js server with Deepgram integration
├── package.json       # Dependencies and scripts
├── .env              # Environment variables (create from .env.example)
├── .env.example      # Example environment file
├── README.md         # This file
└── public/
    └── index.html    # Web interface
```

## Configuration

The server uses the following Deepgram settings:
- **Model:** `nova-3` (latest model)
- **Language:** `en-US`
- **Smart Formatting:** Enabled
- **Interim Results:** Enabled
- **Endpointing:** 300ms

You can modify these settings in `server.js` in the `deepgram.listen.live()` configuration.

## Troubleshooting

- **"Connection failed"**: Check that your Deepgram API key is correctly set in `.env`
- **No transcriptions appearing**: Verify the audio stream URL is accessible and contains audio
- **Port already in use**: Change the `PORT` in `.env` or stop the process using port 3000

## Resources

- [Deepgram Documentation](https://developers.deepgram.com/)
- [Deepgram API Reference](https://developers.deepgram.com/reference)
- [Nova-3 Model Info](https://developers.deepgram.com/docs/models-languages-overview)

## License

MIT

