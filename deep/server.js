const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const { createClient, LiveTranscriptionEvents } = require('@deepgram/sdk');
const dotenv = require('dotenv');
const path = require('path');

dotenv.config();

const app = express();
const server = http.createServer(app);
const io = new Server(server);

// Serve static files from the public directory
app.use(express.static('public'));

// Serve the main page
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Store active connections
const activeConnections = new Map();

io.on('connection', (socket) => {
  console.log('Client connected:', socket.id);

  socket.on('start-transcription', async (data) => {
    try {
      // Create Deepgram client
      const deepgram = createClient(process.env.DEEPGRAM_API_KEY);

      // Create live transcription connection with Nova-3 model
      const connection = deepgram.listen.live({
        model: 'nova-3',
        language: 'en-US',
        smart_format: true,
        interim_results: true,
        endpointing: 300,
        encoding: 'linear16',
        sample_rate: 16000,
        channels: 1,
      });

      // Store connection for this socket
      activeConnections.set(socket.id, connection);

      // Listen for events
      connection.on(LiveTranscriptionEvents.Open, () => {
        console.log('Deepgram connection opened');
        socket.emit('status', { message: 'Connected to Deepgram', connected: true });
      });

      connection.on(LiveTranscriptionEvents.Close, () => {
        console.log('Deepgram connection closed');
        socket.emit('status', { message: 'Disconnected from Deepgram', connected: false });
        activeConnections.delete(socket.id);
      });

      connection.on(LiveTranscriptionEvents.Transcript, (data) => {
        const transcript = data.channel?.alternatives?.[0]?.transcript;
        if (transcript && transcript.length > 0) {
          const isFinal = data.is_final || false;
          socket.emit('transcription', {
            transcript,
            isFinal,
            confidence: data.channel.alternatives[0].confidence,
          });
        }
      });

      connection.on(LiveTranscriptionEvents.Metadata, (data) => {
        console.log('Metadata:', data);
      });

      connection.on(LiveTranscriptionEvents.Error, (err) => {
        console.error('Deepgram error:', err);
        socket.emit('error', { message: err.message || 'Deepgram error occurred' });
      });

      // Listen for audio chunks from microphone
      socket.on('audio-chunk', (chunk) => {
        if (connection && connection.getReadyState() === 1) {
          // chunk is an ArrayBuffer, convert to Buffer
          const audioBuffer = Buffer.from(chunk);
          connection.send(audioBuffer);
        }
      });
    } catch (error) {
      console.error('Error setting up transcription:', error);
      socket.emit('error', { message: error.message || 'Failed to start transcription' });
    }
  });

  socket.on('stop-transcription', () => {
    const connection = activeConnections.get(socket.id);
    if (connection) {
      connection.finish();
      activeConnections.delete(socket.id);
    }
    socket.emit('status', { message: 'Transcription stopped', connected: false });
  });

  socket.on('disconnect', () => {
    console.log('Client disconnected:', socket.id);
    const connection = activeConnections.get(socket.id);
    if (connection) {
      connection.finish();
      activeConnections.delete(socket.id);
    }
  });
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
  console.log('Make sure to set DEEPGRAM_API_KEY in your .env file');
});

