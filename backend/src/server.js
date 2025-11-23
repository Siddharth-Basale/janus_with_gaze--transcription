import express from 'express';
import cors from 'cors';
import { WebSocketServer } from 'ws';
import { createServer } from 'http';
import { config } from './config/env.js';
import roomsRouter from './routes/rooms.js';
import { processFrame } from './services/mlService.js';

const app = express();

app.use(cors());
app.use(express.json({ limit: '10mb' })); // Increase limit for frame data

app.get('/health', (req, res) => {
    res.json({ status: 'ok', janus: config.janusWsUrl });
});

app.get('/api/config', (req, res) => {
    res.json({
        janus: {
            websocket: config.janusWsUrl,
            rest: config.janusRestUrl,
        },
        iceServers: config.iceServers,
    });
});

app.use('/api/rooms', roomsRouter);

app.use((err, req, res, next) => {
    // eslint-disable-next-line no-console
    console.error(err);
    if (res.headersSent) {
        return next(err);
    }
    return res.status(500).json({ error: err.message ?? 'Unexpected error' });
});

// Create HTTP server
const server = createServer(app);

// Create WebSocket server for ML frame processing
const wss = new WebSocketServer({ 
    server,
    path: '/ws/ml',
});

wss.on('connection', (ws, req) => {
    console.log('[ML WebSocket] Client connected');
    
    ws.on('message', async (data) => {
        try {
            const message = JSON.parse(data.toString());
            
            if (message.session_id && message.frame_data) {
                // Forward to Python ML service
                const result = await processFrame(message.session_id, message.frame_data);
                
                // Send result back to client
                ws.send(JSON.stringify(result));
            } else {
                ws.send(JSON.stringify({ error: 'Invalid message format' }));
            }
        } catch (error) {
            console.error('[ML WebSocket] Processing error:', error.message);
            ws.send(JSON.stringify({ 
                error: error.message || 'Frame processing failed' 
            }));
        }
    });
    
    ws.on('error', (error) => {
        console.error('[ML WebSocket] Error:', error);
    });
    
    ws.on('close', () => {
        console.log('[ML WebSocket] Client disconnected');
    });
});

server.listen(config.port, () => {
    // eslint-disable-next-line no-console
    console.log(`Backend listening on http://localhost:${config.port}`);
    // eslint-disable-next-line no-console
    console.log(`ML WebSocket server ready at ws://localhost:${config.port}/ws/ml`);
});

