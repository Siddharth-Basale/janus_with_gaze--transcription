import dotenv from 'dotenv';

dotenv.config();

const parseIceServers = () => {
    if (!process.env.ICE_SERVERS) {
        return [{ urls: 'stun:stun.l.google.com:19302' }];
    }

    try {
        const parsed = JSON.parse(process.env.ICE_SERVERS);
        return Array.isArray(parsed) ? parsed : [{ urls: 'stun:stun.l.google.com:19302' }];
    } catch {
        return [{ urls: 'stun:stun.l.google.com:19302' }];
    }
};

export const config = {
    port: Number(process.env.PORT ?? 4000),
    janusWsUrl: process.env.JANUS_WS_URL ?? 'ws://localhost:8188',
    janusRestUrl: process.env.JANUS_REST_URL ?? 'http://localhost:8088/janus',
    janusApiSecret: process.env.JANUS_API_SECRET ?? '',
    janusDefaultPublishers: Number(process.env.JANUS_MAX_PUBLISHERS ?? 6),
    requestTimeoutMs: Number(process.env.JANUS_REQUEST_TIMEOUT_MS ?? 8000),
    iceServers: parseIceServers(),
};

