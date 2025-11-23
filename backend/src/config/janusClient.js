import crypto from 'node:crypto';
import WebSocket from 'ws';
import { config } from './env.js';

const JANUS_PLUGIN = 'janus.plugin.videoroom';

export class JanusWsClient {
    constructor({ url = config.janusWsUrl, apiSecret = config.janusApiSecret, requestTimeout = config.requestTimeoutMs } = {}) {
        this.url = url;
        this.apiSecret = apiSecret;
        this.requestTimeout = requestTimeout;
        this.pending = new Map();
        this.sessionId = null;
        this.handleId = null;
        this.ws = null;
    }

    async connect() {
        if (this.ws) {
            return;
        }

        this.ws = new WebSocket(this.url, 'janus-protocol', {
            handshakeTimeout: this.requestTimeout,
        });

        await new Promise((resolve, reject) => {
            const onError = (err) => {
                this.ws?.off('open', onOpen);
                reject(err);
            };
            const onOpen = () => {
                this.ws?.off('error', onError);
                resolve();
            };
            this.ws.once('open', onOpen);
            this.ws.once('error', onError);
        });

        this.ws.on('message', (raw) => {
            try {
                const message = JSON.parse(raw.toString());
                this.handleMessage(message);
            } catch (err) {
                // eslint-disable-next-line no-console
                console.error('Failed to parse Janus message', err);
            }
        });

        this.ws.on('error', (err) => {
            for (const [, pending] of this.pending) {
                pending.reject(err);
            }
            this.pending.clear();
        });
    }

    async close() {
        if (!this.ws) {
            return;
        }
        await new Promise((resolve) => {
            this.ws?.once('close', resolve);
            this.ws?.close();
        });
        this.ws = null;
    }

    async createSession() {
        const response = await this.send({ janus: 'create' });
        this.sessionId = response.data?.id;
        return this.sessionId;
    }

    async attachVideoRoom() {
        if (!this.sessionId) {
            throw new Error('Cannot attach plugin before creating a session');
        }
        const response = await this.send({
            janus: 'attach',
            session_id: this.sessionId,
            plugin: JANUS_PLUGIN,
            opaque_id: `node-gateway-${crypto.randomUUID().slice(0, 8)}`,
        });
        this.handleId = response.data?.id;
        return this.handleId;
    }

    async destroySession() {
        if (!this.sessionId) {
            return;
        }
        try {
            await this.send({ janus: 'destroy', session_id: this.sessionId });
        } catch (err) {
            // eslint-disable-next-line no-console
            console.warn('Failed to destroy Janus session', err.message);
        } finally {
            this.sessionId = null;
            this.handleId = null;
        }
    }

    async createRoom({ roomId, description, publishers }) {
        const result = await this.sendPluginMessage({
            request: 'create',
            room: roomId,
            description,
            publishers,
            bitrate: 512000,
            audiolevel_event: true,
            videoorient_ext: true,
        });
        return result;
    }

    async listRooms() {
        const result = await this.sendPluginMessage({ request: 'list' });
        return result?.list ?? [];
    }

    async sendPluginMessage(body) {
        if (!this.sessionId || !this.handleId) {
            throw new Error('Session and plugin handle are required before sending plugin messages');
        }
        const response = await this.send({
            janus: 'message',
            session_id: this.sessionId,
            handle_id: this.handleId,
            body,
        });
        return response.plugindata?.data;
    }

    async send(payload) {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            throw new Error('Janus connection is not open');
        }

        const transaction = crypto.randomUUID();
        const message = {
            ...payload,
            transaction,
        };

        if (this.apiSecret) {
            message.apisecret = this.apiSecret;
        }

        return new Promise((resolve, reject) => {
            const timeout = setTimeout(() => {
                this.pending.delete(transaction);
                reject(new Error(`Janus request timed out after ${this.requestTimeout}ms`));
            }, this.requestTimeout);

            this.pending.set(transaction, {
                resolve: (data) => {
                    clearTimeout(timeout);
                    resolve(data);
                },
                reject: (err) => {
                    clearTimeout(timeout);
                    reject(err);
                },
            });

            this.ws.send(JSON.stringify(message), (err) => {
                if (err) {
                    clearTimeout(timeout);
                    this.pending.delete(transaction);
                    reject(err);
                }
            });
        });
    }

    handleMessage(message) {
        const { transaction } = message;
        if (!transaction) {
            return;
        }

        const pending = this.pending.get(transaction);
        if (!pending) {
            return;
        }

        if (message.janus === 'ack') {
            return;
        }

        this.pending.delete(transaction);

        if (message.janus === 'error') {
            pending.reject(new Error(message.error?.reason ?? 'Janus returned an error'));
            return;
        }

        pending.resolve(message);
    }
}

