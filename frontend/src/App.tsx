import { type FormEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import Janus from 'janus-gateway';
import adapter from 'webrtc-adapter';
import { FrameProcessor, type FrameProcessingResult } from './services/frameProcessor';
import { ConcentrationOverlay } from './components/ConcentrationOverlay';
import { mlConfig, updateMLConfig, type MLConfig } from './config/mlConfig';

declare global {
    interface Window {
        adapter?: typeof adapter;
    }
}

if (typeof window !== 'undefined') {
    window.adapter = adapter;
}

type RoomSummary = {
    roomId: number;
    description?: string;
    maxPublishers?: number;
};

type GatewayConfig = {
    janus: {
        websocket: string;
        rest: string;
    };
    iceServers: RTCIceServer[];
};

const fetchJson = async <T,>(input: RequestInfo, init?: RequestInit) => {
    const response = await fetch(input, init);
    if (!response.ok) {
        const text = await response.text();
        throw new Error(text || 'Request failed');
    }
    return response.json() as Promise<T>;
};

export default function App() {
    const [roomName, setRoomName] = useState('');
    const [displayName, setDisplayName] = useState(() => `user-${crypto.randomUUID().slice(0, 5)}`);
    const [status, setStatus] = useState<string | null>(null);
    const [config, setConfig] = useState<GatewayConfig | null>(null);
    const [loading, setLoading] = useState(false);
    const [joining, setJoining] = useState(false);
    const [activeRoomId, setActiveRoomId] = useState<number | null>(null);
    const [joinedRoomName, setJoinedRoomName] = useState<string | null>(null);
    const [view, setView] = useState<'lobby' | 'room'>('lobby');
    const [debugLines, setDebugLines] = useState<string[]>([]);
    const [localPreviewStream, setLocalPreviewStream] = useState<MediaStream | null>(null);
    const [janusDebugMode, setJanusDebugMode] = useState<boolean>(() => {
        if (typeof window === 'undefined') {
            return false;
        }
        return window.localStorage.getItem('janus-debug') === 'true';
    });

    const janusRef = useRef<any>(null);
    const janusPromiseRef = useRef<Promise<any> | null>(null);
    const publisherRef = useRef<any>(null);
    const ownFeedIdRef = useRef<number | null>(null);
    const privateIdRef = useRef<number | null>(null);
    const remoteFeedsRef = useRef<Record<number, any>>({});
    const remoteTrackStreamsRef = useRef<Record<number, MediaStream>>({});
    const localVideoRef = useRef<HTMLVideoElement | null>(null);
    const localStreamRef = useRef<MediaStream | null>(null);
    const [remoteStreams, setRemoteStreams] = useState<Array<{ id: number; stream: MediaStream; display?: string }>>([]);
    const localFrameProcessorRef = useRef<FrameProcessor | null>(null);
    const remoteFrameProcessorsRef = useRef<Map<number, FrameProcessor>>(new Map());
    const [localConcentration, setLocalConcentration] = useState<FrameProcessingResult | null>(null);
    const [remoteConcentrations, setRemoteConcentrations] = useState<Map<number, FrameProcessingResult>>(new Map());
    const [mlConfigOpen, setMlConfigOpen] = useState(false);
    const [currentMLConfig, setCurrentMLConfig] = useState<MLConfig>(mlConfig);

    const logDebug = useCallback((message: string) => {
        const line = `${new Date().toLocaleTimeString()} ${message}`;
        setDebugLines((prev) => [line, ...prev].slice(0, 40));
        // eslint-disable-next-line no-console
        console.debug('[janus-ui]', message);
    }, []);

    const reportStatus = useCallback(
        (message: string) => {
            setStatus(message);
            logDebug(message);
        },
        [logDebug],
    );

    const loadConfig = useCallback(async () => {
        try {
            const data = await fetchJson<GatewayConfig>('/api/config');
            setConfig(data);
        } catch (err) {
            reportStatus((err as Error).message);
        }
    }, [reportStatus]);

    useEffect(() => {
        loadConfig();
    }, [loadConfig]);

    const cleanupLocalMedia = useCallback(() => {
        if (localStreamRef.current instanceof MediaStream) {
            localStreamRef.current.getTracks().forEach((track) => track.stop());
        }
        localStreamRef.current = null;
        setLocalPreviewStream(null);
        if (localVideoRef.current) {
            localVideoRef.current.srcObject = null;
        }
    }, []);

    const addRemoteStream = useCallback((feedId: number, stream: MediaStream, display?: string) => {
        setRemoteStreams((prev) => {
            const existing = prev.find((feed) => feed.id === feedId);
            if (existing) {
                return prev.map((feed) => (feed.id === feedId ? { ...feed, stream } : feed));
            }
            return [...prev, { id: feedId, stream, display }];
        });
    }, []);

    const removeRemoteStream = useCallback((feedId: number) => {
        setRemoteStreams((prev) => prev.filter((feed) => feed.id !== feedId));
    }, []);

    const getJanusInstance = useCallback(async () => {
        if (janusRef.current) {
            return janusRef.current;
        }
        if (!config) {
            throw new Error('Janus config not ready yet');
        }
        if (!janusPromiseRef.current) {
            janusPromiseRef.current = new Promise((resolve, reject) => {
                Janus.init({
                    debug: janusDebugMode ? ['trace', 'debug', 'log', 'warn', 'error'] : ['warn', 'error'],
                    dependencies: Janus.useDefaultDependencies(),
                    callback: () => {
                        const instance = new Janus({
                            server: config.janus.websocket,
                            iceServers: config.iceServers,
                            success: () => {
                                janusRef.current = instance;
                                resolve(instance);
                            },
                            error: (cause: unknown) => {
                                janusPromiseRef.current = null;
                                reject(typeof cause === 'string' ? new Error(cause) : cause);
                            },
                            destroyed: () => {
                                janusRef.current = null;
                                janusPromiseRef.current = null;
                                publisherRef.current = null;
                                remoteFeedsRef.current = {};
                                cleanupLocalMedia();
                                setActiveRoomId(null);
                                setRemoteStreams([]);
                            },
                        });
                    },
                });
            });
            janusPromiseRef.current.catch(() => {
                janusPromiseRef.current = null;
            });
        }
        return janusPromiseRef.current;
    }, [cleanupLocalMedia, config, janusDebugMode]);

    const publishOwnFeed = useCallback(() => {
        const plugin = publisherRef.current;
        if (!plugin) {
            return;
        }
        plugin.createOffer({
            media: { audioRecv: false, videoRecv: false, audioSend: true, videoSend: true },
            stream: localStreamRef.current ?? undefined,
            success: (jsep: unknown) => {
                plugin.send({
                    message: { request: 'publish', audio: true, video: true },
                    jsep,
                });
            },
            error: (err: unknown) => {
                reportStatus(`WebRTC publish error: ${(err as Error).message ?? err}`);
            },
        });
    }, []);

    const handleNewRemoteFeed = useCallback(
        (feedId: number, roomId: number, display?: string) => {
            const janus = janusRef.current;
            if (!janus || remoteFeedsRef.current[feedId]) {
                return;
            }
            logDebug(`Attaching subscriber for feed ${feedId}${display ? ` (${display})` : ''}`);
            janus.attach({
                plugin: 'janus.plugin.videoroom',
                opaqueId: `react-subscriber-${feedId}`,
                success: (pluginHandle: any) => {
                    remoteFeedsRef.current[feedId] = pluginHandle;
                    const subscribe: Record<string, unknown> = {
                        request: 'join',
                        room: roomId,
                        ptype: 'subscriber',
                        feed: feedId,
                    };
                    if (privateIdRef.current) {
                        subscribe.private_id = privateIdRef.current;
                    }
                    pluginHandle.send({ message: subscribe });
                },
                error: (reason: unknown) => {
                    reportStatus(`Remote feed error: ${(reason as Error).message ?? reason}`);
                },
                onmessage: (msg: any, jsep: any) => {
                    if (jsep) {
                        remoteFeedsRef.current[feedId]?.createAnswer({
                            jsep,
                            media: { audioSend: false, videoSend: false, audioRecv: true, videoRecv: true },
                            success: (answerJsep: any) => {
                                logDebug(`Answer created for feed ${feedId}, starting stream`);
                                remoteFeedsRef.current[feedId]?.send({
                                    message: { request: 'start', room: roomId },
                                    jsep: answerJsep,
                                });
                            },
                            error: (err: unknown) => reportStatus(`Remote SDP issue: ${(err as Error).message ?? err}`),
                        });
                    }
                    const event = msg?.videoroom;
                    if (event === 'event' && msg.unpublished) {
                        logDebug(`Feed ${feedId} unpublished, removing`);
                        removeRemoteStream(feedId);
                        if (remoteFeedsRef.current[feedId]) {
                            remoteFeedsRef.current[feedId].detach();
                            delete remoteFeedsRef.current[feedId];
                        }
                    }
                },
                onremotestream: (stream: MediaStream) => {
                    logDebug(`Remote stream ready for feed ${feedId}`);
                    addRemoteStream(feedId, stream, display);
                },
                onremotetrack: (track: MediaStreamTrack, on: boolean) => {
                    if (!on) {
                        logDebug(`Remote track removed for feed ${feedId}`);
                        if (remoteTrackStreamsRef.current[feedId]) {
                            const stream = remoteTrackStreamsRef.current[feedId];
                            stream.removeTrack(track);
                            if (stream.getTracks().length === 0) {
                                delete remoteTrackStreamsRef.current[feedId];
                                removeRemoteStream(feedId);
                            }
                        }
                        return;
                    }
                    if (track.kind === 'video' || track.kind === 'audio') {
                        const existing = remoteTrackStreamsRef.current[feedId] ?? new MediaStream();
                        if (!existing.getTracks().includes(track)) {
                            existing.addTrack(track);
                        }
                        remoteTrackStreamsRef.current[feedId] = existing;
                        logDebug(`Remote ${track.kind} track ready for feed ${feedId}`);
                        addRemoteStream(feedId, existing, display);
                    }
                },
                oncleanup: () => {
                    logDebug(`Remote feed ${feedId} cleanup`);
                    removeRemoteStream(feedId);
                    delete remoteFeedsRef.current[feedId];
                    delete remoteTrackStreamsRef.current[feedId];
                },
            });
        },
        [addRemoteStream, logDebug, removeRemoteStream],
    );

    const handlePublisherMessage = useCallback(
        (msg: any, jsep: any, roomId: number) => {
            const event = msg?.videoroom;
            if (event === 'joined') {
                ownFeedIdRef.current = msg.id ?? null;
                privateIdRef.current = msg.private_id ?? null;
                reportStatus(`Joined room ${roomId}. Publishing local media…`);
                publishOwnFeed();
                const publishers = (msg.publishers ?? []).filter((pub: any) => pub.id !== ownFeedIdRef.current);
                if (publishers.length > 0) {
                    reportStatus(`Found ${publishers.length} active participant(s). Subscribing…`);
                }
                publishers.forEach((pub: any) => handleNewRemoteFeed(pub.id, roomId, pub.display));
            } else if (event === 'event') {
                const publishers = (msg.publishers ?? []).filter((pub: any) => pub.id !== ownFeedIdRef.current);
                if (publishers.length > 0) {
                    reportStatus(`New participant event for ${publishers.length} feed(s). Subscribing…`);
                }
                publishers.forEach((pub: any) => handleNewRemoteFeed(pub.id, roomId, pub.display));
                if (msg.leaving) {
                    const feedId = msg.leaving;
                    if (remoteFeedsRef.current[feedId]) {
                        remoteFeedsRef.current[feedId].detach();
                        delete remoteFeedsRef.current[feedId];
                        removeRemoteStream(feedId);
                    }
                }
            }
            if (jsep && publisherRef.current) {
                publisherRef.current.handleRemoteJsep({ jsep });
            }
        },
        [handleNewRemoteFeed, publishOwnFeed, removeRemoteStream, reportStatus],
    );

    const attachPublisher = useCallback(
        async (roomId: number, friendlyName: string) => {
            const janus = await getJanusInstance();
            return new Promise<void>((resolve, reject) => {
                janus.attach({
                    plugin: 'janus.plugin.videoroom',
                    opaqueId: `react-publisher-${Janus.randomString(6)}`,
                    success: (pluginHandle: any) => {
                        publisherRef.current = pluginHandle;
                        const register = {
                            request: 'join',
                            room: roomId,
                            ptype: 'publisher',
                            display: friendlyName,
                        };
                        pluginHandle.send({ message: register });
                        resolve();
                    },
                    error: (cause: unknown) => reject(new Error((cause as Error).message ?? 'Unable to attach publisher')),
                    onlocalstream: (stream: MediaStream) => {
                        localStreamRef.current = stream;
                        setLocalPreviewStream(stream);
                        if (localVideoRef.current) {
                            localVideoRef.current.srcObject = stream;
                        }
                    },
                    onmessage: (msg: any, jsep: any) => handlePublisherMessage(msg, jsep, roomId),
                    oncleanup: () => {
                        cleanupLocalMedia();
                    },
                });
            });
        },
        [cleanupLocalMedia, getJanusInstance, handlePublisherMessage],
    );

    const leaveRoom = useCallback(
        ({ silent }: { silent?: boolean } = {}) => {
            if (!silent) {
                reportStatus('Leaving room…');
            }
            if (publisherRef.current) {
                publisherRef.current.send({ message: { request: 'leave' } });
                publisherRef.current.hangup();
                publisherRef.current.detach();
                publisherRef.current = null;
            }
            Object.values(remoteFeedsRef.current).forEach((feed) => feed.detach());
            remoteFeedsRef.current = {};
            cleanupLocalMedia();
            setRemoteStreams([]);
            setActiveRoomId(null);
            setJoinedRoomName(null);
            ownFeedIdRef.current = null;
            privateIdRef.current = null;
            setView('lobby');
        },
        [cleanupLocalMedia, reportStatus],
    );

    const ensureRoom = useCallback(
        async (targetName: string) => {
            const payload = await fetchJson<{ room: any }>(`/api/rooms`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: targetName }),
            });
            const details = payload.room;
            const roomId = details.roomId ?? details.room;
            const summary: RoomSummary = {
                roomId,
                description: details.description ?? targetName,
                maxPublishers: details.max_publishers ?? details.maxPublishers,
            };
            return summary;
        },
        [],
    );

    const handleRoomSubmit = useCallback(
        async (event: FormEvent) => {
            event.preventDefault();
            if (!roomName.trim()) {
                reportStatus('Provide a room name first');
                return;
            }
            setLoading(true);
            reportStatus('Contacting Janus…');
            try {
                if (!localStreamRef.current) {
                    const constraints: MediaStreamConstraints = { audio: true, video: { facingMode: 'user' } };
                    localStreamRef.current = await navigator.mediaDevices.getUserMedia(constraints);
                    setLocalPreviewStream(localStreamRef.current);
                    if (localVideoRef.current) {
                        localVideoRef.current.srcObject = localStreamRef.current;
                    }
                }
                const trimmedName = roomName.trim();
                const summary = await ensureRoom(trimmedName);
                reportStatus(`Room "${summary.description}" is ready. Joining now…`);
                setActiveRoomId(summary.roomId);
                setJoinedRoomName(summary.description ?? trimmedName);
                setJoining(true);
                await attachPublisher(summary.roomId, displayName.trim() || 'guest');
                setView('room');
            } catch (err) {
                reportStatus((err as Error).message);
                setActiveRoomId(null);
            } finally {
                setJoining(false);
                setLoading(false);
            }
        },
        [attachPublisher, displayName, ensureRoom, reportStatus, roomName],
    );

    useEffect(
        () => () => {
            leaveRoom({ silent: true });
            if (janusRef.current) {
                janusRef.current.destroy();
            }
        },
        [leaveRoom],
    );

    useEffect(() => {
        if (typeof window !== 'undefined') {
            window.localStorage.setItem('janus-debug', janusDebugMode ? 'true' : 'false');
        }
        if (janusRef.current) {
            janusRef.current.destroy();
        }
    }, [janusDebugMode]);

    const isReadyToPublish = useMemo(() => Boolean(activeRoomId && publisherRef.current), [activeRoomId]);
    const participantList = useMemo(() => {
        const base = activeRoomId
            ? [
                {
                    id: 'local',
                    display: displayName || 'You',
                },
            ]
            : [];
        const remote = remoteStreams.map((feed) => ({
            id: `remote-${feed.id}`,
            display: feed.display ?? `Participant ${feed.id}`,
        }));
        return [...base, ...remote];
    }, [activeRoomId, displayName, remoteStreams]);
    const debugSteps = useMemo(
        () => [
            '1. Validate backend → run curl http://localhost:4000/health to ensure the Node API is up.',
            '2. Validate Janus → run docker logs janus --tail 50 and look for transport/plugin errors.',
            '3. Use the browser devtools console for [janus-ui] lines and WebRTC errors.',
            '4. If media fails, reload tabs and ensure camera/mic permissions are granted.',
        ],
        [],
    );

    useEffect(() => {
        if (localVideoRef.current && localPreviewStream) {
            localVideoRef.current.srcObject = localPreviewStream;
            void localVideoRef.current.play().catch(() => { });
            
            // Initialize frame processor for local video
            if (mlConfig.enableLocalProcessing && localVideoRef.current) {
                const sessionId = `local-${displayName}`;
                if (!localFrameProcessorRef.current) {
                    localFrameProcessorRef.current = new FrameProcessor(sessionId, {
                        onResult: (result) => {
                            setLocalConcentration(result);
                        },
                        onError: (error) => {
                            console.error('[Local Frame Processor] Error:', error);
                        },
                    });
                }
                
                // Start processing when video is ready
                const video = localVideoRef.current;
                const startProcessing = () => {
                    if (video.readyState >= 2 && localFrameProcessorRef.current) {
                        localFrameProcessorRef.current.start(video);
                    }
                };
                
                video.addEventListener('loadedmetadata', startProcessing);
                if (video.readyState >= 2) {
                    startProcessing();
                }
                
                return () => {
                    video.removeEventListener('loadedmetadata', startProcessing);
                    if (localFrameProcessorRef.current) {
                        localFrameProcessorRef.current.destroy();
                        localFrameProcessorRef.current = null;
                    }
                };
            }
        }
    }, [localPreviewStream, view, displayName]);

    return (
        <div className="app meeting-shell">
            {view === 'lobby' && (
                <div className="lobby-screen">
                    <div className="lobby-card">
                        <h1>Join a meeting</h1>
                        <form onSubmit={handleRoomSubmit} className="room-form">
                            <label htmlFor="room-name">Room name</label>
                            <input
                                id="room-name"
                                value={roomName}
                                onChange={(event) => setRoomName(event.target.value)}
                                placeholder="e.g. monday-standup"
                                autoComplete="off"
                            />
                            <label htmlFor="display-name">Display name</label>
                            <input
                                id="display-name"
                                value={displayName}
                                onChange={(event) => setDisplayName(event.target.value)}
                                placeholder="How you appear to others"
                                autoComplete="off"
                            />
                            <button type="submit" disabled={loading || joining}>
                                {loading || joining ? 'Joining…' : 'Join'}
                            </button>
                        </form>
                        {status && <p className="status">{status}</p>}
                    </div>
                </div>
            )}

            {view === 'room' && (
                <>
                    <header className="meeting-bar">
                        <div>
                            <p className="meeting-room-label">Room</p>
                            <h2>{joinedRoomName ?? roomName}</h2>
                        </div>
                        <div className="meeting-actions">
                            <button type="button" className="secondary" onClick={() => leaveRoom()}>
                                Leave
                            </button>
                        </div>
                    </header>

                    <div className="meeting-layout">
                        <section className="video-grid">
                            {localPreviewStream ? (
                                <div className="video-card self">
                                    <video ref={localVideoRef} autoPlay playsInline muted />
                                    <span className="badge">{displayName || 'You'}</span>
                                    {mlConfig.enableLocalProcessing && (
                                        <ConcentrationOverlay 
                                            result={localConcentration}
                                            showScore={mlConfig.showConcentrationScore}
                                            showStatus={mlConfig.showStatus}
                                            showGaze={mlConfig.showGazeDirection}
                                        />
                                    )}
                                </div>
                            ) : (
                                <div className="video-card self placeholder">
                                    <span className="badge">{displayName || 'You'}</span>
                                    <p className="waiting">Camera preview will appear once media is granted.</p>
                                </div>
                            )}
                            {remoteStreams.map((feed) => (
                                <RemoteVideoTile
                                    key={feed.id}
                                    stream={feed.stream}
                                    label={feed.display ?? `Participant ${feed.id}`}
                                    feedId={feed.id}
                                    enableProcessing={mlConfig.enableRemoteProcessing}
                                />
                            ))}
                            {!isReadyToPublish && <p className="waiting">Waiting for media…</p>}
                        </section>

                        <aside className="participants-panel">
                            <h3>Participants ({participantList.length})</h3>
                            <ul>
                                {participantList.map((person) => (
                                    <li key={person.id}>{person.display}</li>
                                ))}
                            </ul>
                            {status && <p className="status">{status}</p>}
                            <div className="ml-config-section">
                                <button 
                                    type="button" 
                                    className="ml-config-toggle"
                                    onClick={() => setMlConfigOpen(!mlConfigOpen)}
                                >
                                    {mlConfigOpen ? '▼' : '▶'} ML Settings
                                </button>
                                {mlConfigOpen && (
                                    <div className="ml-config-panel">
                                        <label>
                                            Frame Rate (FPS):
                                            <input
                                                type="number"
                                                min="1"
                                                max="30"
                                                value={currentMLConfig.frameRate}
                                                onChange={(e) => {
                                                    const val = parseInt(e.target.value, 10);
                                                    setCurrentMLConfig({ ...currentMLConfig, frameRate: val });
                                                    updateMLConfig({ frameRate: val });
                                                    localFrameProcessorRef.current?.updateConfig({ frameRate: val });
                                                }}
                                            />
                                        </label>
                                        <label>
                                            JPEG Quality:
                                            <input
                                                type="number"
                                                min="10"
                                                max="100"
                                                value={currentMLConfig.jpegQuality}
                                                onChange={(e) => {
                                                    const val = parseInt(e.target.value, 10);
                                                    setCurrentMLConfig({ ...currentMLConfig, jpegQuality: val });
                                                    updateMLConfig({ jpegQuality: val });
                                                }}
                                            />
                                        </label>
                                        <label>
                                            Max Width:
                                            <input
                                                type="number"
                                                min="160"
                                                max="1920"
                                                step="160"
                                                value={currentMLConfig.maxWidth}
                                                onChange={(e) => {
                                                    const val = parseInt(e.target.value, 10);
                                                    setCurrentMLConfig({ ...currentMLConfig, maxWidth: val });
                                                    updateMLConfig({ maxWidth: val });
                                                }}
                                            />
                                        </label>
                                        <label>
                                            Max Height:
                                            <input
                                                type="number"
                                                min="120"
                                                max="1080"
                                                step="120"
                                                value={currentMLConfig.maxHeight}
                                                onChange={(e) => {
                                                    const val = parseInt(e.target.value, 10);
                                                    setCurrentMLConfig({ ...currentMLConfig, maxHeight: val });
                                                    updateMLConfig({ maxHeight: val });
                                                }}
                                            />
                                        </label>
                                        <label>
                                            <input
                                                type="checkbox"
                                                checked={currentMLConfig.enableLocalProcessing}
                                                onChange={(e) => {
                                                    setCurrentMLConfig({ ...currentMLConfig, enableLocalProcessing: e.target.checked });
                                                    updateMLConfig({ enableLocalProcessing: e.target.checked });
                                                }}
                                            />
                                            Process Local Video
                                        </label>
                                        <label>
                                            <input
                                                type="checkbox"
                                                checked={currentMLConfig.enableRemoteProcessing}
                                                onChange={(e) => {
                                                    setCurrentMLConfig({ ...currentMLConfig, enableRemoteProcessing: e.target.checked });
                                                    updateMLConfig({ enableRemoteProcessing: e.target.checked });
                                                }}
                                            />
                                            Process Remote Videos
                                        </label>
                                        <label>
                                            <input
                                                type="checkbox"
                                                checked={currentMLConfig.showConcentrationScore}
                                                onChange={(e) => {
                                                    setCurrentMLConfig({ ...currentMLConfig, showConcentrationScore: e.target.checked });
                                                    updateMLConfig({ showConcentrationScore: e.target.checked });
                                                }}
                                            />
                                            Show Concentration Score
                                        </label>
                                        <label>
                                            <input
                                                type="checkbox"
                                                checked={currentMLConfig.showStatus}
                                                onChange={(e) => {
                                                    setCurrentMLConfig({ ...currentMLConfig, showStatus: e.target.checked });
                                                    updateMLConfig({ showStatus: e.target.checked });
                                                }}
                                            />
                                            Show Status
                                        </label>
                                    </div>
                                )}
                            </div>
                            {config && (
                                <div className="gateway-pill">
                                    <span>Gateway</span>
                                    <code>{config.janus.websocket}</code>
                                </div>
                            )}
                        </aside>
                    </div>
                </>
            )}

            <section className="panel debug-panel">
                <h2>Debug helpers</h2>
                <div className="debug-toggle">
                    <label htmlFor="janus-debug">
                        <input
                            id="janus-debug"
                            type="checkbox"
                            checked={janusDebugMode}
                            onChange={(event) => setJanusDebugMode(event.target.checked)}
                        />
                        Verbose Janus debug (destroys current session)
                    </label>
                    <p>
                        When enabled, the Janus SDK prints trace/debug logs to the console. Toggle it on and rejoin if you need
                        deeper insight.
                    </p>
                </div>
                <ol className="debug-steps">
                    {debugSteps.map((step) => (
                        <li key={step}>{step}</li>
                    ))}
                </ol>
                <div className="debug-log">
                    <strong>Recent events</strong>
                    {debugLines.length === 0 ? (
                        <p>No events captured yet.</p>
                    ) : (
                        <ul className="debug-log-list">
                            {debugLines.map((line) => (
                                <li key={line}>{line}</li>
                            ))}
                        </ul>
                    )}
                </div>
            </section>
        </div>
    );
}

function RemoteVideoTile({ 
    stream, 
    label, 
    feedId,
    enableProcessing,
}: { 
    stream: MediaStream; 
    label: string;
    feedId: number;
    enableProcessing: boolean;
}) {
    const videoRef = useRef<HTMLVideoElement | null>(null);
    const frameProcessorRef = useRef<FrameProcessor | null>(null);
    const [concentration, setConcentration] = useState<FrameProcessingResult | null>(null);
    const [currentConfig, setCurrentConfig] = useState(mlConfig);

    useEffect(() => {
        if (videoRef.current) {
            videoRef.current.srcObject = stream;
        }
    }, [stream]);

    useEffect(() => {
        // Update config when mlConfig changes
        setCurrentConfig(mlConfig);
    }, []);

    useEffect(() => {
        if (enableProcessing && videoRef.current) {
            const sessionId = `remote-${feedId}`;
            if (!frameProcessorRef.current) {
                frameProcessorRef.current = new FrameProcessor(sessionId, {
                    onResult: (result) => {
                        setConcentration(result);
                    },
                    onError: (error) => {
                        console.error(`[Remote Frame Processor ${feedId}] Error:`, error);
                    },
                });
            }
            
            const video = videoRef.current;
            const startProcessing = () => {
                if (video.readyState >= 2 && frameProcessorRef.current) {
                    frameProcessorRef.current.start(video);
                }
            };
            
            video.addEventListener('loadedmetadata', startProcessing);
            if (video.readyState >= 2) {
                startProcessing();
            }
            
            return () => {
                video.removeEventListener('loadedmetadata', startProcessing);
                if (frameProcessorRef.current) {
                    frameProcessorRef.current.destroy();
                    frameProcessorRef.current = null;
                }
            };
        }
    }, [feedId, enableProcessing]);

    return (
        <div className="video-card">
            <video ref={videoRef} autoPlay playsInline />
            <span className="badge">{label}</span>
            {enableProcessing && (
                <ConcentrationOverlay 
                    result={concentration}
                    showScore={currentConfig.showConcentrationScore}
                    showStatus={currentConfig.showStatus}
                    showGaze={currentConfig.showGazeDirection}
                />
            )}
        </div>
    );
}

