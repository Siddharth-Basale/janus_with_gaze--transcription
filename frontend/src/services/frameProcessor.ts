/**
 * Frame extraction and processing service
 * Extracts frames from video elements and sends to ML service
 */

import { mlConfig } from '../config/mlConfig';

export interface FrameProcessingResult {
  concentration: number;
  status: string;
  gaze_direction: string;
  blink_detected: boolean;
  eyes_closed: boolean;
  calibrated: boolean;
  smooth_score: number;
}

export interface FrameProcessorCallbacks {
  onResult?: (result: FrameProcessingResult) => void;
  onError?: (error: Error) => void;
}

export class FrameProcessor {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private sessionId: string;
  private isProcessing = false;
  private lastFrameTime = 0;
  private frameInterval: number;
  private callbacks: FrameProcessorCallbacks;
  private ws: WebSocket | null = null;
  private pendingRequest = false;

  constructor(sessionId: string, callbacks: FrameProcessorCallbacks = {}) {
    this.sessionId = sessionId;
    this.callbacks = callbacks;
    this.canvas = document.createElement('canvas');
    this.ctx = this.canvas.getContext('2d')!;
    this.frameInterval = 1000 / mlConfig.frameRate; // ms between frames
    
    // Connect to WebSocket
    this.connectWebSocket();
  }

  private connectWebSocket() {
    const wsUrl = `ws://${window.location.hostname}:4000/ws/ml`;
    this.ws = new WebSocket(wsUrl);
    
    this.ws.onopen = () => {
      console.log('[FrameProcessor] WebSocket connected');
    };
    
    this.ws.onmessage = (event) => {
      try {
        const result: FrameProcessingResult = JSON.parse(event.data);
        this.pendingRequest = false;
        this.callbacks.onResult?.(result);
      } catch (error) {
        console.error('[FrameProcessor] Failed to parse result:', error);
        this.callbacks.onError?.(error as Error);
      }
    };
    
    this.ws.onerror = (error) => {
      console.error('[FrameProcessor] WebSocket error:', error);
      this.callbacks.onError?.(new Error('WebSocket connection failed'));
    };
    
    this.ws.onclose = () => {
      console.log('[FrameProcessor] WebSocket closed, reconnecting...');
      setTimeout(() => this.connectWebSocket(), 3000);
    };
  }

  /**
   * Extract frame from video element and send for processing
   */
  async processFrame(videoElement: HTMLVideoElement): Promise<void> {
    if (!this.isProcessing) return;
    if (!videoElement || videoElement.readyState < 2) return; // Not ready
    
    const now = Date.now();
    if (now - this.lastFrameTime < this.frameInterval) return; // Too soon
    
    // Skip if previous request pending and skipFramesIfBusy is enabled
    if (mlConfig.skipFramesIfBusy && this.pendingRequest) return;
    
    this.lastFrameTime = now;
    
    try {
      // Set canvas size (resize if needed)
      const videoWidth = videoElement.videoWidth;
      const videoHeight = videoElement.videoHeight;
      
      if (videoWidth === 0 || videoHeight === 0) return;
      
      // Calculate resize dimensions
      let width = videoWidth;
      let height = videoHeight;
      
      if (width > mlConfig.maxWidth || height > mlConfig.maxHeight) {
        const scale = Math.min(mlConfig.maxWidth / width, mlConfig.maxHeight / height);
        width = Math.floor(width * scale);
        height = Math.floor(height * scale);
      }
      
      this.canvas.width = width;
      this.canvas.height = height;
      
      // Draw video frame to canvas
      this.ctx.drawImage(videoElement, 0, 0, width, height);
      
      // Convert to JPEG
      const jpegData = this.canvas.toDataURL('image/jpeg', mlConfig.jpegQuality / 100);
      const base64Data = jpegData.split(',')[1]; // Remove data:image/jpeg;base64, prefix
      
      // Send via WebSocket if connected, otherwise fallback to HTTP
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.pendingRequest = true;
        this.ws.send(JSON.stringify({
          session_id: this.sessionId,
          frame_data: base64Data,
          timestamp: now,
        }));
      } else {
        // Fallback to HTTP
        await this.sendFrameHTTP(base64Data, now);
      }
    } catch (error) {
      console.error('[FrameProcessor] Frame processing error:', error);
      this.callbacks.onError?.(error as Error);
      this.pendingRequest = false;
    }
  }

  private async sendFrameHTTP(base64Data: string, timestamp: number): Promise<void> {
    try {
      const response = await fetch('http://localhost:8000/process-frame', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: this.sessionId,
          frame_data: base64Data,
          timestamp,
        }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const result: FrameProcessingResult = await response.json();
      this.pendingRequest = false;
      this.callbacks.onResult?.(result);
    } catch (error) {
      this.pendingRequest = false;
      this.callbacks.onError?.(error as Error);
    }
  }

  /**
   * Start processing frames from video element
   */
  start(videoElement: HTMLVideoElement): void {
    if (this.isProcessing) return;
    
    this.isProcessing = true;
    this.lastFrameTime = 0;
    
    // Use requestVideoFrameCallback if available (Chrome 94+)
    if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
      const processFrame = () => {
        if (this.isProcessing) {
          this.processFrame(videoElement);
          videoElement.requestVideoFrameCallback(processFrame);
        }
      };
      videoElement.requestVideoFrameCallback(processFrame);
    } else {
      // Fallback to setInterval
      const interval = setInterval(() => {
        if (!this.isProcessing) {
          clearInterval(interval);
          return;
        }
        this.processFrame(videoElement);
      }, this.frameInterval);
    }
  }

  /**
   * Stop processing frames
   */
  stop(): void {
    this.isProcessing = false;
  }

  /**
   * Cleanup resources
   */
  destroy(): void {
    this.stop();
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  /**
   * Update configuration
   */
  updateConfig(config: Partial<typeof mlConfig>): void {
    Object.assign(mlConfig, config);
    this.frameInterval = 1000 / mlConfig.frameRate;
  }
}

