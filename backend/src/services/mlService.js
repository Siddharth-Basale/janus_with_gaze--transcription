/**
 * Service to communicate with Python ML service
 */
import axios from 'axios';
import { config } from '../config/env.js';

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000';

export async function processFrame(sessionId, frameData) {
    try {
        const response = await axios.post(`${ML_SERVICE_URL}/process-frame`, {
            session_id: sessionId,
            frame_data: frameData,
            timestamp: Date.now(),
        }, {
            timeout: 5000, // 5 second timeout
        });
        return response.data;
    } catch (error) {
        if (error.response) {
            throw new Error(`ML service error: ${error.response.status} - ${error.response.data?.detail || error.message}`);
        } else if (error.request) {
            throw new Error('ML service unreachable - is it running?');
        } else {
            throw new Error(`ML service request failed: ${error.message}`);
        }
    }
}

export async function resetSession(sessionId) {
    try {
        const response = await axios.post(`${ML_SERVICE_URL}/reset-session/${sessionId}`);
        return response.data;
    } catch (error) {
        console.error('Failed to reset ML session:', error.message);
        throw error;
    }
}

export async function checkMLServiceHealth() {
    try {
        const response = await axios.get(`${ML_SERVICE_URL}/health`, { timeout: 2000 });
        return response.data;
    } catch (error) {
        return { status: 'error', message: error.message };
    }
}

