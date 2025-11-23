import { useMemo } from 'react';
import type { FrameProcessingResult } from '../services/frameProcessor';

interface ConcentrationOverlayProps {
    result: FrameProcessingResult | null;
    showScore?: boolean;
    showStatus?: boolean;
    showGaze?: boolean;
}

export function ConcentrationOverlay({
    result,
    showScore = true,
    showStatus = true,
    showGaze = false
}: ConcentrationOverlayProps) {
    const scoreColor = useMemo(() => {
        if (!result) return '#888';
        const score = result.smooth_score;
        if (score >= 60) return '#0f0'; // Green
        if (score >= 40) return '#0af'; // Blue
        return '#f80'; // Orange
    }, [result]);

    const statusColor = useMemo(() => {
        if (!result) return '#888';
        const status = result.status;
        if (status === 'CONCENTRATED' || status === 'OCCLUDED') return '#0f0';
        if (status === 'BLINK') return '#ff0';
        if (status === 'NO FACE' || status === 'NO_FACE') return '#f00';
        if (status === 'NOISY') return '#f00';
        if (status === 'DISTRACTED') return '#f80';
        return '#f80';
    }, [result]);

    if (!result) {
        return (
            <div className="concentration-overlay">
                <div className="concentration-status">Calibrating...</div>
            </div>
        );
    }

    return (
        <div className="concentration-overlay">
            {showScore && (
                <div className="concentration-score" style={{ color: scoreColor }}>
                    {result.smooth_score}%
                </div>
            )}
            {showStatus && (
                <div className="concentration-status" style={{ color: statusColor }}>
                    {result.status}
                </div>
            )}
            {showGaze && result.gaze_direction !== 'UNKNOWN' && (
                <div className="concentration-gaze">
                    Gaze: {result.gaze_direction}
                </div>
            )}
            {result.eyes_closed && (
                <div className="concentration-warning">⚠ Eyes Closed</div>
            )}
            {!result.calibrated && (
                <div className="concentration-calibrating">Calibrating...</div>
            )}
        </div>
    );
}

