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
        const score = result.gaze_on_screen_percentage || result.smooth_score || 0;
        if (score >= 80) return '#0f0'; // Green - looking at screen
        if (score >= 50) return '#0af'; // Blue - partially looking
        if (score >= 20) return '#ff0'; // Yellow - distracted
        return '#f80'; // Orange/Red - off screen
    }, [result]);

    const statusColor = useMemo(() => {
        if (!result) return '#888';
        const status = result.status;
        if (status === 'FOCUSED') return '#0f0'; // Green
        if (status === 'PARTIAL') return '#0af'; // Blue
        if (status === 'DISTRACTED') return '#ff0'; // Yellow
        if (status === 'OFF_SCREEN') return '#f80'; // Orange
        if (status === 'NO FACE' || status === 'NO_FACE') return '#f00'; // Red
        return '#888';
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
                    {result.gaze_on_screen_percentage || result.smooth_score || 0}%
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
            {!result.calibrated && (
                <div className="concentration-calibrating">Calibrating...</div>
            )}
        </div>
    );
}

