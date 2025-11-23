/**
 * Configuration for ML frame processing
 * Adjust these values to experiment with different settings
 */

export interface MLConfig {
  // Frame extraction settings
  frameRate: number; // Frames per second to extract (5-15 recommended)
  jpegQuality: number; // JPEG compression quality (0-100)
  maxWidth: number; // Maximum frame width (smaller = faster)
  maxHeight: number; // Maximum frame height
  
  // Processing settings
  enableLocalProcessing: boolean; // Process local video
  enableRemoteProcessing: boolean; // Process remote videos
  skipFramesIfBusy: boolean; // Skip frames if previous request pending
  
  // Display settings
  showConcentrationScore: boolean;
  showStatus: boolean;
  showGazeDirection: boolean;
}

export const defaultMLConfig: MLConfig = {
  // Frame extraction - lower FPS = less bandwidth, still feels real-time
  frameRate: 8, // Process 8 frames per second (good balance)
  jpegQuality: 70, // 70% quality (good compression/quality balance)
  maxWidth: 640, // Resize to 640px width
  maxHeight: 480, // Resize to 480px height
  
  // Processing
  enableLocalProcessing: true, // Process your own video
  enableRemoteProcessing: true, // Process remote participants
  skipFramesIfBusy: true, // Skip if previous frame still processing
  
  // Display
  showConcentrationScore: true,
  showStatus: true,
  showGazeDirection: false, // Can be enabled for debugging
};

// Export config for easy access
export let mlConfig: MLConfig = { ...defaultMLConfig };

// Function to update config
export function updateMLConfig(updates: Partial<MLConfig>) {
  mlConfig = { ...mlConfig, ...updates };
}

// Function to reset to defaults
export function resetMLConfig() {
  mlConfig = { ...defaultMLConfig };
}

