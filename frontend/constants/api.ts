export const API_BASE_URL = process.env.EXPO_PUBLIC_API_URL || 'http://localhost:8000';
// For testing with physical device, change to: 'http://192.168.X.X:8000'
export const API_PREDICT_ENDPOINT = `${API_BASE_URL}/predict`;

export const AGREEMENT_THRESHOLDS = {
    STRONG: 0.35,
    MODERATE_LOW: 0.20,
};

export const IMAGE_CONSTRAINTS = {
    MAX_SIZE_MB: 10,
    SUPPORTED_FORMATS: ['image/jpeg', 'image/png'],
};

export const XAI_METHODS = ['gradcam', 'gradcampp', 'layercam', 'shap', 'hirescam', 'consensus'] as const;
