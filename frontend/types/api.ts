export type PredictionResponse = {
    prediction: {
        label: string;
        confidence: number;
    };
    agreement_score: number;
    quality_flags: {
        background: boolean;
        spread: boolean;
        border: boolean;
    };
    explanations: {
        gradcam: string; // base64 or image URL
        gradcampp: string;
        layercam: string;
        shap: string;
        hirescam: string;
        consensus: string;
    };
    final_status: 'accepted' | 'retake_required';
};

export type ApiErrorResponse = {
    detail: string;
};

export type ImagePickerResult = {
    uri: string;
    width: number;
    height: number;
    fileName?: string;
};
