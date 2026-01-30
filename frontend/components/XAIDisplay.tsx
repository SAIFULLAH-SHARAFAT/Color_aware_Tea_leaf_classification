import React from 'react';
import { View, Text, StyleSheet, Image, ScrollView } from 'react-native';
import { PredictionResponse } from '@/types/api';
import { AGREEMENT_THRESHOLDS } from '@/constants/api';

interface XAIOverlayProps {
    explanations: PredictionResponse['explanations'];
    method: 'gradcam' | 'gradcampp' | 'layercam' | 'shap' | 'hirescam' | 'consensus';
}

export function XAIOverlay({ explanations, method }: XAIOverlayProps) {
    let imageData = explanations[method];

    if (
        imageData &&
        !imageData.startsWith('file://') &&
        !imageData.startsWith('data:image')
    ) {
        imageData = `data:image/png;base64,${imageData}`;
    }

    return (
        <View style={styles.container}>
            <Text style={styles.methodTitle}>{method === 'consensus' ? 'CONSENSUS (MERGED)' : method.toUpperCase()}</Text>
            {imageData ? (
                <Image
                    source={{ uri: imageData }}
                    style={styles.image}
                    resizeMode="contain"
                />
            ) : (
                <View style={styles.placeholder}>
                    <Text style={styles.placeholderText}>No visualization available</Text>
                </View>
            )}
        </View>
    );
}

interface ReliabilityBadgeProps {
    score: number;
}

export function ReliabilityBadge({ score }: ReliabilityBadgeProps) {
    const getReliabilityLevel = (score: number): { label: string; color: string } => {
        if (score >= AGREEMENT_THRESHOLDS.STRONG) {
            return { label: 'High Confidence', color: '#10B981' };
        } else if (score >= AGREEMENT_THRESHOLDS.MODERATE_LOW) {
            return { label: 'Moderate', color: '#F59E0B' };
        } else {
            return { label: 'Low Confidence', color: '#DC2626' };
        }
    };

    const { label, color } = getReliabilityLevel(score);

    return (
        <View style={[styles.badge, { backgroundColor: color + '20', borderColor: color }]}>
            <Text style={[styles.badgeText, { color }]}>
                {label} ({score.toFixed(2)})
            </Text>
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        marginVertical: 12,
        backgroundColor: '#FFFFFF',
        borderRadius: 12,
        padding: 12,
    },
    methodTitle: {
        fontSize: 14,
        fontWeight: '600',
        color: '#0F172A',
        marginBottom: 8,
    },
    image: {
        width: '100%',
        height: 250,
        borderRadius: 8,
        backgroundColor: '#F1F5F9',
    },
    placeholder: {
        width: '100%',
        height: 250,
        borderRadius: 8,
        backgroundColor: '#F1F5F9',
        justifyContent: 'center',
        alignItems: 'center',
    },
    placeholderText: {
        color: '#94A3B8',
        fontSize: 14,
    },
    badge: {
        borderRadius: 8,
        paddingVertical: 8,
        paddingHorizontal: 12,
        borderWidth: 1,
        alignSelf: 'flex-start',
        marginVertical: 12,
    },
    badgeText: {
        fontSize: 13,
        fontWeight: '600',
    },
});
