import React, { useEffect, useMemo } from 'react';
import {
    View,
    ScrollView,
    StyleSheet,
    Text,
    Image,
    TouchableOpacity,
    BackHandler,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { MaterialIcons } from '@expo/vector-icons';
import { PredictionResponse } from '@/types/api';
import { XAIOverlay, ReliabilityBadge } from '@/components/XAIDisplay';
import { ActionButton } from '@/components/ActionButton';
import { predictionStore } from '@/constants/store';

export default function ResultsScreen() {
    const router = useRouter();
    const params = useLocalSearchParams();

    const { data: prediction, imageUri } = predictionStore.getPrediction();
    const status = (params.status as 'accepted' | 'retake_required') || prediction?.final_status;

    const handleBack = () => {
        if (router.canGoBack()) {
            router.back();
        } else {
            predictionStore.clear();
            router.replace('/');
        }
    };

    useEffect(() => {
        const backHandler = BackHandler.addEventListener(
            'hardwareBackPress',
            () => {
                handleBack();
                return true;
            }
        );

        return () => backHandler.remove();
    }, [router]);

    if (!prediction || !status) {
        return (
            <SafeAreaView style={styles.safeArea}>
                <View style={styles.errorContainer}>
                    <Text style={styles.errorText}>No prediction data available</Text>
                    <ActionButton
                        icon="home"
                        label="Back to Home"
                        onPress={() => {
                            predictionStore.clear();
                            router.replace('/');
                        }}
                    />
                </View>
            </SafeAreaView>
        );
    }

    const isAccepted = status === 'accepted';
    const confidencePct = Math.round(prediction.prediction.confidence * 100);

    return (
        <SafeAreaView style={styles.safeArea}>
            <ScrollView contentContainerStyle={styles.container}>
                <View style={styles.header}>
                    <TouchableOpacity onPress={handleBack}>
                        <MaterialIcons name="arrow-back" size={24} color="#0F172A" />
                    </TouchableOpacity>
                    <Text style={styles.headerTitle}>
                        {isAccepted ? 'Prediction Result' : 'Image Quality Issue'}
                    </Text>
                    <View style={{ width: 24 }} />
                </View>

                {imageUri && (
                    <View style={styles.imageCard}>
                        <Text style={styles.cardLabel}>Original Image</Text>
                        <Image source={{ uri: imageUri }} style={styles.image} />
                    </View>
                )}

                {isAccepted ? (
                    <>
                        <View style={styles.resultCard}>
                            <View style={styles.resultHeader}>
                                <MaterialIcons name="check-circle" size={32} color="#10B981" />
                                <View style={styles.resultInfo}>
                                    <Text style={styles.resultLabel}>Disease Detected</Text>
                                    <Text style={styles.diseaseName}>{prediction.prediction.label}</Text>
                                </View>
                            </View>

                            <View style={styles.statsGrid}>
                                <View style={styles.statBox}>
                                    <Text style={styles.statLabel}>Confidence</Text>
                                    <Text style={styles.statValue}>{confidencePct}%</Text>
                                </View>
                                <View style={styles.statBox}>
                                    <Text style={styles.statLabel}>Agreement</Text>
                                    <Text style={styles.statValue}>
                                        {prediction.agreement_score.toFixed(2)}
                                    </Text>
                                </View>
                            </View>

                            <ReliabilityBadge score={prediction.agreement_score} />
                        </View>

                        <View style={styles.qualityCard}>
                            <Text style={styles.cardLabel}>Quality Assessment</Text>
                            <View style={styles.flagRow}>
                                <View style={styles.flagItem}>
                                    <MaterialIcons
                                        name={prediction.quality_flags.background ? 'close' : 'check'}
                                        size={20}
                                        color={prediction.quality_flags.background ? '#DC2626' : '#10B981'}
                                    />
                                    <Text style={styles.flagLabel}>Background</Text>
                                </View>
                                <View style={styles.flagItem}>
                                    <MaterialIcons
                                        name={prediction.quality_flags.spread ? 'close' : 'check'}
                                        size={20}
                                        color={prediction.quality_flags.spread ? '#DC2626' : '#10B981'}
                                    />
                                    <Text style={styles.flagLabel}>Focus</Text>
                                </View>
                                <View style={styles.flagItem}>
                                    <MaterialIcons
                                        name={prediction.quality_flags.border ? 'close' : 'check'}
                                        size={20}
                                        color={prediction.quality_flags.border ? '#DC2626' : '#10B981'}
                                    />
                                    <Text style={styles.flagLabel}>Edges</Text>
                                </View>
                            </View>
                        </View>

                        <View style={styles.xaiSection}>
                            <Text style={styles.cardLabel}>Explanation Methods</Text>
                            <Text style={styles.xaiSubtitle}>
                                How the model identified this disease
                            </Text>
                            <View style={{ marginBottom: 16 }}>
                                <Text style={styles.xaiSubtitle}>
                                    Consensus Result (Merged Logic)
                                </Text>
                                <Text style={{ fontSize: 12, color: '#64748B', marginBottom: 8 }}>
                                    Combines all 5 methods to show the most accurate disease location.
                                </Text>
                                <XAIOverlay explanations={prediction.explanations} method="consensus" />
                            </View>

                            <Text style={styles.xaiSubtitle}>Individual Methods</Text>

                            <XAIOverlay explanations={prediction.explanations} method="gradcam" />
                            <XAIOverlay explanations={prediction.explanations} method="gradcampp" />
                            <XAIOverlay explanations={prediction.explanations} method="layercam" />
                            <XAIOverlay explanations={prediction.explanations} method="shap" />
                            <XAIOverlay explanations={prediction.explanations} method="hirescam" />
                        </View>
                    </>
                ) : (
                    <>
                        <View style={styles.retakeCard}>
                            <View style={styles.retakeHeader}>
                                <MaterialIcons name="warning" size={40} color="#F59E0B" />
                                <Text style={styles.retakeTitle}>Please Retake Photo</Text>
                            </View>

                            <Text style={styles.retakeMessage}>
                                The image quality is insufficient for reliable disease classification.
                            </Text>

                            <View style={styles.suggestionBox}>
                                <Text style={styles.suggestionTitle}>Suggestions:</Text>
                                <Text style={styles.suggestionText}>
                                    - Capture the leaf closer and centered{'\n'}
                                    - Avoid hands, pots, and background{'\n'}
                                    - Ensure good lighting{'\n'}
                                    - Keep the leaf in focus
                                </Text>
                            </View>

                            <View style={styles.failedFlagsBox}>
                                <Text style={styles.failedFlagsTitle}>Issues detected:</Text>
                                {prediction.quality_flags.background && (
                                    <Text style={styles.failedFlag}>
                                        - Focus on background instead of leaf
                                    </Text>
                                )}
                                {prediction.quality_flags.spread && (
                                    <Text style={styles.failedFlag}>
                                        - Attention too scattered
                                    </Text>
                                )}
                                {prediction.quality_flags.border && (
                                    <Text style={styles.failedFlag}>
                                        - Attention concentrated at edges
                                    </Text>
                                )}
                                {prediction.agreement_score < 0.2 && (
                                    <Text style={styles.failedFlag}>
                                        - Low agreement between methods
                                    </Text>
                                )}
                            </View>
                        </View>
                    </>
                )}

                <View style={styles.footer}>
                    <ActionButton
                        icon={isAccepted ? 'home' : 'camera'}
                        label={isAccepted ? 'New Analysis' : 'Try Again'}
                        onPress={() => {
                            predictionStore.clear();
                            router.replace('/');
                        }}
                    />
                </View>
            </ScrollView>
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    safeArea: {
        flex: 1,
        backgroundColor: '#F8FAFC',
    },
    container: {
        padding: 16,
        gap: 16,
    },
    header: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        marginBottom: 8,
    },
    headerTitle: {
        fontSize: 18,
        fontWeight: '600',
        color: '#0F172A',
    },
    errorContainer: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        padding: 16,
    },
    errorText: {
        fontSize: 16,
        color: '#DC2626',
        marginBottom: 24,
    },
    imageCard: {
        backgroundColor: '#FFFFFF',
        borderRadius: 12,
        padding: 12,
        gap: 8,
    },
    cardLabel: {
        fontSize: 16,
        fontWeight: '600',
        color: '#0F172A',
    },
    image: {
        width: '100%',
        height: 240,
        borderRadius: 10,
        backgroundColor: '#E2E8F0',
    },
    resultCard: {
        backgroundColor: '#FFFFFF',
        borderRadius: 12,
        padding: 16,
        gap: 16,
    },
    resultHeader: {
        flexDirection: 'row',
        gap: 12,
    },
    resultInfo: {
        flex: 1,
        justifyContent: 'center',
    },
    resultLabel: {
        fontSize: 12,
        color: '#64748B',
        textTransform: 'uppercase',
    },
    diseaseName: {
        fontSize: 20,
        fontWeight: '700',
        color: '#0F172A',
    },
    statsGrid: {
        flexDirection: 'row',
        gap: 12,
    },
    statBox: {
        flex: 1,
        backgroundColor: '#F1F5F9',
        borderRadius: 10,
        padding: 12,
        alignItems: 'center',
    },
    statLabel: {
        fontSize: 12,
        color: '#64748B',
        marginBottom: 4,
    },
    statValue: {
        fontSize: 18,
        fontWeight: '700',
        color: '#0F172A',
    },
    qualityCard: {
        backgroundColor: '#FFFFFF',
        borderRadius: 12,
        padding: 16,
        gap: 12,
    },
    flagRow: {
        flexDirection: 'row',
        justifyContent: 'space-around',
    },
    flagItem: {
        alignItems: 'center',
        gap: 8,
    },
    flagLabel: {
        fontSize: 12,
        color: '#64748B',
    },
    xaiSection: {
        gap: 12,
    },
    xaiSubtitle: {
        fontSize: 13,
        color: '#64748B',
        marginBottom: 8,
    },
    retakeCard: {
        backgroundColor: '#FFFFFF',
        borderRadius: 12,
        padding: 16,
        gap: 16,
    },
    retakeHeader: {
        alignItems: 'center',
        gap: 12,
    },
    retakeTitle: {
        fontSize: 20,
        fontWeight: '700',
        color: '#F59E0B',
    },
    retakeMessage: {
        fontSize: 14,
        color: '#64748B',
        lineHeight: 20,
    },
    suggestionBox: {
        backgroundColor: '#FEF3C7',
        borderRadius: 10,
        padding: 12,
        gap: 8,
    },
    suggestionTitle: {
        fontSize: 13,
        fontWeight: '600',
        color: '#92400E',
    },
    suggestionText: {
        fontSize: 13,
        color: '#92400E',
        lineHeight: 18,
    },
    failedFlagsBox: {
        backgroundColor: '#FEE2E2',
        borderRadius: 10,
        padding: 12,
        gap: 8,
    },
    failedFlagsTitle: {
        fontSize: 13,
        fontWeight: '600',
        color: '#991B1B',
    },
    failedFlag: {
        fontSize: 13,
        color: '#991B1B',
    },
    footer: {
        marginTop: 8,
    },
});
