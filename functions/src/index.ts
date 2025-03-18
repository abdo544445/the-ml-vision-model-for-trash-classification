import { onObjectFinalized } from 'firebase-functions/v2/storage';
import { onRequest } from 'firebase-functions/v2/https';
import * as admin from 'firebase-admin';
import { predictImage } from './ml-processor';

admin.initializeApp();
const cache = new Map();
const CACHE_EXPIRY = 3600000; // 1 hour in milliseconds

// Helper function to get image hash for caching
async function getImageHash(bucket: any, filePath: string): Promise<string> {
    const file = bucket.file(filePath);
    const [metadata] = await file.getMetadata();
    return metadata.md5Hash || '';
}

interface StorageEvent {
    data: {
        name: string;
        bucket: string;
    };
}

export const processImageUpload = onObjectFinalized({
    memory: '2GiB',  // Increased memory for ML processing
    timeoutSeconds: 540,  // Extended timeout
}, async (event: StorageEvent) => {
    try {
        const filePath = event.data.name;
        const bucket = admin.storage().bucket(event.data.bucket);
        
        // Check cache using image hash
        const imageHash = await getImageHash(bucket, filePath);
        if (cache.has(imageHash)) {
            const cachedResult = cache.get(imageHash);
            if (Date.now() - cachedResult.timestamp < CACHE_EXPIRY) {
                await savePrediction(filePath, cachedResult.predictions, event.data.bucket);
                return true;
            }
            cache.delete(imageHash);
        }

        // Download and process image
        const tempFilePath = `/tmp/${filePath.split('/').pop()}`;
        await bucket.file(filePath).download({ destination: tempFilePath });

        // ML Prediction
        const predictions = await predictImage(tempFilePath);

        // Cache the result
        cache.set(imageHash, {
            predictions,
            timestamp: Date.now()
        });

        await savePrediction(filePath, predictions, event.data.bucket);
        return true;
    } catch (error) {
        console.error('Error processing image:', error instanceof Error ? error.message : 'Unknown error');
        throw error;
    }
});

async function savePrediction(filePath: string, predictions: any, bucketName: string) {
    const db = admin.database();
    const ref = db.ref(`predictions/${filePath.replace(/\//g, '_')}`);
    
    await ref.set({
        timestamp: Date.now(),
        results: predictions,
        imageUrl: `gs://${bucketName}/${filePath}`
    });
}

// Endpoint for browser-based TF.js fallback
export const getBrowserDetection = onRequest({
    cors: true,
    maxInstances: 100
}, async (req: any, res: any) => {
    if (req.method !== 'POST') {
        res.status(405).send('Method Not Allowed');
        return;
    }

    try {
        // Return configuration for browser-based detection
        res.json({
            modelUrl: 'https://tfhub.dev/tensorflow/tfjs-model/ssd_mobilenet_v2/1/default/1',
            labels: ['person', 'car', 'dog', '...'],  // Your class labels
            threshold: 0.5
        });
    } catch (error) {
        console.error('Error in browser detection endpoint:', error instanceof Error ? error.message : 'Unknown error');
        res.status(500).json({ error: error instanceof Error ? error.message : 'Unknown error' });
    }
}); 