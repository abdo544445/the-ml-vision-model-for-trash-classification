import { predictImage } from './ml-processor';
import * as path from 'path';

async function runTest() {
    try {
        // Test with a sample image
        const testImagePath = path.join(__dirname, 'test-image.jpg');
        console.log('Testing ML processing with image:', testImagePath);
        
        const predictions = await predictImage(testImagePath);
        console.log('Predictions:', JSON.stringify(predictions, null, 2));
    } catch (error) {
        console.error('Test failed:', error);
        process.exit(1);
    }
}

runTest(); 