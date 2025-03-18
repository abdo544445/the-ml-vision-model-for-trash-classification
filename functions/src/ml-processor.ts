import { spawn } from 'child_process';
import * as path from 'path';

interface Detection {
    class: string;
    confidence: number;
    bbox: number[];
}

export async function predictImage(imagePath: string): Promise<Detection[]> {
    return new Promise((resolve, reject) => {
        const pythonProcess = spawn('python3', [
            path.join(__dirname, 'ML-Processing-Module.py'),
            imagePath
        ]);

        let output = '';
        let errorOutput = '';

        pythonProcess.stdout.on('data', (data: Buffer) => {
            output += data.toString();
        });

        pythonProcess.stderr.on('data', (data: Buffer) => {
            errorOutput += data.toString();
        });

        pythonProcess.on('close', (code: number) => {
            if (code !== 0) {
                reject(new Error(`Python process failed: ${errorOutput}`));
                return;
            }

            try {
                const predictions = JSON.parse(output);
                resolve(predictions);
            } catch (error) {
                reject(new Error(`Failed to parse Python output: ${error instanceof Error ? error.message : 'Unknown error'}`));
            }
        });
    });
} 