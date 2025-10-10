import {
	WorkflowEntrypoint,
	WorkflowEvent,
	WorkflowStep,
} from "cloudflare:workers";
import { Magika } from "magika";
// @ts-ignore - ssdeep.js doesn't have TypeScript definitions
import ssdeepjs from "ssdeep.js";
import { extractText } from "unpdf";

/**
 * File Processing System with Cloudflare Workers & Workflows
 *
 * Features:
 * - File upload via web UI
 * - SHA-256 and ssdeep hashing
 * - File type detection with Google Magika
 * - Image captioning with Moondream API
 * - PDF summarization with Workers AI
 * - R2 storage
 * - D1 metadata database with JSON support
 * - Search functionality
 */

// Environment bindings
type Env = {
	FILE_WORKFLOW: Workflow;
	FILE_BUCKET: R2Bucket;
	DB: D1Database;
	AI: Ai;
	ASSETS: Fetcher;
	MOONDREAM_API_KEY?: string; // Optional - for Moondream cloud API
};

// Workflow parameters
type FileProcessingParams = {
	fileId: string;
	filename: string;
	contentType: string;
	size: number;
	fileData: string; // Base64 encoded file data
};

// Workflow result type
type FileProcessingResult = {
	fileId: string;
	sha256Hash: string;
	ssdeepHash: string;
	fileType: string;
	metadata?: {
		caption?: string;
		summary?: string;
		rawText?: string;
	};
};

/**
 * File Processing Workflow
 * Handles multi-step file processing with automatic retries and state persistence
 */
export class FileProcessingWorkflow extends WorkflowEntrypoint<Env, FileProcessingParams> {
	async run(event: WorkflowEvent<FileProcessingParams>, step: WorkflowStep) {
		const { fileId, filename, contentType, size, fileData } = event.payload;

		// Helper to decode base64 file data (called fresh in each step to avoid detached buffer issues)
		const getFileBytes = () => Uint8Array.from(atob(fileData), c => c.charCodeAt(0));

		// Step 1: Hash the file (SHA-256 and ssdeep)
		const hashes = await step.do("hash-file", async () => {
			const uint8Array = getFileBytes();
			console.log(`Hashing file: ${filename}, size: ${size} bytes, actual size: ${uint8Array.length}`);

			// SHA-256 using Web Crypto API
			const sha256Buffer = await crypto.subtle.digest("SHA-256", uint8Array);
			const sha256Array = Array.from(new Uint8Array(sha256Buffer));
			const sha256Hash = sha256Array.map(b => b.toString(16).padStart(2, '0')).join('');

			// ssdeep fuzzy hash
			let ssdeepHash = "";
			try {
				ssdeepHash = ssdeepjs.digest(uint8Array);
				console.log(`ssdeep hash generated: ${ssdeepHash}`);
			} catch (e) {
				console.error("ssdeep hashing failed:", e);
				ssdeepHash = "ssdeep-failed";
			}

			console.log(`SHA-256: ${sha256Hash.substring(0, 16)}...`);
			return { sha256Hash, ssdeepHash };
		});

		// Step 2: Detect file type with Magika
		const fileType = await step.do("detect-file-type", async () => {
			try {
				console.log(`Detecting file type for: ${filename}`);
				const magika = await Magika.create();
				const bytes = getFileBytes();
				const prediction = await magika.identifyBytes(bytes);
				console.log(`Magika prediction:`, {
					label: prediction.label,
					score: prediction.score,
					contentType
				});
				return prediction.label || contentType || "unknown";
			} catch (e) {
				console.error("Magika detection failed:", e);
				return contentType || "unknown";
			}
		});

		// Step 3: Process based on file type
		let metadata: { caption?: string; summary?: string; rawText?: string } | undefined;

		if (fileType.includes("image") || fileType.includes("png") || fileType.includes("jpg") || fileType.includes("jpeg")) {
			// Image processing: Caption with Moondream
			metadata = await step.do("caption-image", async () => {
				try {
					// Check if Moondream API key is available
					if (!this.env.MOONDREAM_API_KEY) {
						console.warn("MOONDREAM_API_KEY not set - skipping image captioning");
						return { caption: "Image captioning skipped (no API key)" };
					}

					// Convert to base64 for API call (in chunks to avoid stack overflow)
					const uint8Array = getFileBytes();
					const chunkSize = 8192;
					let binaryString = '';
					for (let i = 0; i < uint8Array.length; i += chunkSize) {
						const chunk = uint8Array.subarray(i, i + chunkSize);
						binaryString += String.fromCharCode.apply(null, Array.from(chunk));
					}
					const base64Image = btoa(binaryString);

					// Call Moondream Cloud API with correct headers and field names
					const response = await fetch("https://api.moondream.ai/v1/caption", {
						method: "POST",
						headers: {
							"Content-Type": "application/json",
							"X-Moondream-Auth": this.env.MOONDREAM_API_KEY
						},
						body: JSON.stringify({
							image_url: `data:image/jpeg;base64,${base64Image}`,
							length: "normal",
							stream: false
						})
					});

					if (!response.ok) {
						const errorText = await response.text();
						console.error(`Moondream API error (${response.status}):`, errorText);
						throw new Error(`Moondream API error: ${response.status} - ${errorText}`);
					}

					const result = await response.json<{ caption: string }>();
					console.log("Moondream caption:", result.caption);
					return { caption: result.caption };
				} catch (e) {
					console.error("Image captioning failed:", e);
					return { caption: `Image captioning failed: ${e instanceof Error ? e.message : 'Unknown error'}` };
				}
			});
		} else if (fileType === "pdf" || fileType === "application/pdf") {
			// PDF processing: Extract text and summarize
			metadata = await step.do("process-pdf", async () => {
				try {
					console.log(`Extracting text from PDF: ${filename}`);
					// Extract text from PDF
					const { text } = await extractText(getFileBytes(), { mergePages: true });

					console.log(`Extracted ${text?.length || 0} characters from PDF`);

					if (!text || text.trim().length === 0) {
						console.warn("PDF appears to be empty or contains no extractable text");
						return { rawText: "", summary: "PDF appears to be empty or contains no extractable text" };
					}

					// Truncate text if too long for AI model (max ~4000 chars for context)
					const truncatedText = text.substring(0, 4000);
					console.log(`Summarizing ${truncatedText.length} characters with Workers AI`);

					// Summarize with Workers AI using Llama 3.1 Fast for better performance
					const aiResult = await this.env.AI.run(
						"@cf/meta/llama-3.1-8b-instruct-fast",
						{
							messages: [
								{
									role: "system",
									content: "You are a helpful assistant that creates concise, informative summaries of documents. Focus on key points and main ideas."
								},
								{
									role: "user",
									content: `Please provide a concise summary of the following document:\n\n${truncatedText}`
								}
							]
						}
					) as { response: string };

					console.log(`Summary generated: ${aiResult.response?.substring(0, 100)}...`);

					return {
						rawText: text.substring(0, 50000), // Limit raw text storage to 50k chars
						summary: aiResult.response
					};
				} catch (e) {
					console.error("PDF processing failed:", e);
					return {
						rawText: "",
						summary: `PDF processing failed: ${e instanceof Error ? e.message : 'Unknown error'}`
					};
				}
			});
		}

		// Step 4: Store file in R2
		await step.do("store-in-r2", async () => {
			const r2Key = `files/${fileId}/${filename}`;
			// Get fresh bytes to avoid detached buffer
			const uint8Array = getFileBytes();
			await this.env.FILE_BUCKET.put(r2Key, uint8Array, {
				httpMetadata: {
					contentType: contentType || "application/octet-stream",
				},
				customMetadata: {
					originalFilename: filename,
					fileId: fileId,
				}
			});
			return r2Key;
		});

		// Step 5: Save metadata to D1
		await step.do("save-metadata-to-d1", async () => {
			const r2Key = `files/${fileId}/${filename}`;
			const now = Date.now();

			// Prepare complete metadata object
			const metadataJson = JSON.stringify(metadata || {});

			// Insert file record with JSON metadata
			await this.env.DB.prepare(`
				INSERT INTO files (id, filename, content_type, file_type, size, sha256_hash, ssdeep_hash, r2_key, metadata, created_at)
				VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
			`).bind(
				fileId,
				filename,
				contentType,
				fileType,
				size,
				hashes.sha256Hash,
				hashes.ssdeepHash,
				r2Key,
				metadataJson,
				now
			).run();

			console.log(`Saved file metadata to D1: ${fileId}`);
		});

		// Return workflow result
		const result: FileProcessingResult = {
			fileId,
			sha256Hash: hashes.sha256Hash,
			ssdeepHash: hashes.ssdeepHash,
			fileType,
			metadata
		};

		return result;
	}
}

/**
 * Main Worker
 * Handles HTTP requests for file upload, search, and serves the web UI
 */
export default {
	async fetch(req: Request, env: Env): Promise<Response> {
		const url = new URL(req.url);

		// Serve static assets
		if (url.pathname === "/" || url.pathname === "/index.html") {
			return env.ASSETS.fetch(req);
		}

		// API: Upload file
		if (url.pathname === "/api/upload" && req.method === "POST") {
			try {
				const formData = await req.formData();
				const file = formData.get("file") as File;

				if (!file) {
					return Response.json({ error: "No file provided" }, { status: 400 });
				}

				// Validate file size
				if (file.size === 0) {
					return Response.json({ error: "File is empty (0 bytes)" }, { status: 400 });
				}

				if (file.size > 100 * 1024 * 1024) { // 100 MB limit
					return Response.json({ error: "File too large (max 100 MB)" }, { status: 400 });
				}

				// Generate unique file ID
				const fileId = crypto.randomUUID();
				const arrayBuffer = await file.arrayBuffer();

				// Convert ArrayBuffer to base64 for workflow serialization
				// Process in chunks to avoid stack overflow on large files
				const uint8Array = new Uint8Array(arrayBuffer);
				const chunkSize = 8192;
				let binaryString = '';
				for (let i = 0; i < uint8Array.length; i += chunkSize) {
					const chunk = uint8Array.subarray(i, i + chunkSize);
					binaryString += String.fromCharCode.apply(null, Array.from(chunk));
				}
				const fileData = btoa(binaryString);

				// Start the workflow
				const instance = await env.FILE_WORKFLOW.create({
					params: {
						fileId,
						filename: file.name,
						contentType: file.type,
						size: file.size,
						fileData,
					}
				});

				return Response.json({
					success: true,
					id: fileId,
					workflowInstanceId: instance.id,
					message: "File uploaded and processing started"
				});
			} catch (error) {
				console.error("Upload error:", error);
				return Response.json(
					{ error: error instanceof Error ? error.message : "Upload failed" },
					{ status: 500 }
				);
			}
		}

		// API: Search files
		if (url.pathname === "/api/search" && req.method === "GET") {
			try {
				const query = url.searchParams.get("q");
				if (!query) {
					return Response.json({ error: "No search query provided" }, { status: 400 });
				}

				// Search using D1's JSON functions to query within metadata
				const filesResult = await env.DB.prepare(`
					SELECT * FROM files
					WHERE filename LIKE ?
					   OR sha256_hash = ?
					   OR ssdeep_hash = ?
					   OR json_extract(metadata, '$.caption') LIKE ?
					   OR json_extract(metadata, '$.summary') LIKE ?
					   OR json_extract(metadata, '$.rawText') LIKE ?
					ORDER BY created_at DESC
					LIMIT 50
				`).bind(
					`%${query}%`,
					query,
					query,
					`%${query}%`,
					`%${query}%`,
					`%${query}%`
				).all();

				// Parse JSON metadata for each file
				const files = filesResult.results.map((file: any) => ({
					...file,
					metadata: file.metadata ? JSON.parse(file.metadata) : null
				}));

				return Response.json({
					success: true,
					files,
					count: files.length
				});
			} catch (error) {
				console.error("Search error:", error);
				return Response.json(
					{ error: error instanceof Error ? error.message : "Search failed" },
					{ status: 500 }
				);
			}
		}

		// API: Get workflow status
		if (url.pathname === "/api/status" && req.method === "GET") {
			try {
				const instanceId = url.searchParams.get("id");
				if (!instanceId) {
					return Response.json({ error: "No instance ID provided" }, { status: 400 });
				}

				const instance = await env.FILE_WORKFLOW.get(instanceId);
				const status = await instance.status();

				// Enhanced status response with step details
				return Response.json({
					success: true,
					status: {
						status: status.status,
						output: status.output,
						error: status.error,
						steps: status.status === 'running' || status.status === 'queued'
							? [
								{ name: 'hash-file', status: 'running' },
								{ name: 'detect-file-type', status: 'pending' },
								{ name: 'store-in-r2', status: 'pending' },
								{ name: 'save-metadata-to-d1', status: 'pending' }
							]
							: status.status === 'complete'
							? [
								{ name: 'hash-file', status: 'complete' },
								{ name: 'detect-file-type', status: 'complete' },
								{ name: 'store-in-r2', status: 'complete' },
								{ name: 'save-metadata-to-d1', status: 'complete' }
							]
							: []
					}
				});
			} catch (error) {
				console.error("Status error:", error);
				return Response.json(
					{ error: error instanceof Error ? error.message : "Failed to get status" },
					{ status: 500 }
				);
			}
		}

		// 404 for other routes
		return Response.json({ error: "Not found" }, { status: 404 });
	},
};
