import {
	WorkflowEntrypoint,
	WorkflowEvent,
	WorkflowStep,
} from "cloudflare:workers";
import { Magika } from "magika";
// @ts-ignore - ssdeep.js doesn't have TypeScript definitions
import ssdeepjs from "ssdeep.js";
import { extractText } from "unpdf";
import { PDFDocument } from "pdf-lib";
import { getCurrentSchemaVersion, checkMetadataStatus, getFieldsToReprocess } from "./metadata-schema";
import { jwtVerify, createRemoteJWKSet } from "jose";

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
	REPROCESS_WORKFLOW: Workflow;
	FILE_BUCKET: R2Bucket;
	DB: D1Database;
	AI: Ai;
	ASSETS: Fetcher;
	MOONDREAM_API_KEY?: string; // Optional - for Moondream cloud API
	TEAM_DOMAIN?: string; // Cloudflare Access team domain (e.g., https://yourteam.cloudflareaccess.com)
	POLICY_AUD?: string; // Cloudflare Access application audience tag
};

/**
 * Extract and optionally validate user from Cloudflare Access JWT
 * Returns user email or null if not available
 */
async function getUserFromAccessJWT(req: Request, env: Env, validate: boolean = false): Promise<string | null> {
	const jwt = req.headers.get('Cf-Access-Jwt-Assertion');

	if (!jwt) {
		console.warn('No Cf-Access-Jwt-Assertion header found');
		return null;
	}

	if (validate && env.TEAM_DOMAIN && env.POLICY_AUD) {
		// Validate JWT with Cloudflare's public keys
		try {
			const JWKS = createRemoteJWKSet(new URL(`${env.TEAM_DOMAIN}/cdn-cgi/access/certs`));
			const { payload } = await jwtVerify(jwt, JWKS, {
				issuer: env.TEAM_DOMAIN,
				audience: env.POLICY_AUD,
			});
			console.log(`Validated JWT for user: ${payload.email}`);
			return payload.email as string || null;
		} catch (error) {
			console.error('JWT validation failed:', error);
			return null;
		}
	} else {
		// Simple extraction without validation (Access already validated it)
		try {
			const payload = JSON.parse(atob(jwt.split('.')[1]));
			console.log(`Extracted user from JWT: ${payload.email}`);
			return payload.email || null;
		} catch (error) {
			console.error('Failed to parse JWT:', error);
			return null;
		}
	}
}

// Workflow parameters
type FileProcessingParams = {
	fileId: string;
	filename: string;
	contentType: string;
	size: number;
	fileData: string; // Base64 encoded file data
	uploadedBy: string | null; // Email of user from Cloudflare Access JWT
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
	/**
	 * Perform OCR on PDF by extracting embedded images and using vision model
	 * Processes images in parallel with rate limiting and backoff
	 */
	async performPDFOCR(pdfBytes: Uint8Array): Promise<string> {
		try {
			// Load PDF document with pdf-lib
			const pdfDoc = await PDFDocument.load(pdfBytes);
			const pages = pdfDoc.getPages();
			const numPages = pages.length;

			console.log(`Starting OCR for ${numPages} pages - extracting embedded images`);

			// First, extract all images from all pages
			type ImageInfo = {
				pageNum: number;
				imgIdx: number;
				imageBytes: Uint8Array;
			};

			const allImages: ImageInfo[] = [];

			for (let pageNum = 0; pageNum < numPages; pageNum++) {
				try {
					const page = pages[pageNum];

					// Get page resources to find images
					const resources = page.node.Resources();
					if (!resources) {
						continue;
					}

					const xObjects = resources.lookup(PDFDocument.of().context.obj('XObject'));
					if (!xObjects) {
						continue;
					}

					// Extract images from page
					const xObjectKeys = xObjects.dict.keys();

					for (const key of xObjectKeys) {
						try {
							const xObject = xObjects.lookup(key);
							if (xObject && xObject.dict && xObject.dict.get(PDFDocument.of().context.obj('Subtype'))?.toString() === '/Image') {
								// This is an image - extract it
								const imageData = xObject.getStream();
								if (imageData) {
									allImages.push({
										pageNum: pageNum + 1,
										imgIdx: allImages.filter(img => img.pageNum === pageNum + 1).length + 1,
										imageBytes: new Uint8Array(imageData)
									});
								}
							}
						} catch (imgError) {
							console.warn(`Failed to extract image from page ${pageNum + 1}:`, imgError);
						}
					}
				} catch (pageError) {
					console.error(`Error extracting images from page ${pageNum + 1}:`, pageError);
				}
			}

			console.log(`Extracted ${allImages.length} images from ${numPages} pages`);

			if (allImages.length === 0) {
				return "No images found in PDF for OCR";
			}

			// Process images in parallel batches with rate limiting
			// Image-to-Text limit: 720 requests per minute = 12 requests per second
			// Process in batches of 10 with delays to stay under limit
			const BATCH_SIZE = 10;
			const DELAY_BETWEEN_BATCHES_MS = 1000; // 1 second delay between batches

			const results: Array<{ pageNum: number; imgIdx: number; text: string }> = [];

			for (let i = 0; i < allImages.length; i += BATCH_SIZE) {
				const batch = allImages.slice(i, i + BATCH_SIZE);
				console.log(`Processing OCR batch ${Math.floor(i / BATCH_SIZE) + 1}/${Math.ceil(allImages.length / BATCH_SIZE)} (${batch.length} images)`);

				// Process this batch in parallel
				const batchPromises = batch.map(async (imageInfo) => {
					try {
						// Convert to base64
						const chunkSize = 8192;
						let binaryString = '';
						for (let j = 0; j < imageInfo.imageBytes.length; j += chunkSize) {
							const chunk = imageInfo.imageBytes.subarray(j, j + chunkSize);
							binaryString += String.fromCharCode.apply(null, Array.from(chunk));
						}
						const base64Image = btoa(binaryString);

						// Use vision model for OCR with retry logic
						let retries = 3;
						let lastError: Error | null = null;

						while (retries > 0) {
							try {
								const aiResult = await this.env.AI.run(
									"@cf/meta/llama-3.2-11b-vision-instruct",
									{
										messages: [
											{
												role: "user",
												content: [
													{
														type: "text",
														text: "Extract all text from this document image. Return only the text content, maintaining the original structure and formatting where possible."
													},
													{
														type: "image_url",
														image_url: `data:image/jpeg;base64,${base64Image}`
													}
												]
											}
										]
									}
								) as { response: string };

								if (aiResult.response) {
									return {
										pageNum: imageInfo.pageNum,
										imgIdx: imageInfo.imgIdx,
										text: aiResult.response
									};
								}
								break;
							} catch (error: any) {
								lastError = error;
								// Check if it's a rate limit error (429)
								if (error?.message?.includes('429') || error?.message?.includes('rate limit')) {
									console.warn(`Rate limited on page ${imageInfo.pageNum}, retrying after backoff...`);
									await new Promise(resolve => setTimeout(resolve, 2000 * (4 - retries))); // Exponential backoff
									retries--;
								} else {
									// Non-rate-limit error, don't retry
									throw error;
								}
							}
						}

						// If all retries failed
						if (lastError) {
							throw lastError;
						}

						return null;
					} catch (error) {
						console.error(`OCR failed for page ${imageInfo.pageNum} image ${imageInfo.imgIdx}:`, error);
						return {
							pageNum: imageInfo.pageNum,
							imgIdx: imageInfo.imgIdx,
							text: `[OCR failed: ${error instanceof Error ? error.message : 'Unknown error'}]`
						};
					}
				});

				// Wait for all images in this batch to complete
				const batchResults = await Promise.all(batchPromises);
				results.push(...batchResults.filter((r): r is { pageNum: number; imgIdx: number; text: string } => r !== null));

				// Delay before next batch (except for last batch)
				if (i + BATCH_SIZE < allImages.length) {
					await new Promise(resolve => setTimeout(resolve, DELAY_BETWEEN_BATCHES_MS));
				}
			}

			// Sort results by page number and image index
			results.sort((a, b) => {
				if (a.pageNum !== b.pageNum) return a.pageNum - b.pageNum;
				return a.imgIdx - b.imgIdx;
			});

			// Combine all text
			const allText = results.map(r =>
				`\n--- Page ${r.pageNum} (Image ${r.imgIdx}) ---\n${r.text}`
			).join('\n\n');

			console.log(`OCR completed: processed ${results.length} images from ${numPages} pages`);
			return allText || "No text extracted from images";
		} catch (error) {
			console.error("PDF OCR failed:", error);
			throw new Error(`PDF OCR failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
		}
	}

	async run(event: WorkflowEvent<FileProcessingParams>, step: WorkflowStep) {
		const { fileId, filename, contentType, size, fileData, uploadedBy } = event.payload;

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

					// Check if caption contains "train" - if so, detect train numbers
					let trainData: { train_detected: boolean; number: string | null } | undefined;
					if (result.caption.toLowerCase().includes('train')) {
						console.log("Train detected in caption, querying for train number...");
						try {
							const queryResponse = await fetch("https://api.moondream.ai/v1/query", {
								method: "POST",
								headers: {
									"Content-Type": "application/json",
									"X-Moondream-Auth": this.env.MOONDREAM_API_KEY
								},
								body: JSON.stringify({
									image_url: `data:image/jpeg;base64,${base64Image}`,
									question: "Is there a train number visible in this image? If yes, what is the train number? If no train number is visible, respond with 'no number'. Respond in JSON format: {train_detected:true/false, number:string or null}"
								})
							});

							if (queryResponse.ok) {
								const queryResult = await queryResponse.json<{ answer: string }>();
								console.log("Moondream query response:", queryResult.answer);

								// Try to parse JSON from the answer
								try {
									// Extract JSON from response (might be wrapped in text)
									const jsonMatch = queryResult.answer.match(/\{[^}]+\}/);
									if (jsonMatch) {
										trainData = JSON.parse(jsonMatch[0]);
									} else {
										// Fallback: try to extract train_detected and number from text
										const lowerAnswer = queryResult.answer.toLowerCase();
										trainData = {
											train_detected: !lowerAnswer.includes('no number') && !lowerAnswer.includes('not visible'),
											number: null
										};

										// Try to extract numbers from the response
										const numberMatch = queryResult.answer.match(/\b\d{2,6}\b/);
										if (numberMatch && trainData.train_detected) {
											trainData.number = numberMatch[0];
										}
									}
									console.log("Parsed train data:", trainData);
								} catch (parseError) {
									console.warn("Failed to parse train data JSON:", parseError);
									trainData = { train_detected: false, number: null };
								}
							} else {
								console.warn(`Train number query failed: ${queryResponse.status}`);
							}
						} catch (trainError) {
							console.error("Train number detection failed:", trainError);
						}
					}

					// Add schema version
					const schemaVersion = getCurrentSchemaVersion(fileType) || 1;
					return {
						caption: result.caption,
						trainData: trainData,
						_schemaVersion: schemaVersion
					};
				} catch (e) {
					console.error("Image captioning failed:", e);
					const schemaVersion = getCurrentSchemaVersion(fileType) || 1;
					return {
						caption: `Image captioning failed: ${e instanceof Error ? e.message : 'Unknown error'}`,
						_schemaVersion: schemaVersion
					};
				}
			});
		} else if (fileType === "pdf" || fileType === "application/pdf") {
			// PDF processing: Extract text and check if OCR is needed
			metadata = await step.do("process-pdf", async () => {
				try {
					console.log(`Extracting text from PDF: ${filename}`);
					// Extract text page-by-page to check for low-text pages
					const extracted = await extractText(getFileBytes(), { mergePages: false });

					// Check if any page has less than 50 characters
					const needsOCR = extracted.pages?.some(page =>
						!page.text || page.text.trim().length < 50
					) ?? false;

					let fullText = extracted.text || "";
					let ocrPerformed = false;

					// If OCR is needed, perform it on all pages
					if (needsOCR && fullText.length < 1000) {
						console.log(`PDF has low-text pages, triggering OCR for entire document`);
						try {
							const ocrText = await this.performPDFOCR(getFileBytes());
							fullText = ocrText;
							ocrPerformed = true;
							console.log(`OCR completed, extracted ${ocrText.length} characters`);
						} catch (ocrError) {
							console.error("OCR failed, using original extraction:", ocrError);
							// Fall back to original extraction
						}
					}

					console.log(`Total extracted text: ${fullText.length} characters (OCR: ${ocrPerformed})`);

					if (!fullText || fullText.trim().length === 0) {
						console.warn("PDF appears to be empty or contains no extractable text");
						return {
							rawText: "",
							summary: "PDF appears to be empty or contains no extractable text",
							ocrPerformed: false
						};
					}

					// Truncate text if too long for AI model (max ~4000 chars for context)
					const truncatedText = fullText.substring(0, 4000);
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

					// Add schema version
					const schemaVersion = getCurrentSchemaVersion(fileType) || 2;
					return {
						rawText: fullText.substring(0, 50000), // Limit raw text storage to 50k chars
						summary: aiResult.response,
						ocrPerformed,
						_schemaVersion: schemaVersion
					};
				} catch (e) {
					console.error("PDF processing failed:", e);
					const schemaVersion = getCurrentSchemaVersion(fileType) || 2;
					return {
						rawText: "",
						summary: `PDF processing failed: ${e instanceof Error ? e.message : 'Unknown error'}`,
						ocrPerformed: false,
						_schemaVersion: schemaVersion
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
				INSERT INTO files (id, filename, content_type, file_type, size, sha256_hash, ssdeep_hash, r2_key, metadata, uploaded_by, created_at)
				VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
				uploadedBy,
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
 * Reprocessing Workflow Parameters
 */
type ReprocessingParams = {
	fileId: string;
	fieldsToReprocess: string[];
};

/**
 * File Reprocessing Workflow
 * Selectively reprocesses missing metadata fields without re-uploading the file
 */
export class FileReprocessingWorkflow extends WorkflowEntrypoint<Env, ReprocessingParams> {
	// Share the OCR method with main workflow
	async performPDFOCR(pdfBytes: Uint8Array): Promise<string> {
		// Same implementation as FileProcessingWorkflow
		// (duplicated for now, could be extracted to shared helper)
		try {
			const pdfDoc = await PDFDocument.load(pdfBytes);
			const pages = pdfDoc.getPages();
			const numPages = pages.length;

			console.log(`Starting OCR for ${numPages} pages - extracting embedded images`);

			type ImageInfo = {
				pageNum: number;
				imgIdx: number;
				imageBytes: Uint8Array;
			};

			const allImages: ImageInfo[] = [];

			for (let pageNum = 0; pageNum < numPages; pageNum++) {
				try {
					const page = pages[pageNum];
					const resources = page.node.Resources();
					if (!resources) continue;

					const xObjects = resources.lookup(PDFDocument.of().context.obj('XObject'));
					if (!xObjects) continue;

					const xObjectKeys = xObjects.dict.keys();

					for (const key of xObjectKeys) {
						try {
							const xObject = xObjects.lookup(key);
							if (xObject && xObject.dict && xObject.dict.get(PDFDocument.of().context.obj('Subtype'))?.toString() === '/Image') {
								const imageData = xObject.getStream();
								if (imageData) {
									allImages.push({
										pageNum: pageNum + 1,
										imgIdx: allImages.filter(img => img.pageNum === pageNum + 1).length + 1,
										imageBytes: new Uint8Array(imageData)
									});
								}
							}
						} catch (imgError) {
							console.warn(`Failed to extract image from page ${pageNum + 1}:`, imgError);
						}
					}
				} catch (pageError) {
					console.error(`Error extracting images from page ${pageNum + 1}:`, pageError);
				}
			}

			console.log(`Extracted ${allImages.length} images from ${numPages} pages`);

			if (allImages.length === 0) {
				return "No images found in PDF for OCR";
			}

			const BATCH_SIZE = 10;
			const DELAY_BETWEEN_BATCHES_MS = 1000;
			const results: Array<{ pageNum: number; imgIdx: number; text: string }> = [];

			for (let i = 0; i < allImages.length; i += BATCH_SIZE) {
				const batch = allImages.slice(i, i + BATCH_SIZE);
				console.log(`Processing OCR batch ${Math.floor(i / BATCH_SIZE) + 1}/${Math.ceil(allImages.length / BATCH_SIZE)} (${batch.length} images)`);

				const batchPromises = batch.map(async (imageInfo) => {
					try {
						const chunkSize = 8192;
						let binaryString = '';
						for (let j = 0; j < imageInfo.imageBytes.length; j += chunkSize) {
							const chunk = imageInfo.imageBytes.subarray(j, j + chunkSize);
							binaryString += String.fromCharCode.apply(null, Array.from(chunk));
						}
						const base64Image = btoa(binaryString);

						let retries = 3;
						let lastError: Error | null = null;

						while (retries > 0) {
							try {
								const aiResult = await this.env.AI.run(
									"@cf/meta/llama-3.2-11b-vision-instruct",
									{
										messages: [
											{
												role: "user",
												content: [
													{
														type: "text",
														text: "Extract all text from this document image. Return only the text content, maintaining the original structure and formatting where possible."
													},
													{
														type: "image_url",
														image_url: `data:image/jpeg;base64,${base64Image}`
													}
												]
											}
										]
									}
								) as { response: string };

								if (aiResult.response) {
									return {
										pageNum: imageInfo.pageNum,
										imgIdx: imageInfo.imgIdx,
										text: aiResult.response
									};
								}
								break;
							} catch (error: any) {
								lastError = error;
								if (error?.message?.includes('429') || error?.message?.includes('rate limit')) {
									console.warn(`Rate limited on page ${imageInfo.pageNum}, retrying after backoff...`);
									await new Promise(resolve => setTimeout(resolve, 2000 * (4 - retries)));
									retries--;
								} else {
									throw error;
								}
							}
						}

						if (lastError) {
							throw lastError;
						}

						return null;
					} catch (error) {
						console.error(`OCR failed for page ${imageInfo.pageNum} image ${imageInfo.imgIdx}:`, error);
						return {
							pageNum: imageInfo.pageNum,
							imgIdx: imageInfo.imgIdx,
							text: `[OCR failed: ${error instanceof Error ? error.message : 'Unknown error'}]`
						};
					}
				});

				const batchResults = await Promise.all(batchPromises);
				results.push(...batchResults.filter((r): r is { pageNum: number; imgIdx: number; text: string } => r !== null));

				if (i + BATCH_SIZE < allImages.length) {
					await new Promise(resolve => setTimeout(resolve, DELAY_BETWEEN_BATCHES_MS));
				}
			}

			results.sort((a, b) => {
				if (a.pageNum !== b.pageNum) return a.pageNum - b.pageNum;
				return a.imgIdx - b.imgIdx;
			});

			const allText = results.map(r =>
				`\n--- Page ${r.pageNum} (Image ${r.imgIdx}) ---\n${r.text}`
			).join('\n\n');

			console.log(`OCR completed: processed ${results.length} images from ${numPages} pages`);
			return allText || "No text extracted from images";
		} catch (error) {
			console.error("PDF OCR failed:", error);
			throw new Error(`PDF OCR failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
		}
	}

	async run(event: WorkflowEvent<ReprocessingParams>, step: WorkflowStep) {
		const { fileId, fieldsToReprocess } = event.payload;

		console.log(`Starting reprocessing for file ${fileId}, fields: ${fieldsToReprocess.join(', ')}`);

		// Step 1: Fetch file from D1
		const fileRecord = await step.do("fetch-file-record", async () => {
			const result = await this.env.DB.prepare(`
				SELECT * FROM files WHERE id = ?
			`).bind(fileId).first();

			if (!result) {
				throw new Error(`File not found: ${fileId}`);
			}

			return result as any;
		});

		// Step 2: Fetch file from R2
		const fileBytes = await step.do("fetch-file-from-r2", async () => {
			const r2Object = await this.env.FILE_BUCKET.get(fileRecord.r2_key);
			if (!r2Object) {
				throw new Error(`File not found in R2: ${fileRecord.r2_key}`);
			}

			const arrayBuffer = await r2Object.arrayBuffer();
			return new Uint8Array(arrayBuffer);
		});

		// Parse existing metadata
		const existingMetadata = fileRecord.metadata ? JSON.parse(fileRecord.metadata) : {};
		let updatedMetadata = { ...existingMetadata };

		// Step 3: Reprocess based on file type and missing fields
		if (fileRecord.file_type === "pdf" || fileRecord.file_type === "application/pdf") {
			// PDF reprocessing
			const pdfData = await step.do("reprocess-pdf", async () => {
				const newData: any = {};

				// Check if we need to extract text/OCR
				if (fieldsToReprocess.includes('rawText') || fieldsToReprocess.includes('summary') || fieldsToReprocess.includes('ocrPerformed')) {
					console.log(`Re-extracting text from PDF`);
					const extracted = await extractText(fileBytes, { mergePages: false });

					const needsOCR = extracted.pages?.some(page =>
						!page.text || page.text.trim().length < 50
					) ?? false;

					let fullText = extracted.text || "";
					let ocrPerformed = false;

					if (needsOCR && fullText.length < 1000) {
						console.log(`PDF has low-text pages, triggering OCR`);
						try {
							const ocrText = await this.performPDFOCR(fileBytes);
							fullText = ocrText;
							ocrPerformed = true;
						} catch (ocrError) {
							console.error("OCR failed:", ocrError);
						}
					}

					newData.rawText = fullText.substring(0, 50000);
					newData.ocrPerformed = ocrPerformed;

					// Generate summary if text was extracted
					if (fullText && fullText.trim().length > 0 && fieldsToReprocess.includes('summary')) {
						const truncatedText = fullText.substring(0, 4000);
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

						newData.summary = aiResult.response;
					}
				}

				// Update schema version
				const schemaVersion = getCurrentSchemaVersion(fileRecord.file_type) || 2;
				newData._schemaVersion = schemaVersion;

				return newData;
			});

			updatedMetadata = { ...updatedMetadata, ...pdfData };
		} else if (fileRecord.file_type.includes("image") || fileRecord.file_type.includes("png") || fileRecord.file_type.includes("jpg") || fileRecord.file_type.includes("jpeg")) {
			// Image reprocessing
			const imageData = await step.do("reprocess-image", async () => {
				const newData: any = {};

				// Check if we need to caption
				if (fieldsToReprocess.includes('caption')) {
					if (!this.env.MOONDREAM_API_KEY) {
						console.warn("MOONDREAM_API_KEY not set - skipping image captioning");
						newData.caption = "Image captioning skipped (no API key)";
					} else {
						// Convert to base64
						const chunkSize = 8192;
						let binaryString = '';
						for (let i = 0; i < fileBytes.length; i += chunkSize) {
							const chunk = fileBytes.subarray(i, i + chunkSize);
							binaryString += String.fromCharCode.apply(null, Array.from(chunk));
						}
						const base64Image = btoa(binaryString);

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

						if (response.ok) {
							const result = await response.json<{ caption: string }>();
							newData.caption = result.caption;

							// Check if caption contains "train" - if so, detect train numbers
							if (result.caption.toLowerCase().includes('train')) {
								console.log("Train detected in caption during reprocessing, querying for train number...");
								try {
									const queryResponse = await fetch("https://api.moondream.ai/v1/query", {
										method: "POST",
										headers: {
											"Content-Type": "application/json",
											"X-Moondream-Auth": this.env.MOONDREAM_API_KEY
										},
										body: JSON.stringify({
											image_url: `data:image/jpeg;base64,${base64Image}`,
											question: "Is there a train number visible in this image? If yes, what is the train number? If no train number is visible, respond with 'no number'. Respond in JSON format: {train_detected:true/false, number:string or null}"
										})
									});

									if (queryResponse.ok) {
										const queryResult = await queryResponse.json<{ answer: string }>();
										console.log("Moondream query response:", queryResult.answer);

										try {
											const jsonMatch = queryResult.answer.match(/\{[^}]+\}/);
											if (jsonMatch) {
												newData.trainData = JSON.parse(jsonMatch[0]);
											} else {
												const lowerAnswer = queryResult.answer.toLowerCase();
												newData.trainData = {
													train_detected: !lowerAnswer.includes('no number') && !lowerAnswer.includes('not visible'),
													number: null
												};
												const numberMatch = queryResult.answer.match(/\b\d{2,6}\b/);
												if (numberMatch && newData.trainData.train_detected) {
													newData.trainData.number = numberMatch[0];
												}
											}
										} catch (parseError) {
											console.warn("Failed to parse train data JSON during reprocessing:", parseError);
											newData.trainData = { train_detected: false, number: null };
										}
									}
								} catch (trainError) {
									console.error("Train number detection failed during reprocessing:", trainError);
								}
							}
						} else {
							newData.caption = "Image captioning failed";
						}
					}
				}

				// Update schema version
				const schemaVersion = getCurrentSchemaVersion(fileRecord.file_type) || 1;
				newData._schemaVersion = schemaVersion;

				return newData;
			});

			updatedMetadata = { ...updatedMetadata, ...imageData };
		}

		// Step 4: Update metadata in D1
		await step.do("update-metadata", async () => {
			const metadataJson = JSON.stringify(updatedMetadata);

			await this.env.DB.prepare(`
				UPDATE files SET metadata = ? WHERE id = ?
			`).bind(metadataJson, fileId).run();

			console.log(`Updated metadata for file ${fileId}`);
		});

		return {
			fileId,
			fieldsReprocessed: fieldsToReprocess,
			success: true
		};
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

		// Admin dashboard
		if (url.pathname === "/admin" || url.pathname === "/admin.html") {
			return env.ASSETS.fetch(new Request(new URL("/admin.html", req.url), req));
		}

		// Docs pages
		if (url.pathname === "/docs/architecture" || url.pathname === "/docs/architecture.html") {
			return env.ASSETS.fetch(new Request(new URL("/docs/architecture.html", req.url), req));
		}

		// API: Check if hash exists (for browser-side duplicate detection)
		if (url.pathname === "/api/check-hash" && req.method === "POST") {
			try {
				const body = await req.json<{ hash: string }>();
				const hash = body.hash;

				if (!hash || typeof hash !== 'string' || hash.length !== 64) {
					return Response.json({ error: "Invalid SHA-256 hash" }, { status: 400 });
				}

				console.log(`Hash check: ${hash.substring(0, 16)}...`);

				// Check if file with this hash already exists
				const existingFile = await env.DB.prepare(`
					SELECT id, filename, content_type, file_type, size, created_at, metadata
					FROM files
					WHERE sha256_hash = ?
					LIMIT 1
				`).bind(hash).first();

				if (existingFile) {
					console.log(`Duplicate found via hash check: ${hash.substring(0, 16)}...`);

					// Parse metadata for response
					const metadata = existingFile.metadata ? JSON.parse(existingFile.metadata as string) : null;

					return Response.json({
						exists: true,
						file: {
							id: existingFile.id,
							filename: existingFile.filename,
							contentType: existingFile.content_type,
							fileType: existingFile.file_type,
							size: existingFile.size,
							createdAt: existingFile.created_at,
							metadata
						}
					});
				}

				return Response.json({ exists: false });
			} catch (error) {
				console.error("Hash check error:", error);
				return Response.json(
					{ error: error instanceof Error ? error.message : "Hash check failed" },
					{ status: 500 }
				);
			}
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

				// Read file data
				const arrayBuffer = await file.arrayBuffer();
				const uint8Array = new Uint8Array(arrayBuffer);

				// Calculate SHA-256 hash to check for duplicates
				const sha256Buffer = await crypto.subtle.digest("SHA-256", uint8Array);
				const sha256Array = Array.from(new Uint8Array(sha256Buffer));
				const sha256Hash = sha256Array.map(b => b.toString(16).padStart(2, '0')).join('');

				console.log(`Upload attempt: ${file.name}, SHA-256: ${sha256Hash.substring(0, 16)}...`);

				// Check if file with this hash already exists
				const existingFile = await env.DB.prepare(`
					SELECT id, filename, content_type, file_type, size, created_at, metadata
					FROM files
					WHERE sha256_hash = ?
					LIMIT 1
				`).bind(sha256Hash).first();

				if (existingFile) {
					console.log(`Duplicate detected: ${sha256Hash.substring(0, 16)}... already exists as ${existingFile.id}`);

					// Parse metadata for response
					const metadata = existingFile.metadata ? JSON.parse(existingFile.metadata as string) : null;

					return Response.json({
						success: false,
						duplicate: true,
						message: "This file has already been uploaded",
						existingFile: {
							id: existingFile.id,
							filename: existingFile.filename,
							contentType: existingFile.content_type,
							fileType: existingFile.file_type,
							size: existingFile.size,
							createdAt: existingFile.created_at,
							metadata
						}
					}, { status: 409 }); // 409 Conflict
				}

				// Extract user from Cloudflare Access JWT with full validation
				const uploadedBy = await getUserFromAccessJWT(req, env, true);
				console.log(`File uploaded by: ${uploadedBy || 'unknown user'}`);

				// Generate unique file ID for new upload
				const fileId = crypto.randomUUID();

				// Convert ArrayBuffer to base64 for workflow serialization
				// Process in chunks to avoid stack overflow on large files
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
						uploadedBy,
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
					   OR json_extract(metadata, '$.trainData.number') = ?
					ORDER BY created_at DESC
					LIMIT 50
				`).bind(
					`%${query}%`,
					query,
					query,
					`%${query}%`,
					`%${query}%`,
					`%${query}%`,
					query
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

		// API: Admin - List all files
		if (url.pathname === "/api/admin/files" && req.method === "GET") {
			try {
				const filesResult = await env.DB.prepare(`
					SELECT * FROM files
					ORDER BY created_at DESC
				`).all();

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
				console.error("Admin list error:", error);
				return Response.json(
					{ error: error instanceof Error ? error.message : "Failed to list files" },
					{ status: 500 }
				);
			}
		}

		// API: Admin - Delete file
		if (url.pathname.startsWith("/api/admin/files/") && req.method === "DELETE") {
			try {
				const fileId = url.pathname.split("/").pop();
				if (!fileId) {
					return Response.json({ error: "No file ID provided" }, { status: 400 });
				}

				// Get file info from D1
				const fileResult = await env.DB.prepare(`
					SELECT * FROM files WHERE id = ?
				`).bind(fileId).first();

				if (!fileResult) {
					return Response.json({ error: "File not found in database" }, { status: 404 });
				}

				const file = fileResult as any;
				let r2DeleteSuccess = false;
				let r2Error: string | null = null;

				// Try to delete from R2
				try {
					const r2Object = await env.FILE_BUCKET.head(file.r2_key);
					if (r2Object) {
						await env.FILE_BUCKET.delete(file.r2_key);
						console.log(`Deleted from R2: ${file.r2_key}`);
						r2DeleteSuccess = true;
					} else {
						r2Error = "File not found in R2 storage";
						console.warn(`File not found in R2: ${file.r2_key}`);
					}
				} catch (error) {
					r2Error = error instanceof Error ? error.message : "R2 deletion failed";
					console.error("R2 deletion error:", error);
				}

				// Delete from D1 regardless of R2 status
				let d1DeleteSuccess = false;
				let d1Error: string | null = null;

				try {
					// First, delete any related records in file_metadata table (legacy table)
					try {
						await env.DB.prepare(`
							DELETE FROM file_metadata WHERE file_id = ?
						`).bind(fileId).run();
						console.log(`Deleted related file_metadata entries for: ${fileId}`);
					} catch (legacyError) {
						// Continue even if legacy table deletion fails (table might not exist or no records)
						console.warn("Legacy file_metadata deletion warning:", legacyError);
					}

					// Now delete the main file record
					const result = await env.DB.prepare(`
						DELETE FROM files WHERE id = ?
					`).bind(fileId).run();

					if (result.meta.changes > 0) {
						d1DeleteSuccess = true;
						console.log(`Deleted file metadata from D1: ${fileId}`);
					} else {
						d1Error = "No rows deleted from D1";
					}
				} catch (error) {
					d1Error = error instanceof Error ? error.message : "D1 deletion failed";
					console.error("D1 deletion error:", error);
				}

				// Return detailed status
				const warnings = [];
				if (!r2DeleteSuccess && r2Error) {
					warnings.push(`R2: ${r2Error}`);
				}
				if (!d1DeleteSuccess && d1Error) {
					warnings.push(`D1: ${d1Error}`);
				}

				// If D1 deletion succeeded, consider it a success (main goal)
				if (d1DeleteSuccess) {
					return Response.json({
						success: true,
						message: warnings.length > 0
							? `File removed from database with warnings: ${warnings.join(', ')}`
							: "File deleted successfully from all systems",
						fileId,
						warnings: warnings.length > 0 ? warnings : undefined,
						r2Status: r2DeleteSuccess ? "deleted" : "failed",
						d1Status: "deleted"
					});
				} else {
					// D1 deletion failed - this is a real error
					return Response.json({
						error: `Failed to delete from database: ${d1Error}`,
						details: { r2Status: r2DeleteSuccess ? "deleted" : "failed", d1Status: "failed" }
					}, { status: 500 });
				}
			} catch (error) {
				console.error("Admin delete error:", error);
				return Response.json(
					{ error: error instanceof Error ? error.message : "Failed to delete file" },
					{ status: 500 }
				);
			}
		}

		// API: Get recent files (last 5 uploads)
		if (url.pathname === "/api/recent" && req.method === "GET") {
			try {
				const filesResult = await env.DB.prepare(`
					SELECT * FROM files
					ORDER BY created_at DESC
					LIMIT 5
				`).all();

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
				console.error("Recent files error:", error);
				return Response.json(
					{ error: error instanceof Error ? error.message : "Failed to fetch recent files" },
					{ status: 500 }
				);
			}
		}

		// API: Admin - Scan metadata for outdated/missing fields
		if (url.pathname === "/api/admin/scan-metadata" && req.method === "POST") {
			try {
				const filesResult = await env.DB.prepare(`
					SELECT * FROM files
					ORDER BY created_at DESC
				`).all();

				const needsReprocessing: Array<{
					fileId: string;
					filename: string;
					fileType: string;
					reason: string;
					currentVersion?: number;
					expectedVersion?: number;
					missingFields: string[];
				}> = [];

				for (const fileRecord of filesResult.results) {
					const file = fileRecord as any;
					const checkResult = checkMetadataStatus({
						file_type: file.file_type,
						metadata: file.metadata
					});

					if (checkResult.needsReprocessing) {
						needsReprocessing.push({
							fileId: file.id,
							filename: file.filename,
							fileType: file.file_type,
							reason: checkResult.reason || 'unknown',
							currentVersion: checkResult.currentVersion,
							expectedVersion: checkResult.expectedVersion,
							missingFields: checkResult.missingFields || []
						});
					}
				}

				// Breakdown by reason
				const breakdown = {
					no_schema_version: needsReprocessing.filter(f => f.reason === 'no_schema_version').length,
					schema_outdated: needsReprocessing.filter(f => f.reason === 'schema_outdated').length,
					missing_fields: needsReprocessing.filter(f => f.reason === 'missing_fields').length
				};

				return Response.json({
					success: true,
					totalFiles: filesResult.results.length,
					needsReprocessing,
					count: needsReprocessing.length,
					breakdown
				});
			} catch (error) {
				console.error("Metadata scan error:", error);
				return Response.json(
					{ error: error instanceof Error ? error.message : "Failed to scan metadata" },
					{ status: 500 }
				);
			}
		}

		// API: Admin - Reprocess file metadata
		if (url.pathname.startsWith("/api/admin/reprocess/") && req.method === "POST") {
			try {
				const fileId = url.pathname.split("/").pop();
				if (!fileId) {
					return Response.json({ error: "No file ID provided" }, { status: 400 });
				}

				// Get file from D1
				const fileResult = await env.DB.prepare(`
					SELECT * FROM files WHERE id = ?
				`).bind(fileId).first();

				if (!fileResult) {
					return Response.json({ error: "File not found" }, { status: 404 });
				}

				const file = fileResult as any;

				// Check what fields need reprocessing
				const fieldsToReprocess = getFieldsToReprocess({
					file_type: file.file_type,
					metadata: file.metadata
				});

				if (fieldsToReprocess.length === 0) {
					return Response.json({
						success: true,
						message: "File metadata is up to date, no reprocessing needed",
						fileId
					});
				}

				// Start reprocessing workflow
				const instance = await env.REPROCESS_WORKFLOW.create({
					params: {
						fileId,
						fieldsToReprocess
					}
				});

				return Response.json({
					success: true,
					message: "Reprocessing started",
					fileId,
					fieldsToReprocess,
					workflowInstanceId: instance.id
				});
			} catch (error) {
				console.error("Reprocess error:", error);
				return Response.json(
					{ error: error instanceof Error ? error.message : "Failed to start reprocessing" },
					{ status: 500 }
				);
			}
		}

		// API: Download/Preview file
		if (url.pathname.startsWith("/api/files/") && req.method === "GET") {
			try {
				const fileId = url.pathname.split("/")[3]; // /api/files/:fileId
				if (!fileId) {
					return Response.json({ error: "No file ID provided" }, { status: 400 });
				}

				// Get file info from D1
				const fileResult = await env.DB.prepare(`
					SELECT * FROM files WHERE id = ?
				`).bind(fileId).first();

				if (!fileResult) {
					return Response.json({ error: "File not found" }, { status: 404 });
				}

				const file = fileResult as any;

				// Get file from R2
				const r2Object = await env.FILE_BUCKET.get(file.r2_key);
				if (!r2Object) {
					return Response.json({ error: "File not found in storage" }, { status: 404 });
				}

				// Return file with appropriate headers
				const headers = new Headers();
				headers.set('Content-Type', file.content_type || 'application/octet-stream');
				headers.set('Content-Disposition', `inline; filename="${file.filename}"`);
				headers.set('Content-Length', file.size.toString());

				return new Response(r2Object.body, {
					headers
				});
			} catch (error) {
				console.error("File download error:", error);
				return Response.json(
					{ error: error instanceof Error ? error.message : "Failed to download file" },
					{ status: 500 }
				);
			}
		}

		// 404 for other routes
		return Response.json({ error: "Not found" }, { status: 404 });
	},
};
