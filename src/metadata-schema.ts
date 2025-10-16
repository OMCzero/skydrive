/**
 * Metadata Schema Definitions
 *
 * This file defines the expected metadata structure for each file type.
 * When adding new metadata fields, increment the version number for that file type.
 */

export type FileTypeSchema = {
	version: number;
	requiredFields: string[];
	optionalFields: string[];
	description: string;
};

export const METADATA_SCHEMAS: Record<string, FileTypeSchema> = {
	'pdf': {
		version: 2,
		requiredFields: ['rawText', 'summary', 'ocrPerformed'],
		optionalFields: ['pageCount', 'author', 'title'],
		description: 'PDF documents with text extraction, OCR, and summarization'
	},
	'png': {
		version: 1,
		requiredFields: ['caption'],
		optionalFields: ['dimensions', 'colorSpace', 'exifData'],
		description: 'PNG images with AI-generated captions'
	},
	'jpg': {
		version: 1,
		requiredFields: ['caption'],
		optionalFields: ['dimensions', 'colorSpace', 'exifData'],
		description: 'JPEG images with AI-generated captions'
	},
	'jpeg': {
		version: 1,
		requiredFields: ['caption'],
		optionalFields: ['dimensions', 'colorSpace', 'exifData'],
		description: 'JPEG images with AI-generated captions'
	}
};

export type MetadataCheckResult = {
	needsReprocessing: boolean;
	reason?: 'schema_outdated' | 'missing_fields' | 'no_schema_version';
	currentVersion?: number;
	expectedVersion?: number;
	missingFields?: string[];
	fileType?: string;
};

/**
 * Check if a file needs reprocessing based on its metadata
 */
export function checkMetadataStatus(file: {
	file_type: string;
	metadata: any;
}): MetadataCheckResult {
	const schema = METADATA_SCHEMAS[file.file_type];

	// If no schema exists for this file type, no reprocessing needed
	if (!schema) {
		return { needsReprocessing: false };
	}

	// Parse metadata if it's a string
	const metadata = typeof file.metadata === 'string'
		? JSON.parse(file.metadata)
		: file.metadata;

	// Check if metadata has a schema version
	if (!metadata || !metadata._schemaVersion) {
		return {
			needsReprocessing: true,
			reason: 'no_schema_version',
			expectedVersion: schema.version,
			fileType: file.file_type,
			missingFields: schema.requiredFields
		};
	}

	// Check if schema version is outdated
	if (metadata._schemaVersion < schema.version) {
		// Find which fields are missing
		const existingFields = Object.keys(metadata);
		const missingFields = schema.requiredFields.filter(
			field => !existingFields.includes(field) || metadata[field] === null || metadata[field] === undefined
		);

		return {
			needsReprocessing: true,
			reason: 'schema_outdated',
			currentVersion: metadata._schemaVersion,
			expectedVersion: schema.version,
			fileType: file.file_type,
			missingFields
		};
	}

	// Check if required fields are present
	const existingFields = Object.keys(metadata);
	const missingFields = schema.requiredFields.filter(
		field => !existingFields.includes(field) || metadata[field] === null || metadata[field] === undefined
	);

	if (missingFields.length > 0) {
		return {
			needsReprocessing: true,
			reason: 'missing_fields',
			currentVersion: metadata._schemaVersion,
			expectedVersion: schema.version,
			fileType: file.file_type,
			missingFields
		};
	}

	return { needsReprocessing: false };
}

/**
 * Get a list of fields that need to be reprocessed
 */
export function getFieldsToReprocess(file: {
	file_type: string;
	metadata: any;
}): string[] {
	const check = checkMetadataStatus(file);
	if (!check.needsReprocessing) {
		return [];
	}
	return check.missingFields || [];
}

/**
 * Get the current schema version for a file type
 */
export function getCurrentSchemaVersion(fileType: string): number | null {
	const schema = METADATA_SCHEMAS[fileType];
	return schema ? schema.version : null;
}
