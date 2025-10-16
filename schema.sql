-- File metadata database schema for D1

-- Main files table with JSON metadata column
CREATE TABLE IF NOT EXISTS files (
  id TEXT PRIMARY KEY,
  filename TEXT NOT NULL,
  content_type TEXT,
  file_type TEXT,
  size INTEGER,
  sha256_hash TEXT NOT NULL,
  ssdeep_hash TEXT,
  r2_key TEXT NOT NULL,
  metadata JSON, -- Store all processing metadata as JSON
  uploaded_by TEXT, -- Email of user who uploaded (from Cloudflare Access JWT)
  created_at INTEGER NOT NULL
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_files_sha256 ON files(sha256_hash);
CREATE INDEX IF NOT EXISTS idx_files_filename ON files(filename);
CREATE INDEX IF NOT EXISTS idx_files_ssdeep ON files(ssdeep_hash);
CREATE INDEX IF NOT EXISTS idx_files_created ON files(created_at);
CREATE INDEX IF NOT EXISTS idx_files_uploaded_by ON files(uploaded_by);
