"""
Search service for the file management system using Meilisearch.
"""
import os
import json
import time
import uuid
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

try:
    import meilisearch
    MEILISEARCH_AVAILABLE = True
except ImportError:
    MEILISEARCH_AVAILABLE = False
    print("Meilisearch not available - install with: pip install meilisearch")

# Meilisearch configuration (can be set via environment variables)
MEILISEARCH_HOST = os.environ.get("MEILISEARCH_HOST", "http://localhost:7700")
MEILISEARCH_API_KEY = os.environ.get("MEILISEARCH_API_KEY", "masterKey")  # Set default API key to match docker container
MEILISEARCH_INDEX = os.environ.get("MEILISEARCH_INDEX", "files")

class SearchService:
    """
    Service for managing search functionality with Meilisearch.
    """
    def __init__(self):
        self.client = None
        self.index = None
        self.available = False
        
        if not MEILISEARCH_AVAILABLE:
            print("Meilisearch Python client not installed")
            return
            
        # Initialize Meilisearch client
        try:
            self.client = meilisearch.Client(MEILISEARCH_HOST, MEILISEARCH_API_KEY)
            health = self.client.health()
            if health and health.get("status") == "available":
                print(f"Connected to Meilisearch at {MEILISEARCH_HOST}")
                self.available = True
                
                # Create or get the index
                try:
                    # Check if index exists
                    index_exists = False
                    
                    # First approach - try to get the index directly
                    try:
                        # Try to get the specific index
                        self.client.get_index(MEILISEARCH_INDEX)
                        index_exists = True
                        print(f"Found existing index '{MEILISEARCH_INDEX}'")
                    except Exception as direct_check_error:
                        # If this fails, the index might not exist
                        print(f"Direct index check failed: {str(direct_check_error)}")
                        
                        # Second approach - get all indexes and check
                        try:
                            indexes = self.client.get_indexes()
                            
                            # Parse response based on client version
                            if hasattr(indexes, "results"):
                                # Newer client versions return an object with a results attribute
                                index_list = indexes.results
                                for idx in index_list:
                                    if hasattr(idx, "uid") and idx.uid == MEILISEARCH_INDEX:
                                        index_exists = True
                                        break
                            elif isinstance(indexes, dict) and "results" in indexes:
                                # Some versions return a dict with results key
                                index_list = indexes["results"]
                                for idx in index_list:
                                    if isinstance(idx, dict) and idx.get("uid") == MEILISEARCH_INDEX:
                                        index_exists = True
                                        break
                            elif hasattr(indexes, "__iter__"):
                                # Iterable returned
                                for idx in indexes:
                                    # Check for uid attribute or dict key
                                    uid = None
                                    if hasattr(idx, "uid"):
                                        uid = idx.uid
                                    elif isinstance(idx, dict) and "uid" in idx:
                                        uid = idx["uid"]
                                    
                                    if uid == MEILISEARCH_INDEX:
                                        index_exists = True
                                        break
                            
                            print(f"Index '{MEILISEARCH_INDEX}' {'exists' if index_exists else 'does not exist'}")
                        except Exception as e:
                            print(f"Error listing indexes: {str(e)}, assuming index doesn't exist")
                            index_exists = False
                    
                    if not index_exists:
                        # Create the index
                        self.client.create_index(MEILISEARCH_INDEX, {"primaryKey": "id"})
                        print(f"Created Meilisearch index: {MEILISEARCH_INDEX}")
                        
                        # Wait for index creation
                        time.sleep(1)
                    
                    # Get the index
                    self.index = self.client.index(MEILISEARCH_INDEX)
                    
                    # Configure index settings
                    self.configure_index()
                    
                    print(f"Meilisearch index '{MEILISEARCH_INDEX}' is ready")
                except Exception as idx_error:
                    print(f"Error setting up Meilisearch index: {str(idx_error)}")
                    self.available = False
            else:
                print("Meilisearch service is not healthy")
                self.available = False
        except Exception as e:
            print(f"Error connecting to Meilisearch: {str(e)}")
            print(f"Make sure Meilisearch is running at {MEILISEARCH_HOST}")
            self.available = False
    
    def configure_index(self):
        """
        Configure the Meilisearch index settings for optimal search experience.
        """
        if not self.available or not self.index:
            return
        
        try:
            # Configure searchable attributes (fields to search in)
            self.index.update_searchable_attributes([
                # File info
                "file_info.filename",
                "file_info.mime_type",
                "file_info.extension",
                
                # Text content
                "text_extraction.extracted_text",
                "text_extraction.llm_summary",
                "text_extraction.metadata.format",
                "text_extraction.metadata.encoding",
                
                # EXIF data
                "exif.Title",
                "exif.Description",
                "exif.Comment",
                "exif.Subject",
                "exif.Author",
                "exif.Keywords",
                "exif.Make",
                "exif.Model",
                "exif.Software",
                "exif.Artist",
                "exif.Copyright",
                "exif.Location",
                "exif.GPSLatitude",
                "exif.GPSLongitude",
                "exif.GPSPosition",
                
                # Magika detection
                "file_info.magika.mime_type",
                "file_info.magika.label",
                
                # Enrichment
                "enrichment.keywords",
                "enrichment.tags",
                "enrichment.llm_description",  # Image descriptions
                "enrichment.colors",
                "enrichment.image_stats.avg_color",
                "enrichment.width",
                "enrichment.height",
                
                # Hashes for similarity search
                "hashes.md5",
                "hashes.sha1",
                "hashes.sha256",
                "hashes.tlsh",
                "perceptual_hashes.phash",
                "perceptual_hashes.dhash",
                "perceptual_hashes.ahash",
                "perceptual_hashes.whash",

                # User-provided tags
                "tags",

                # User login information
                "user_login"
            ])
            
            # Configure filterable attributes (for faceted search)
            self.index.update_filterable_attributes([
                # Document ID (required for downloads)
                "id",
                
                # Basic file properties
                "file_info.mime_type",
                "file_info.extension",
                "file_info.size_category",
                "file_info.magika.mime_type",
                "file_info.magika.label",
                
                # Media properties
                "enrichment.width",
                "enrichment.height",
                "enrichment.format",
                "enrichment.color_mode",
                "enrichment.animation.is_animated",
                
                # EXIF data
                "exif.Make",
                "exif.Model",
                "exif.Software",
                "exif.DateTimeOriginal",
                "exif.CreateDate",
                "exif.ModifyDate",
                
                # Hashes for duplicate detection and similarity search
                "hashes.md5",
                "hashes.sha1",
                "hashes.sha256",
                "hashes.tlsh",
                "perceptual_hashes.phash",
                "perceptual_hashes.dhash",
                "perceptual_hashes.ahash",
                "perceptual_hashes.whash",
                
                # Temporal
                "upload_date",
                "text_extraction.metadata.transcription_model",
                "text_extraction.metadata.llm_summary_model",
                
                # User-provided tags
                "tags",

                # User login information
                "user_login"
            ])
            
            # Configure ranking rules
            self.index.update_ranking_rules([
                "words",
                "typo",
                "proximity",
                "attribute",
                "sort",
                "exactness"
            ])
            
            print("Meilisearch index configuration updated")
        except Exception as config_error:
            print(f"Error configuring Meilisearch index: {str(config_error)}")
    
    def add_document(self, file_metadata: Dict[str, Any]) -> Optional[str]:
        """
        Add a document to the search index.
        
        Args:
            file_metadata: The file metadata from file processing
            
        Returns:
            Document ID if successful, None otherwise
        """
        if not self.available or not self.index:
            return None
        
        try:
            # Generate a unique ID for the document
            doc_id = str(uuid.uuid4())
            
            # Create a search document from the metadata
            search_doc = self._prepare_document(file_metadata, doc_id)
            
            # Add to index
            result = self.index.add_documents([search_doc])
            
            # Wait for indexing to complete (optional, for testing)
            if "taskUid" in result:
                self.client.wait_for_task(result["taskUid"], timeout_in_ms=10000)
            
            return doc_id
        except Exception as add_error:
            print(f"Error adding document to search index: {str(add_error)}")
            return None
    
    def _prepare_document(self, metadata: Dict[str, Any], doc_id: str) -> Dict[str, Any]:
        """
        Prepare a document for indexing by flattening and optimizing for search.
        
        Args:
            metadata: The raw metadata from file processing
            doc_id: The unique ID for this document
            
        Returns:
            Document optimized for search indexing
        """
        # Start with a copy of the original metadata
        document = metadata.copy()
        
        # Add required ID field
        document["id"] = doc_id
        
        # Add upload date/time
        document["upload_date"] = datetime.now().isoformat()
        
        # Add size category for filtering
        if "file_info" in document and "size_bytes" in document["file_info"]:
            size_bytes = document["file_info"]["size_bytes"]
            if size_bytes < 10_000:  # 10 KB
                size_category = "tiny"
            elif size_bytes < 1_000_000:  # 1 MB
                size_category = "small"
            elif size_bytes < 10_000_000:  # 10 MB
                size_category = "medium"
            elif size_bytes < 100_000_000:  # 100 MB
                size_category = "large"
            else:
                size_category = "huge"
            
            if "file_info" in document:
                document["file_info"]["size_category"] = size_category
        
        # Ensure tags field exists, even if empty, for consistent schema
        if "tags" not in document:
            document["tags"] = []
        
        return document
    
    def search(self, query: str, filters: Optional[Dict[str, Any]] = None, limit: int = 20) -> Dict[str, Any]:
        """
        Search for documents in the index.
        
        Args:
            query: The search query
            filters: Optional dict of filters to apply
            limit: Maximum number of results to return
            
        Returns:
            Search results
        """
        if not self.available or not self.index:
            return {"hits": [], "error": "Search service unavailable"}
        
        try:
            # Build search parameters
            search_params = {
                "limit": limit
            }
            
            # Add filters if provided
            if filters:
                filter_conditions = []
                for key, value in filters.items():
                    # Special case for id field which doesn't need the prefix
                    if key == "id":
                        if isinstance(value, list):
                            or_conditions = [f"id = '{v}'" for v in value]
                            filter_conditions.append(f"({' OR '.join(or_conditions)})")
                        else:
                            filter_conditions.append(f"id = '{value}'")
                    elif isinstance(value, list):
                        # Handle multiple values for a field (OR condition)
                        or_conditions = [f"{key} = '{v}'" for v in value]
                        filter_conditions.append(f"({' OR '.join(or_conditions)})")
                    else:
                        # Handle single value
                        filter_conditions.append(f"{key} = '{value}'")
                
                if filter_conditions:
                    search_params["filter"] = " AND ".join(filter_conditions)
            
            # Execute search
            results = self.index.search(query, search_params)
            return results
        except Exception as search_error:
            print(f"Search error: {str(search_error)}")
            return {"hits": [], "error": str(search_error)}
    
    def suggest_tags(self, query: str, limit: int = 10) -> List[str]:
        """
        Suggest existing tags based on a query prefix using facet search.

        Args:
            query: The prefix to search for in tags.
            limit: Maximum number of suggestions to return.

        Returns:
            A list of matching tag suggestions.
        """
        if not self.available or not self.index:
            print("Tag suggestion unavailable: Search service not ready.")
            return []

        try:
            # Perform facet search on the 'tags' field
            # Note: Requires 'tags' to be in filterable_attributes
            results = self.index.facet_search(
                facet_name="tags", 
                facet_query=query 
            )

            # Extract the tag names from the results
            # The structure is {'facetHits': [{'value': 'tag_name', 'count': N}, ...]}
            if results and "facetHits" in results:
                suggestions = [hit["value"] for hit in results["facetHits"]]
                print(f"Tag suggestions for '{query}': {suggestions}")
                return suggestions
            else:
                print(f"No facet hits found for query '{query}'. Results: {results}")
                return []
        except Exception as suggest_error:
            print(f"Error suggesting tags for query '{query}': {str(suggest_error)}")
            return []
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from the index.
        
        Args:
            doc_id: The ID of the document to delete
            
        Returns:
            True if successful, False otherwise
        """
        if not self.available or not self.index:
            return False
        
        try:
            result = self.index.delete_document(doc_id)
            
            # Wait for deletion to complete (optional)
            if "taskUid" in result:
                self.client.wait_for_task(result["taskUid"])
            
            return True
        except Exception as del_error:
            print(f"Error deleting document: {str(del_error)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the search index.
        
        Returns:
            Statistics about the index
        """
        if not self.available or not self.index:
            return {"available": False}
        
        try:
            stats = self.index.get_stats()
            
            # Handle both dictionary and object response formats
            if hasattr(stats, "number_of_documents"):
                # Object format (newer versions)
                return {
                    "available": True,
                    "documents_count": getattr(stats, "number_of_documents", 0),
                    "is_indexing": getattr(stats, "is_indexing", False)
                }
            else:
                # Dictionary format (older versions)
                return {
                    "available": True,
                    "documents_count": stats.get("numberOfDocuments", 0),
                    "is_indexing": stats.get("isIndexing", False)
                }
        except Exception as stats_error:
            print(f"Error getting stats: {str(stats_error)}")
            return {"available": False, "error": str(stats_error)}


# Create a singleton instance
search_service = SearchService()