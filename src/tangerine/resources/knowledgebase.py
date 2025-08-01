import io
import logging
from typing import List

from flask import request
from flask_restful import Resource

from tangerine.file import File
from tangerine.models import KnowledgeBase
from tangerine.vector import vector_db

log = logging.getLogger("tangerine.resources.knowledgebase")


class KnowledgeBasesApi(Resource):
    def get(self):
        """Get all knowledgebases."""
        knowledgebases = KnowledgeBase.list()
        return {"data": [kb.to_dict() for kb in knowledgebases]}

    def post(self):
        """Create a new knowledgebase."""
        data = request.get_json()
        
        if not data:
            return {"error": "Request body is required"}, 400
        
        name = data.get("name")
        description = data.get("description")
        
        if not name:
            return {"error": "Name is required"}, 400
        if not description:
            return {"error": "Description is required"}, 400
        
        # Check if knowledgebase with this name already exists
        existing_kb = KnowledgeBase.get_by_name(name)
        if existing_kb:
            return {"error": f"KnowledgeBase with name '{name}' already exists"}, 409
        
        try:
            kb = KnowledgeBase.create(name=name, description=description)
            log.info("created knowledgebase %d: %s", kb.id, kb.name)
            return {"data": kb.to_dict()}, 201
        except Exception as e:
            log.exception("error creating knowledgebase")
            return {"error": f"Failed to create knowledgebase: {str(e)}"}, 500


class KnowledgeBaseApi(Resource):
    def get(self, id):
        """Get a specific knowledgebase."""
        try:
            kb_id = int(id)
        except ValueError:
            return {"error": "Invalid knowledgebase ID"}, 400
        
        kb = KnowledgeBase.get(kb_id)
        if not kb:
            return {"error": "KnowledgeBase not found"}, 404
        
        return {"data": kb.to_dict()}

    def put(self, id):
        """Update a knowledgebase."""
        try:
            kb_id = int(id)
        except ValueError:
            return {"error": "Invalid knowledgebase ID"}, 400
        
        kb = KnowledgeBase.get(kb_id)
        if not kb:
            return {"error": "KnowledgeBase not found"}, 404
        
        data = request.get_json()
        if not data:
            return {"error": "Request body is required"}, 400
        
        # Check for name conflicts if name is being updated
        new_name = data.get("name")
        if new_name and new_name != kb.name:
            existing_kb = KnowledgeBase.get_by_name(new_name)
            if existing_kb:
                return {"error": f"KnowledgeBase with name '{new_name}' already exists"}, 409
        
        try:
            updated_kb = kb.update(**data)
            log.info("updated knowledgebase %d", updated_kb.id)
            return {"data": updated_kb.to_dict()}
        except Exception as e:
            log.exception("error updating knowledgebase %d", kb_id)
            return {"error": f"Failed to update knowledgebase: {str(e)}"}, 500

    def delete(self, id):
        """Delete a knowledgebase."""
        try:
            kb_id = int(id)
        except ValueError:
            return {"error": "Invalid knowledgebase ID"}, 400
        
        kb = KnowledgeBase.get(kb_id)
        if not kb:
            return {"error": "KnowledgeBase not found"}, 404
        
        try:
            # This will raise ValueError if still associated with assistants
            kb.delete()
            
            # Delete all document chunks for this knowledgebase from vector DB
            search_filter = {"knowledgebase_id": str(kb_id)}
            deleted_docs = vector_db.delete_document_chunks(search_filter)
            
            log.info("deleted knowledgebase %d and %d document chunks", kb_id, len(deleted_docs))
            return {"message": f"KnowledgeBase deleted successfully. Removed {len(deleted_docs)} document chunks."}
        except ValueError as e:
            return {"error": str(e)}, 409
        except Exception as e:
            log.exception("error deleting knowledgebase %d", kb_id)
            return {"error": f"Failed to delete knowledgebase: {str(e)}"}, 500


class KnowledgeBaseDocuments(Resource):
    def post(self, id):
        """Upload documents to a knowledgebase."""
        try:
            kb_id = int(id)
        except ValueError:
            return {"error": "Invalid knowledgebase ID"}, 400
        
        kb = KnowledgeBase.get(kb_id)
        if not kb:
            return {"error": "KnowledgeBase not found"}, 404
        
        if "files" not in request.files:
            return {"error": "No files provided"}, 400
        
        files = request.files.getlist("files")
        if not files or (len(files) == 1 and files[0].filename == ""):
            return {"error": "No files selected"}, 400
        
        uploaded_files = []
        errors = []
        
        for uploaded_file in files:
            if uploaded_file.filename == "":
                continue
            
            try:
                file_content = uploaded_file.read()
                file_stream = io.BytesIO(file_content)
                
                file = File(
                    display_name=uploaded_file.filename,
                    file_stream=file_stream,
                    content_type=uploaded_file.content_type
                )
                
                # Add file to vector database
                vector_db.add_file(file, kb_id)
                
                # Add filename to knowledgebase
                kb.add_files([uploaded_file.filename])
                
                uploaded_files.append(uploaded_file.filename)
                log.info("uploaded file %s to knowledgebase %d", uploaded_file.filename, kb_id)
                
            except Exception as e:
                error_msg = f"Failed to process {uploaded_file.filename}: {str(e)}"
                errors.append(error_msg)
                log.exception("error processing file %s for knowledgebase %d", uploaded_file.filename, kb_id)
        
        response_data = {
            "uploaded_files": uploaded_files,
            "kb_files": kb.filenames or []
        }
        
        if errors:
            response_data["errors"] = errors
            status_code = 207  # Multi-status
        else:
            status_code = 200
        
        return response_data, status_code

    def delete(self, id):
        """Delete documents from a knowledgebase."""
        try:
            kb_id = int(id)
        except ValueError:
            return {"error": "Invalid knowledgebase ID"}, 400
        
        kb = KnowledgeBase.get(kb_id)
        if not kb:
            return {"error": "KnowledgeBase not found"}, 404
        
        data = request.get_json()
        if not data or "filenames" not in data:
            return {"error": "filenames array is required in request body"}, 400
        
        filenames = data["filenames"]
        if not isinstance(filenames, list):
            return {"error": "filenames must be an array"}, 400
        
        deleted_files = []
        errors = []
        
        for filename in filenames:
            try:
                # Delete from vector database
                search_filter = {"knowledgebase_id": str(kb_id), "display_name": filename}
                deleted_docs = vector_db.delete_document_chunks(search_filter)
                
                if deleted_docs:
                    deleted_files.append(filename)
                    log.info("deleted %d chunks for file %s from knowledgebase %d", len(deleted_docs), filename, kb_id)
                else:
                    errors.append(f"No documents found for {filename}")
                
            except Exception as e:
                error_msg = f"Failed to delete {filename}: {str(e)}"
                errors.append(error_msg)
                log.exception("error deleting file %s from knowledgebase %d", filename, kb_id)
        
        # Remove filenames from knowledgebase
        if deleted_files:
            kb.remove_files(deleted_files)
        
        response_data = {
            "deleted_files": deleted_files,
            "kb_files": kb.filenames or []
        }
        
        if errors:
            response_data["errors"] = errors
            status_code = 207  # Multi-status
        else:
            status_code = 200
        
        return response_data, status_code