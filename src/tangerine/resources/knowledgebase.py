import json
import logging

from flask import Response, request, stream_with_context
from flask_restful import Resource

from tangerine.file import File
from tangerine.models import KnowledgeBase
from tangerine.utils import embed_files_for_knowledgebase, remove_files_from_knowledgebase
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

        # This will raise ValueError if still associated with assistants
        try:
            kb.delete()
        except ValueError as e:
            return {"error": str(e)}, 409

        # Delete all document chunks for this knowledgebase from vector DB
        try:
            search_filter = {"knowledgebase_id": str(kb_id)}
            deleted_docs = vector_db.delete_document_chunks(search_filter)
        except Exception as e:
            log.exception("error deleting vector database chunks for knowledgebase %d", kb_id)
            return {"error": f"Failed to delete document chunks: {str(e)}"}, 500

        log.info("deleted knowledgebase %d and %d document chunks", kb_id, len(deleted_docs))
        return {
            "message": f"KnowledgeBase deleted successfully. Removed {len(deleted_docs)} document chunks."
        }


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

        # Check if the post request has the file part
        if "file" not in request.files:
            return {"error": "No file part"}, 400

        request_source = request.form.get("source", "default")

        files = []
        for file in request.files.getlist("file"):
            content = file.stream.read()
            if not file.filename:
                return {"error": "File must have a filename"}, 400
            new_file = File(
                source=request_source, full_path=file.filename, content=content.decode("utf-8")
            )
            try:
                new_file.validate()
            except ValueError as err:
                return {"error": f"validation failed for {file.filename}: {str(err)}"}, 400
            files.append(new_file)

        def generate_progress():
            for file in files:
                yield json.dumps({"file": file.display_name, "step": "start"}) + "\n"
                embed_files_for_knowledgebase([file], kb_id)
                yield json.dumps({"file": file.display_name, "step": "end"}) + "\n"

        return Response(stream_with_context(generate_progress()), mimetype="application/json")

    def delete(self, id):
        """Delete documents from a knowledgebase."""
        try:
            kb_id = int(id)
        except ValueError:
            return {"error": "Invalid knowledgebase ID"}, 400

        kb = KnowledgeBase.get(kb_id)
        if not kb:
            return {"error": "KnowledgeBase not found"}, 404

        if not request.json:
            return {"error": "No JSON data provided"}, 400

        source = request.json.get("source")
        full_path = request.json.get("full_path")
        delete_all = bool(request.json.get("all", False))

        if not source and not full_path and not delete_all:
            return {"error": "'source' or 'full_path' required when not using 'all'"}, 400

        metadata = {}
        if source:
            metadata["source"] = source
        if full_path:
            metadata["full_path"] = full_path

        try:
            deleted = remove_files_from_knowledgebase(kb, metadata)
        except ValueError as err:
            return {"error": str(err)}, 400
        except Exception:
            err = "unexpected error deleting document(s) from DB"
            log.exception(err)
            return {"error": err}, 500

        count = len(deleted)
        return {"message": f"{count} document(s) deleted", "count": count, "deleted": deleted}, 200
