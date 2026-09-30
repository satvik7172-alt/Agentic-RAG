# """
# Document upload and processing module.
# """

# import os
# import tempfile

# from fastapi import UploadFile, File
# from langchain_community.document_loaders import PyPDFLoader, TextLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# from src.rag.retriever_setup import retriever_chain
# from src.tools.common_tools import enhance_description_with_llm


# def documents(description: str, file: UploadFile = File(...)):
#     """
#     Process and upload a document for RAG.

#     Validates file type, loads content, enhances description, chunks documents,
#     and stores them in the vector database.

#     Args:
#         description: User-provided document description.
#         file: The uploaded file (PDF or TXT).

#     Returns:
#         Boolean indicating success of the upload process.

#     Raises:
#         HTTPException: If file type is not supported or loading fails.
#     """
#     filename = file.filename
#     print(filename)
#     if not filename.endswith(".pdf") and not filename.endswith(".txt"):
#         from fastapi import HTTPException
#         raise HTTPException(
#             status_code=400,
#             detail="Only PDF and TXT files are supported"
#         )

#     file_bytes = file.file.read()

#     with tempfile.NamedTemporaryFile(
#         delete=False,
#         suffix=os.path.splitext(filename)[1]
#     ) as tmp_file:
#         tmp_file.write(file_bytes)
#         tmp_path = tmp_file.name

#     if filename.endswith(".pdf"):
#         loader = PyPDFLoader(tmp_path)
#     else:
#         loader = TextLoader(tmp_path, encoding="utf-8")

#     try:
#         docs = loader.load()
#     except Exception as e:
#         from fastapi import HTTPException
#         raise HTTPException(
#             status_code=500,
#             detail=f"Error loading file: {e}"
#         )
#     finally:
#         os.unlink(tmp_path)

#     # Enhance description using LLM
#     description_llm = enhance_description_with_llm(description)

#     # Save enhanced description
#     with open("description.txt", "w", encoding="utf-8") as f:
#         f.write(description_llm)

#     with open("description.txt", "r", encoding="utf-8") as f:
#         print("Document description from storage:")
#         print(f.read())

#     # Split documents into chunks
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=150
#     )
#     chunks = splitter.split_documents(docs)

#     return retriever_chain(chunks)



"""
Document upload and vector-store initialization.
"""

import os
import tempfile
import traceback

from fastapi import UploadFile, File, HTTPException
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.rag.retriever_setup import retriever_chain


def documents(description: str, file: UploadFile = File(...)):
    filename = file.filename

    print(f"Uploading file: {filename}")

    if not filename.endswith(".pdf") and not filename.endswith(".txt"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF and TXT files are supported"
        )

    tmp_path = None

    try:
        # Read uploaded file
        file_bytes = file.file.read()

        print(f"Uploaded file size: {len(file_bytes)} bytes")

        if not file_bytes:
            raise ValueError("Uploaded file is empty.")

        # Save temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=os.path.splitext(filename)[1]
        ) as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = tmp_file.name

        print(f"Temporary file created: {tmp_path}")

        # Load document
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
        else:
            loader = TextLoader(
                tmp_path,
                encoding="utf-8"
            )

        docs = loader.load()

        print(f"Loaded {len(docs)} document pages")

        if not docs:
            raise ValueError("No content was extracted from the document.")

        # Split document
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )

        chunks = splitter.split_documents(docs)

        print(f"Created {len(chunks)} document chunks")

        if not chunks:
            raise ValueError("No chunks were created from the document.")

        # Store description without calling the LLM
        description = description.strip()

        if not description:
            description = (
                "Use this tool only to answer questions "
                "about the uploaded document."
            )

        with open("description.txt", "w", encoding="utf-8") as f:
            f.write(description)

        print("Document description saved.")

        # Create FAISS vectorstore
        status = retriever_chain(chunks)

        print(f"FAISS upload status: {status}")

        if not status:
            raise RuntimeError(
                "Failed to create FAISS vectorstore."
            )

        return True

    except HTTPException:
        raise

    except Exception as e:
        print("\n========== DOCUMENT UPLOAD ERROR ==========")
        print(f"Error type: {type(e).__name__}")
        print(f"Error: {e}")
        traceback.print_exc()
        print("===========================================\n")

        raise HTTPException(
            status_code=500,
            detail=f"Document upload failed: {type(e).__name__}: {e}"
        )

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
            print("Temporary file deleted.")
