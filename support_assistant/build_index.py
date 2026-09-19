from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "zepto_policies"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# --------------------------------------------------
# Chunking
# --------------------------------------------------

def chunk_text(text: str, max_chars: int = 500) -> list[str]:
    """
    Split a document into paragraph-based chunks.

    Paragraphs are combined until the chunk reaches
    approximately max_chars characters.
    """
    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n")
        if paragraph.strip()
    ]

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        if not current_chunk:
            current_chunk = paragraph
        elif len(current_chunk) + len(paragraph) + 1 <= max_chars:
            current_chunk += " " + paragraph
        else:
            chunks.append(current_chunk)
            current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# --------------------------------------------------
# Load embedding model
# --------------------------------------------------

print("Loading embedding model...")
model = SentenceTransformer(EMBEDDING_MODEL)

print("Embedding model loaded.")


# --------------------------------------------------
# Connect to ChromaDB
# --------------------------------------------------

print("Creating/opening ChromaDB...")

client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

# Recreate the collection so the index contains
# only the current chunked documents.
try:
    client.delete_collection(COLLECTION_NAME)
except Exception:
    pass

collection = client.create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)

print(f"ChromaDB collection: {COLLECTION_NAME}")


# --------------------------------------------------
# Load and chunk documents
# --------------------------------------------------

documents = []
document_ids = []
metadatas = []

source_files = sorted(DOCS_DIR.glob("*.txt"))

if len(source_files) != 8:
    raise ValueError(
        f"Expected 8 documents, but found {len(source_files)}."
    )

for file_path in source_files:

    text = file_path.read_text(
        encoding="utf-8"
    ).strip()

    if not text:
        print(f"Skipping empty file: {file_path.name}")
        continue

    chunks = chunk_text(text)

    print(
        f"{file_path.name}: {len(chunks)} chunks"
    )

    for chunk_number, chunk in enumerate(chunks, start=1):

        documents.append(chunk)

        document_ids.append(
            f"{file_path.stem}_chunk_{chunk_number}"
        )

        metadatas.append(
            {
                "source": file_path.name,
                "chunk": chunk_number,
            }
        )


print(f"Total chunks: {len(documents)}")


# --------------------------------------------------
# Generate embeddings
# --------------------------------------------------

print("Generating embeddings...")

embeddings = model.encode(
    documents,
    normalize_embeddings=True,
).tolist()

print("Embeddings generated.")


# --------------------------------------------------
# Store chunks in ChromaDB
# --------------------------------------------------

print("Storing chunks in ChromaDB...")

collection.add(
    ids=document_ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas,
)

print("Chunks successfully stored.")


# --------------------------------------------------
# Verification
# --------------------------------------------------

stored = collection.count()

print()
print("=" * 50)
print("INDEX BUILD COMPLETE")
print("=" * 50)
print(f"Source documents: {len(source_files)}")
print(f"Chunks stored: {stored}")
print(f"Collection: {COLLECTION_NAME}")
print(f"Database path: {CHROMA_DIR}")
print("=" * 50)