from sentence_transformers import SentenceTransformer
from database import SessionLocal
from models import ProjectChunk
from chunk import chunks

db = SessionLocal()

model = SentenceTransformer("all-MiniLM-L6-v2")

try:
    for chunk in chunks:
        embedding = model.encode(chunk).tolist()

        data = ProjectChunk(
            chunk=chunk,
            embedding=embedding
        )

        db.add(data)

    db.commit()
    print("Chunks stored successfully!")

except Exception as e:
    db.rollback()
    print("Error:", e)

finally:
    db.close()