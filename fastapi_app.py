from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from gradio_client import Client, handle_file
import base64
import os
import uuid
import shutil
from io import BytesIO
from PIL import Image

app = FastAPI(title="Face Aging Proxy API")

# Initialize the Gradio client
gradio_client = Client("Robys01/Face-Aging")

class AgingRequest(BaseModel):
    image_base64: str
    source_age: int = 20  # Ignored by Gradio image endpoint, but kept for compatibility
    target_age: int

class AgingResponse(BaseModel):
    image_base64: str

@app.post("/generate", response_model=AgingResponse)
def generate_aged_face(req: AgingRequest):
    try:
        # 1. Decode base64 image
        image_bytes = base64.b64decode(req.image_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid base64 encoding")

    # Create a temporary file to save the input image
    temp_id = str(uuid.uuid4())
    input_path = f"temp_input_{temp_id}.jpg"
    
    try:
        # Save input image to disk
        with open(input_path, "wb") as f:
            f.write(image_bytes)
            
        # 2. Call Gradio API
        # The /predict endpoint takes image_path and target_age
        print(f"Sending request to Gradio for target_age: {req.target_age}")
        result_path = gradio_client.predict(
            image_path=handle_file(input_path),
            target_age=req.target_age,
            api_name="/predict"
        )
        print(f"Received result from Gradio: {result_path}")
        
        if not result_path or not os.path.exists(result_path):
            raise Exception("Gradio API did not return a valid image path")

        # 3. Read the output image (.webp or .jpg) and convert to base64
        # We will convert it to standard JPEG to ensure compatibility with Kotlin
        img = Image.open(result_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
            
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=90)
        result_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return AgingResponse(image_base64=result_b64)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal processing error: {str(e)}")
        
    finally:
        # Cleanup temporary input file
        if os.path.exists(input_path):
            os.remove(input_path)

@app.get("/")
def read_root():
    return {"message": "Face Aging Proxy API is running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
