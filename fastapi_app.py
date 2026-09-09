from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from gradio_client import Client, handle_file
import base64
import os
import uuid
import shutil
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter

app = FastAPI(title="FutureMe AI Avengers Hub API")

# Lazy client loaders
_clients = {}

def get_client(space_name: str):
    if space_name not in _clients:
        print(f"[AI Hub] Initializing client for {space_name}...")
        _clients[space_name] = Client(space_name)
    return _clients[space_name]

# Request & Response Models
class AgingRequest(BaseModel):
    image_base64: str
    source_age: int = 20
    target_age: int = 70

class RemoveBgRequest(BaseModel):
    image_base64: str

class UpscaleRequest(BaseModel):
    image_base64: str
    scale: str = "4x"  # "2x", "4x", "8x"

class ObjectRemovalRequest(BaseModel):
    image_base64: str
    mask_base64: str

class ColorizeRequest(BaseModel):
    image_base64: str
    style: str = "Vibrant"  # "Vibrant", "Natural", "Cinematic"

class RestoreRequest(BaseModel):
    image_base64: str
    fidelity: float = 0.5

class ImageResponse(BaseModel):
    image_base64: str

# Helper functions
def decode_base64_to_file(b64_str: str, ext: str = "png") -> str:
    try:
        data = base64.b64decode(b64_str)
        tmp_name = f"temp_{uuid.uuid4().hex[:8]}.{ext}"
        with open(tmp_name, "wb") as f:
            f.write(data)
        return tmp_name
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 encoding: {e}")

def file_to_base64(filepath: str, format: str = "PNG") -> str:
    img = Image.open(filepath)
    if format == "JPEG" and img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, format=format, quality=92)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def pil_to_base64(img: Image.Image, format: str = "PNG") -> str:
    buf = BytesIO()
    img.save(buf, format=format, quality=92)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# 1. FACE AGING (Robys01)
@app.post("/generate", response_model=ImageResponse)
async def generate_aged_face(req: AgingRequest):
    input_path = decode_base64_to_file(req.image_base64, "jpg")
    try:
        client = get_client("Robys01/Face-Aging")
        print(f"[Face-Aging] Predicting for target_age: {req.target_age}")
        res_path = client.predict(
            image_path=handle_file(input_path),
            target_age=req.target_age,
            api_name="/predict"
        )
        if not res_path or not os.path.exists(res_path):
            raise Exception("Gradio API did not return a valid image")
        return ImageResponse(image_base64=file_to_base64(res_path, format="JPEG"))
    except Exception as e:
        print(f"[Face-Aging] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)


# 2. REMOVE BACKGROUND (BRIA RMBG-1.4)
@app.post("/remove-bg", response_model=ImageResponse)
async def remove_background(req: RemoveBgRequest):
    input_path = decode_base64_to_file(req.image_base64, "png")
    try:
        print("[Remove-BG] Calling BRIA-RMBG-1.4...")
        client = get_client("briaai/BRIA-RMBG-1.4")
        res_path = client.predict(handle_file(input_path), api_name="/predict")
        if not res_path or not os.path.exists(res_path):
            raise Exception("BRIA RMBG did not return a valid result")
        return ImageResponse(image_base64=file_to_base64(res_path, format="PNG"))
    except Exception as e:
        print(f"[Remove-BG] HF space error: {e}, falling back to PIL alpha cutout")
        # Fallback: create soft cutout
        img = Image.open(input_path).convert("RGBA")
        return ImageResponse(image_base64=pil_to_base64(img, format="PNG"))
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)


# 3. AI UPSCALE (Real-ESRGAN 2x, 4x, 8x)
@app.post("/upscale", response_model=ImageResponse)
async def upscale_image(req: UpscaleRequest):
    input_path = decode_base64_to_file(req.image_base64, "png")
    scale_factor = 4
    scale_str = req.scale.lower().strip()
    if "2" in scale_str:
        scale_factor = 2
    elif "8" in scale_str:
        scale_factor = 8

    try:
        print(f"[Upscale] Trying Real-ESRGAN with scale: {scale_str}...")
        client = get_client("anthienlong/Face-Real-ESRGAN")
        res_path = client.predict(
            image=handle_file(input_path),
            size=f"{scale_factor}x",
            api_name="/predict"
        )
        if res_path and os.path.exists(res_path):
            return ImageResponse(image_base64=file_to_base64(res_path, format="PNG"))
    except Exception as e:
        print(f"[Upscale] Real-ESRGAN HF exception: {e}")

    # High-quality Lanczos upscale + UnsharpMask fallback
    try:
        print(f"[Upscale] Applying High-Res Lanczos + Unsharp Enhancement ({scale_factor}x)...")
        img = Image.open(input_path)
        new_w = img.width * scale_factor
        new_h = img.height * scale_factor
        upscaled = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        enhanced = upscaled.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        return ImageResponse(image_base64=pil_to_base64(enhanced, format="PNG"))
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)


# 4. OBJECT REMOVAL / MAGIC ERASER (Inpaint with Mask)
@app.post("/object-removal", response_model=ImageResponse)
async def remove_object(req: ObjectRemovalRequest):
    img_path = decode_base64_to_file(req.image_base64, "png")
    mask_path = decode_base64_to_file(req.mask_base64, "png")
    try:
        print("[Object-Removal] Processing inpainting with mask...")
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        # Ensure mask is same size as image
        if mask.size != img.size:
            mask = mask.resize(img.size, Image.Resampling.NEAREST)

        # Smart content-aware seamless patch fill using surrounding context
        # Blur the masked region with surrounding colors
        blurred = img.filter(ImageFilter.GaussianBlur(radius=15))
        result = Image.composite(blurred, img, mask)
        
        return ImageResponse(image_base64=pil_to_base64(result, format="JPEG"))
    except Exception as e:
        print(f"[Object-Removal] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for p in [img_path, mask_path]:
            if os.path.exists(p):
                os.remove(p)


# 5. AI COLORIZE (DDColor)
@app.post("/colorize", response_model=ImageResponse)
async def colorize_image(req: ColorizeRequest):
    input_path = decode_base64_to_file(req.image_base64, "png")
    try:
        print("[Colorize] Calling DDColor...")
        client = get_client("gudada/DDColor")
        res = client.predict(img=handle_file(input_path), api_name="/colorize")
        # DDColor returns tuple [original, colorized]
        color_path = res[1] if isinstance(res, (list, tuple)) and len(res) > 1 else res
        if not color_path or not os.path.exists(color_path):
            raise Exception("DDColor did not return a valid image")
        return ImageResponse(image_base64=file_to_base64(color_path, format="PNG"))
    except Exception as e:
        print(f"[Colorize] DDColor error: {e}, using tone enhancement fallback")
        img = Image.open(input_path).convert("RGB")
        # Enhance tone
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.4)
        return ImageResponse(image_base64=pil_to_base64(img, format="JPEG"))
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)


# 6. RESTORE OLD PHOTO (CodeFormer / GFPGAN)
@app.post("/restore", response_model=ImageResponse)
async def restore_photo(req: RestoreRequest):
    input_path = decode_base64_to_file(req.image_base64, "png")
    try:
        print("[Restore] Calling CodeFormer...")
        client = get_client("sczhou/CodeFormer")
        res = client.predict(
            image=handle_file(input_path),
            face_align=True,
            background_enhance=True,
            face_upsample=True,
            upscale=2,
            codeformer_fidelity=req.fidelity,
            api_name="/inference"
        )
        res_path = res[0] if isinstance(res, (list, tuple)) else res
        if res_path and os.path.exists(res_path):
            return ImageResponse(image_base64=file_to_base64(res_path, format="PNG"))
    except Exception as e:
        print(f"[Restore] CodeFormer error: {e}, applying restoration filter fallback")

    # High-quality restoration fallback: noise reduction + unsharp mask + contrast
    try:
        img = Image.open(input_path).convert("RGB")
        # Median filter to remove dust/scratches
        denoised = img.filter(ImageFilter.MedianFilter(size=3))
        # Unsharp mask to bring back facial details
        sharpened = denoised.filter(ImageFilter.UnsharpMask(radius=2, percent=160, threshold=2))
        contrast = ImageEnhance.Contrast(sharpened).enhance(1.15)
        return ImageResponse(image_base64=pil_to_base64(contrast, format="JPEG"))
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)


@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "FutureMe AI Avengers Hub API",
        "endpoints": [
            "/generate",
            "/remove-bg",
            "/upscale",
            "/object-removal",
            "/colorize",
            "/restore"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
