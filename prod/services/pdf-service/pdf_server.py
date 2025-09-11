"""
PDF Service Server
Отдельный сервис для обработки PDF файлов
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import shutil
import logging
import os
import time
from pathlib import Path
from pdf_text_extractor import extract_pdf_text, DocumentParsingError, OCRConfigError

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Создание FastAPI приложения
app = FastAPI(
    title="PDF Text Extraction Service",
    description="Сервис для извлечения текста из PDF файлов",
    version="1.0.0"
)

# Создание директории для загрузок
UPLOAD_DIR = Path("/app/uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.get("/")
async def root():
    """Корневой эндпоинт"""
    return {
        "service": "PDF Text Extraction Service",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "pdf-service"
    }

@app.post("/pdf/extract")
async def extract_pdf_endpoint(
    file: UploadFile = File(...),
    min_chars_for_digital: int = Form(400),
    ocr_engine: str = Form("tesseract"),  # "paddle" or "tesseract"
    ocr_lang: str = Form("rus+eng"),
    dpi: int = Form(300),
    return_format: str = Form("dict")  # "dict" or "list"
):
    """
    Извлечь текст из PDF файла
    
    Parameters:
    - file: PDF файл для обработки
    - min_chars_for_digital: Минимальное количество символов для определения цифровой страницы
    - ocr_engine: Движок OCR ("paddle" или "tesseract")
    - ocr_lang: Язык для OCR (для paddle: "ru", "en"; для tesseract: "rus+eng")
    - dpi: Разрешение для рендеринга (только для сканированных PDF)
    - return_format: Формат возврата ("dict" или "list")
    
    Returns:
    - JSON с извлеченным текстом по страницам
    """
    
    # Проверка типа файла
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Файл должен быть в формате PDF")
    
    # Создание временного файла
    temp_path = None
    try:
        # Сохраняем загруженный файл в temp директории
        temp_path = f"/app/temp/pdf_{os.getpid()}_{hash(file.filename)}_{int(time.time())}.pdf"
        
        with open(temp_path, 'wb') as temp_file:
            shutil.copyfileobj(file.file, temp_file)
        
        logger.info(f"Processing PDF: {file.filename} with {ocr_engine} engine")
        
        # Извлекаем текст
        result = extract_pdf_text(
            path=temp_path,
            min_chars_for_digital=min_chars_for_digital,
            ocr_engine=ocr_engine,
            ocr_lang=ocr_lang,
            dpi=dpi,
            return_format=return_format
        )
        
        # Debug logging
        logger.info(f"Extract result type: {type(result)}")
        logger.info(f"Extract result length: {len(result) if hasattr(result, '__len__') else 'N/A'}")
        if result and hasattr(result, '__getitem__'):
            logger.info(f"First item type: {type(result[0])}")
            if hasattr(result[0], 'get'):
                logger.info(f"First item keys: {list(result[0].keys()) if hasattr(result[0], 'keys') else 'N/A'}")

        # Handle different return formats
        if return_format == "list":
            # result is a list of strings
            pages_data = result
            extraction_method = "unknown"  # We don't have metadata in list format
            ocr_engine_used = None
        else:
            # result should be a list of dicts: [{"page": N, "source": "digital"|"scan", "text": "..."}]
            if not result:
                pages_data = []
                extraction_method = "unknown"
                ocr_engine_used = None
            else:
                pages_data = result
                # Determine extraction method from first page
                first_page = result[0] if result else {}
                if isinstance(first_page, dict):
                    extraction_method = "digital" if first_page.get("source") == "digital" else "ocr"
                    ocr_engine_used = ocr_engine if first_page.get("source") == "scan" else None
                else:
                    # Fallback if result format is unexpected
                    extraction_method = "unknown"
                    ocr_engine_used = None

        # Добавляем метаинформацию
        response_data = {
            "filename": file.filename,
            "pages_count": len(pages_data),
            "extraction_method": extraction_method,
            "ocr_engine": ocr_engine_used,
            "data": pages_data
        }
        
        logger.info(f"Successfully processed {file.filename}: {len(result)} pages")
        return JSONResponse(content=response_data)
        
    except DocumentParsingError as e:
        logger.error(f"Document parsing error for {file.filename}: {e}")
        raise HTTPException(status_code=400, detail=f"Ошибка обработки документа: {str(e)}")
    except OCRConfigError as e:
        logger.error(f"OCR config error: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка конфигурации OCR: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error processing {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")
    finally:
        # Удаляем временный файл
        if temp_path and Path(temp_path).exists():
            try:
                Path(temp_path).unlink()
            except Exception as e:
                logger.warning(f"Could not delete temp file {temp_path}: {e}")

@app.get("/pdf/info")
async def pdf_info():
    """
    Информация о доступных параметрах PDF экстрактора
    """
    return {
        "available_ocr_engines": ["paddle", "tesseract"],
        "supported_languages": {
            "paddle": ["ru", "en", "ch", "fr", "de", "ja", "ko"],
            "tesseract": ["rus+eng", "rus", "eng", "deu", "fra", "spa"]
        },
        "default_parameters": {
            "min_chars_for_digital": 400,
            "ocr_engine": "tesseract",
            "ocr_lang": "rus+eng",
            "dpi": 300,
            "return_format": "dict"
        },
        "return_formats": {
            "dict": "Массив объектов с метаданными: [{page, source, text}, ...]",
            "list": "Простой массив текстов: ['text1', 'text2', ...]"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4300)