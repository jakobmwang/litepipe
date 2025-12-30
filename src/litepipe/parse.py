# src/litepipe/parse.py
"""
Archive/audio/document/nested/tabular parsing and markdown extraction.
"""
from .blobstorage import BlobStorage
import asyncio
import base64
import clevercsv
import filetype
import gzip
import httpx
import io
import logging
import math
import os
import polars as pl
import tarfile
import zipfile
from typing import BinaryIO


logger = logging.getLogger()
storage = BlobStorage()
parsing_semaphore = asyncio.Semaphore(5) # (NOT IMPLEMENTED)


async def parse_from_key(file_key: str, use_cache: bool = True) -> list[dict] | None:
    """
    Initiates parsing from storage key, leveraging cache.
    Supports archives, audio/video, tabular and documents.
    Returns a list of any markdown that was extracted in the format:
        [{'file_name': 'document.pdf', 'content_type': 'application/pdf', 'markdown': '...'}]
    """
    if use_cache:
        cache = await parse_from_cache(file_key)
        if cache:
            return cache
    file_dict = await storage.get("document-cache", file_key)
    if file_dict is None:
        logger.error(f"File not found in storage for key {file_key}")
        return None
    parsed = await parse_to_markdown(file_dict['file_bytes'], file_dict['file_name'], use_cache)
    parsed_keys = await parse_to_cache(parsed)
    parsed = [{**d, "file_key": k} for d, k in zip(parsed, parsed_keys)]
    return parsed


async def parse_from_bytes(file_bytes: bytes | BinaryIO, file_name: str, use_cache: bool = True) -> list[dict]:
    """
    Initiates parsing from file bytes and name, leveraging cache.
    Supports archives, audio/video, tabular and documents.
    Returns a list of any markdown that was extracted in the format:
        [{'file_name': 'document.pdf', 'content_type': 'application/pdf', 'markdown': '...'}]
    """
    file_bytes = file_bytes.read() if isinstance(file_bytes, BinaryIO) else file_bytes
    file_key = await storage.key(file_bytes)
    if use_cache:
        cache = await parse_from_cache(file_key)
        if cache:
            return cache
    parsed = await parse_to_markdown(file_bytes, file_name, use_cache)
    parsed_keys = await parse_to_cache(parsed)
    parsed = [{**d, "file_key": k} for d, k in zip(parsed, parsed_keys)]
    return parsed
    

async def parse_from_cache(file_key: str) -> list[dict] | None:
    """
    Retrieves cached markdown from given file key.
    """
    cache = await storage.get("markdown-cache", file_key)
    if cache:
        return [{
            'file_name': cache['file_name'],
            'content_type': cache['content_type'],
            'markdown': cache['file_bytes'].decode('utf-8')
        }]
    return None


async def parse_to_cache(parsed: list[dict]) -> list:
    """
    Caches parsed markdown.
    """
    file_keys = [
        await storage.put(
            'markdown-cache',
            p['markdown'].encode('utf-8'),
            p['file_name'],
            p['content_type']
        ) for p in parsed
    ]
    return file_keys


async def parse_to_markdown(file_bytes: bytes, file_name: str, use_cache: bool = True) -> list[dict]:
    """
    Parses most common files and returns markdown.
    Supports archives, audio/video, tabular and documents.
    Returns a list of any markdown that was extracted in the format:
        [{'file_name': 'document.pdf', 'content_type': 'application/pdf', 'markdown': '...'}]
    """
    file_type = filetype.guess(file_bytes)
    if file_type is not None:

        # Test for archive, recursively unarchive and parse -> markdown
        if file_type.extension in ('zip', 'tar', 'gzip'):
            return [
                parsed_item
                for item in await unarchive(file_bytes, file_name)
                for parsed_item in await parse_from_bytes(item['file_bytes'], item['file_name'], use_cache)
            ]
        
        # Test for audio (or video), normalize and perform ASR -> markdown
        if file_type.mime.startswith('audio/') or file_type.mime.startswith('video/'):
            #audio_bytes = await normalize_audio(file_bytes, file_name)  # DONT DO normalize_audio, already included in whisper container
            markdown = await audio_to_markdown(file_bytes, file_name, file_type.mime)
            return [{'file_name': file_name, 'content_type': file_type.mime, 'markdown': markdown}]
        
        # Test for image, normalize and perform OCR -> markdown
        if file_type.mime.startswith('image/') and file_type.extension not in ('jpg', 'png'):
            image_bytes = await normalize_image(file_bytes, file_name)
            markdown = await document_to_markdown(image_bytes, file_name)
            return [{'file_name': file_name, 'content_type': file_type.mime, 'markdown': markdown}]
        
        # Test for common document types
        if file_type.extension in ('pdf', 'docx', 'xlsx', 'pptx'):
            markdown = await document_to_markdown(file_bytes, file_name)
            return [{'file_name': file_name, 'content_type': file_type.mime, 'markdown': markdown}]

    # Test for Parquet
    markdown = await parquet_to_markdown(file_bytes, file_name)
    if markdown:
        return [{'file_name': file_name, 'content_type': 'application/vnd.apache.parquet', 'markdown': markdown}]

    # Ensure normalized text before testing for possible text formats
    file_bytes = await normalize_text(file_bytes)

    # Test for JSON
    markdown = await json_to_markdown(file_bytes, file_name)
    if markdown:
        return [{'file_name': file_name, 'content_type': 'application/json', 'markdown': markdown}]

    # Test for CSV
    markdown = await csv_to_markdown(file_bytes, file_name)
    if markdown:
        return [{'file_name': file_name, 'content_type': 'text/csv', 'markdown': markdown}]

    # Catch all, likely HTML
    markdown = await document_to_markdown(file_bytes, file_name)
    return [{'file_name': file_name, 'content_type': 'text/html', 'markdown': markdown}]


async def unarchive(file_bytes: bytes, file_name: str) -> list[dict]:
    """
    Will unarchive zip, tar and gzip bytes.
    Returns a list of any markdown that was extracted in the format:
        [{'file_bytes': '...', 'file_name': 'document.pdf'}]
    """
    file_ext = filetype.guess_extension(file_bytes)
    extracted = []
    if file_ext == 'zip':
        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
                for name in z.namelist():
                    if not z.getinfo(name).is_dir():  # Files only
                        extracted.append({
                            'file_bytes': z.read(name),
                            'file_name': name,
                        })
        except Exception as e:
            logger.warning(f"Failed unarchiving {file_name}: {e}")
            pass
    elif file_ext == 'tar':
        try:
            with tarfile.open(fileobj=io.BytesIO(file_bytes)) as t:
                for member in t:
                    if member.isfile():  # Files only
                        f = t.extractfile(member)
                        if f:
                            extracted.append({
                                'file_bytes': f.read(),
                                'file_name': member.name,
                            })
        except Exception as e:
            logger.warning(f"Failed unarchiving {file_name}: {e}")
            pass
    elif file_ext == 'gz':
        try:
            extracted = [{
                'file_bytes': gzip.decompress(file_bytes),
                'file_name': file_name,
            }]
        except Exception as e:
            logger.warning(f"Failed unarchiving {file_name}: {e}")
            pass
    return extracted


async def parquet_to_markdown(file_bytes: bytes, file_name: str) -> str:
    """
    Attempts to open file as Parquet in Polars and returns markdown.
    """
    try: # Parquet?
        df = pl.read_parquet(io.BytesIO(file_bytes))
    except:
        return ""
    return await df_to_markdown(df)


async def json_to_markdown(file_bytes: bytes, file_name: str) -> str:
    """
    Attempts to open file as JSON in Polars and returns markdown.
    """
    try: # Regular JSON?
        df = pl.read_json(io.BytesIO(file_bytes))
    except:
        try: # NDJSON?
            df = pl.read_ndjson(io.BytesIO(file_bytes))
        except:
            return ""
    return await df_to_markdown(df)


async def csv_to_markdown(file_bytes: bytes, file_name: str) -> str:
    """
    Attempts to detect and open file as CSV in polars and returns markdown.
    """
    try:
        # Decode and detect dialect (delimiter, quotechar, escapechar)
        text = file_bytes.decode('utf-8', errors='replace')
        sniffer = clevercsv.Sniffer()
        dialect = sniffer.sniff(text, verbose=False)
        if dialect is None or dialect.delimiter is None:
            return ""
        # Detect if file has header
        has_header = sniffer.has_header(text)
        # Map CleverCSV dialect to Polars parameters ('' -> None)
        quote_char = dialect.quotechar if dialect.quotechar != '' else None
        # Read CSV with Polars using detected dialect
        df = pl.read_csv(
            io.BytesIO(file_bytes),
            separator=dialect.delimiter,
            quote_char=quote_char,
            has_header=has_header,
            infer_schema_length=10000,   # Good balance of speed/accuracy
            ignore_errors=True,          # Skip malformed rows
            truncate_ragged_lines=True,  # Handle uneven row lengths
        )
        return await df_to_markdown(df)
    except Exception:
        return ""


async def df_to_markdown(df: pl.DataFrame) -> str:
    """
    Returns df flattened and exported to markdown.
    """
    # Flatten structs and stringify lists (drop primitive lists of len > 4)
    while any(isinstance(dtype, pl.Struct) or isinstance(dtype, pl.List) for dtype in df.dtypes):
        for col in df.columns:
            # Flatten structs
            if isinstance(df[col].dtype, pl.Struct):
                unnested = df[col].struct.unnest()
                unnested = unnested.rename({c: f"{col}.{c}" for c in unnested.columns})
                df = df.drop(col).hstack(unnested)
            elif isinstance(df[col].dtype, pl.List):
                # Drop primitive lists of len > 4
                prim_type = df[col].dtype.inner.is_numeric() or df[col].dtype.inner == pl.Boolean # type: ignore
                max_len_4 = (df[col].list.len().max() or 0) > 4 # type: ignore
                if prim_type and max_len_4:
                    df = df.drop(col)
                # Stringify remaining lists
                else:
                    df = df.with_columns(pl.col(col).list.join(", "))
    return df.to_pandas().to_markdown(index=False)


async def document_to_markdown(file_bytes: bytes, file_name: str) -> str:
    """
    Converts documents to markdown using docling-serve.
    Returns document as markdown, or empty string if conversion fails.
    """
    try:
        endpoint = os.environ["DOCLING_URL"].rstrip("/") + "/v1/convert/source"
        headers = {
            "X-Api-Key": os.environ["DOCLING_API_KEY"],
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        payload = {
            "sources": [{
                "kind": "file",
                "base64_string": base64.b64encode(file_bytes).decode('utf-8'),
                "filename": file_name
            }],
            "options": {
                "to_formats": ["md"],
                "do_ocr": True,
                "ocr_lang": ["da", "en"],
                "ocr_engine": "suryaocr"
            },
        }
        async with httpx.AsyncClient() as client:
            r = await client.post(endpoint, headers=headers, json=payload, timeout=180)
            r.raise_for_status()
            data = r.json()
        md = (data.get("document") or {}).get("md_content")
        md = md.strip() if isinstance(md, str) else ""
        if not md:
            logger.warning(f"Document conversion returned empty content for {file_name}")
        return md
    except httpx.HTTPStatusError as e:
        logger.warning(f"HTTP {e.response.status_code} converting {file_name}: {e.response.text}")
        return ""
    except Exception as e:
        logger.warning(f"Failed converting document to markdown {file_name}: {e}")
        return ""


async def audio_to_markdown(file_bytes: bytes, file_name: str, file_type: str) -> str:
    """
    Transcribes audio/video using Whisper and returns markdown table.
    """
    try:
        endpoint = os.environ.get("WHISPER_URL", "http://whisper:8000").rstrip("/")
        
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(
                f"{endpoint}/v1/audio/transcriptions",
                files={'file': (file_name, file_bytes, file_type)},
                data={
                    'response_format': 'verbose_json',
                    #'vad_filter': 'true', # causes hallucinations :(
                    #'language': 'da', # too restrictive, auto-detect is better
                },
            )
            response.raise_for_status()
            result = response.json()
        
        segments = result.get('segments', [])
        if not segments:
            logger.warning(f"No segments returned from Whisper for {file_name}")
            return ""
        
        # Merge segments adaptively to minute boundaries
        merged = merge_to_adaptive_minutes(segments, min_duration=30.0)
        
        # Build markdown table
        rows = [
            f"| {format_timestamp(seg['start'])} - {format_timestamp(seg['end'])} | {seg['text'].strip()} |"
            for seg in merged
        ]
        
        header = "| Tidsstempel | Tekstsegment |\n|-------------|--------------|"
        return header + "\n" + "\n".join(rows)
        
    except httpx.HTTPError as e:
        logger.warning(f"HTTP error transcribing {file_name}: {e}")
        return ""
    except Exception as e:
        logger.warning(f"Failed transcribing audio {file_name}: {e}")
        return ""


def merge_to_adaptive_minutes(segments: list[dict], min_duration: float = 30.0) -> list[dict]:
    """
    Merges Whisper segments adaptively:
    - Always merge until exceeding next whole minute from buffer start
    - After flush, next buffer must be min_duration AND exceed the next whole minute after that
    """
    
    merged = []
    buffer_segments = []
    buffer_start = 0.0
    
    for seg in segments:
        buffer_segments.append(seg)
        buffer_end = seg['end']
        buffer_duration = buffer_end - buffer_start
        
        # Next whole minute from buffer start
        next_minute_from_start = math.ceil(buffer_start / 60.0) * 60.0
        
        # Have we exceeded that minute?
        if buffer_end > next_minute_from_start:
            # Do we also have minimum duration?
            if buffer_duration >= min_duration:
                # What's the next whole minute after (start + min_duration)?
                min_end = buffer_start + min_duration
                next_minute_after_min = math.ceil(min_end / 60.0) * 60.0
                
                # Have we exceeded that minute too?
                if buffer_end >= next_minute_after_min:
                    # Flush buffer
                    merged_seg = {
                        'start': buffer_start,
                        'end': buffer_end,
                        'text': ' '.join(s['text'].strip() for s in buffer_segments)
                    }
                    merged.append(merged_seg)
                    
                    # Reset for next buffer
                    buffer_start = buffer_end
                    buffer_segments = []
    
    # Flush remaining
    if buffer_segments:
        buffer_end = buffer_segments[-1]['end']
        merged_seg = {
            'start': buffer_start,
            'end': buffer_end,
            'text': ' '.join(s['text'].strip() for s in buffer_segments)
        }
        merged.append(merged_seg)
    
    return merged


def format_timestamp(seconds: float) -> str:
    """
    Formats seconds to HH:MM:SS timestamp.
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
 

async def normalize_text(file_bytes: bytes) -> bytes:
    """
    If given text bytes, will encourage utf-8 and \n EOL.
    Returns bytes.
    """
    for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
        try:
            text = file_bytes.decode(encoding)
            text = text.replace('\r\n', '\n').replace('\r', '\n')
            return text.encode('utf-8')
        except:
            continue
    return file_bytes


async def normalize_audio(file_bytes: bytes, file_name: str) -> bytes:
    """
    Accepts (almost) any audio or video format and returns WAV16 bytes.
    """
    cmd = [
        'ffmpeg',
        '-hide_banner',
        '-nostdin',
        '-i', 'pipe:0',                     # Read input from stdin
        '-vn',                              # Discard video
        '-acodec', 'pcm_s16le',             # Audio Codec: PCM 16-bit (WAV standard)
        '-ar', '16000',                     # Audio Rate: 16 kHz (Whisper standard)
        '-channel_layout', 'mono',          # Mono layout explicited
        '-ac', '1',                         # Audio Channels: 1 (Mono)
        '-sample_fmt', 's16',               # Sample format explicited
        '-af', 'aresample=resampler=soxr',  # Soxr resampler
        '-f', 'wav',                        # Format: WAV
        'pipe:1'                            # Write to stdout
    ]
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate(input=file_bytes)
        if process.returncode != 0:
            error_msg = stderr.decode('utf-8', errors='ignore')
            logger.warning(f"ffmpeg failed for {file_name} (code {process.returncode}): {error_msg}")
            return b""
        return stdout
    except Exception as e:
        logger.warning(f"Failed normalizing audio {file_name}: {e}")
        return b""
    

async def normalize_image(file_bytes: bytes, file_name: str) -> bytes:
    """
    Accepts (almost) any image format and returns PNG bytes.
    """
    cmd = [
        'ffmpeg',
        '-hide_banner',
        '-nostdin',
        '-i', 'pipe:0',                     # Read from stdin
        '-map', '0:v:0',                    # Get first stream (video)
        '-frames:v', '1',                   # Get 1 frame only (for GIF/video)
        '-f', 'image2pipe',                 # Send image-bytes to pipe
        '-vcodec', 'png',                   # Output as PNG
        'pipe:1'                            # Write to stdout
    ]
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate(input=file_bytes)
        if process.returncode != 0:
            error_msg = stderr.decode('utf-8', errors='ignore')
            logger.warning(f"ffmpeg failed for {file_name} (code {process.returncode}): {error_msg}")
            return b""
        return stdout
    except Exception as e:
        logger.warning(f"Failed normalizing image {file_name}: {e}")
        return b""