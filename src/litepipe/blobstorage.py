# src/litepipe/blobstorage.py
"""
Async S3/MinIO blob storage with fan-out content-addressable SHA256 keys.
"""
import aioboto3
import hashlib
import logging
import os
from botocore.exceptions import ClientError
from typing import BinaryIO, Any

BUCKETS = ["document-cache", "markdown-cache", "chunk-cache"]

logger = logging.getLogger()


class BlobStorage:
    """
    Content-addressable blob storage using S3/MinIO.
    """

    def __init__(self):
        self.session = aioboto3.Session()


    def _client(self) -> Any:
        """
        Return S3 client context manager.
        """
        return self.session.client(
            "s3",
            endpoint_url=os.getenv("S3_ENDPOINT_URL", "http://localhost:9000"),
            aws_access_key_id=os.getenv("S3_ACCESS_KEY", "litepipe"),
            aws_secret_access_key=os.getenv("S3_SECRET_KEY", "__CHANGE_ME__"),
            region_name=os.getenv("S3_REGION", "eu-north-1")
        )


    async def ensure_buckets(self) -> None:
        """
        Idempotently create buckets if they do not exist.
        """
        async with self._client() as s3:
            for bucket in BUCKETS:
                try:
                    await s3.head_bucket(Bucket=bucket)
                except ClientError:
                    logger.info(f"Creating bucket: {bucket}")
                    try:
                        await s3.create_bucket(Bucket=bucket)
                    except ClientError as e:
                        logger.error(f"Failed to create {bucket}: {e}")


    async def key(self, file_bytes: bytes | BinaryIO) -> str:
        """
        Generates CAS key from fanned out SHA256 of file bytes.
        """
        file_bytes = file_bytes.read() if isinstance(file_bytes, BinaryIO) else file_bytes
        hash = hashlib.sha256(file_bytes).hexdigest()
        return f"{hash[:2]}/{hash[2:4]}/{hash[4:]}"


    async def put(
        self,
        bucket: str,
        file_bytes: bytes | BinaryIO,
        file_name: str = "Untitled",
        content_type: str = "application/octet-stream"
    ) -> str:
        """
        Store file bytes and return CAS key.
        Optionally store original file name as metadata.
        """
        file_bytes = file_bytes.read() if isinstance(file_bytes, BinaryIO) else file_bytes
        key = await self.key(file_bytes)
        metadata = {'file_name': file_name}
        async with self._client() as s3:
            await s3.put_object(
                Bucket=bucket,
                Key=key,
                Body=file_bytes,
                ContentType=content_type,
                Metadata=metadata
            )
        return key


    async def head(self, bucket: str, key: str) -> dict | None:
        """
        Returns dict with meta data excluding blob, or None if not found.
        """
        async with self._client() as s3:
            try:
                response = await s3.head_object(Bucket=bucket, Key=key)
                return {
                    'file_name': response.get('Metadata', {}).get('file_name'),
                    'content_type': response.get('ContentType'),
                    'content_length': response.get('ContentLength'),
                    'last_modified': response.get('LastModified'),
                    'metadata': response.get('Metadata', {})
                }
            except ClientError:
                return None


    async def get(self, bucket: str, key: str) -> dict | None:
        """
        Returns dict with meta data including blob, or None if not found.
        """
        async with self._client() as s3:
            try:
                response = await s3.get_object(Bucket=bucket, Key=key)
                async with response['Body'] as stream:
                    file_bytes = await stream.read()
                return {
                    'file_bytes': file_bytes,
                    'file_name': response.get('Metadata', {}).get('file_name'),
                    'content_type': response.get('ContentType'),
                    'content_length': response.get('ContentLength'),
                    'last_modified': response.get('LastModified'),
                    'metadata': response.get('Metadata', {})
                }
            except ClientError:
                return None


    async def get_url(
        self,
        bucket: str,
        key: str,
        expiration: int = 3600
    ) -> str | None:
        """
        Generate temporary signed URL for direct browser access.
        Prevents permanent public exposure.
        Expiration in seconds (default 1 hour).
        """
        async with self._client() as s3:
            try:
                return await s3.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': bucket, 'Key': key},
                    ExpiresIn=expiration
                )
            except ClientError as e:
                logger.error(f"Failed to create get_object for {bucket} > {key}: {e}")
                return None