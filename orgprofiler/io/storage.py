import httpx
import base64
from typing import Optional
from loguru import logger
class StorageUploader:
    def __init__(self, supabase_url: str, supabase_key: str):
        self.base = supabase_url.rstrip("/")
        self.key  = supabase_key
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        # Log a hint about the key format (safely)
        key_preview = f"{self.key[:5]}...{self.key[-5:]}" if self.key else "None"
        logger.debug(f"Initializing StorageUploader with key: {key_preview}")
        
        self._client = httpx.AsyncClient(
            http2=False,
            timeout=httpx.Timeout(connect=20, read=60, write=60, pool=20),
            limits=httpx.Limits(max_keepalive_connections=0, max_connections=10),
            headers={
                "Connection": "close", 
                "Authorization": f"Bearer {self.key}",
                "apikey": self.key,  # Most Supabase services require this header
            }
        )
        return self

    async def __aexit__(self, *exc):
        if self._client:
            await self._client.aclose()

    async def upload_png_dataurl(self, bucket: str, path: str, data_url: str) -> str:
        assert self._client is not None
        comma = data_url.find(",")
        if comma == -1:
            raise ValueError("Invalid data URL")
        body = base64.b64decode(data_url[comma+1:], validate=True)

        url = f"{self.base}/storage/v1/object/{bucket}/{path}"
        resp = await self._client.post(
            url,
            headers={"Content-Type": "image/png", "x-upsert": "true"},
            content=body,
        )
        if resp.status_code != 200:
            try:
                error_detail = resp.json()
            except:
                error_detail = resp.text
            raise Exception(f"Supabase Storage Error ({resp.status_code}): {error_detail}")

        return f"{self.base}/storage/v1/object/public/{bucket}/{path}"

