"""OSS 上传签名接口，负责生成客户端直传文件所需的预签名地址。"""

import alibabacloud_oss_v2 as oss
from datetime import timedelta
import os

from dotenv import load_dotenv
from fastapi import APIRouter

load_dotenv()
router = APIRouter()

credentials_provider = oss.credentials.EnvironmentVariableCredentialsProvider()

cfg = oss.config.load_default()
cfg.credentials_provider = credentials_provider

cfg.region = 'cn-beijing'

client = oss.Client(cfg)

OSS_ENDPOINT = os.getenv("OSS_ENDPOINT", "oss-cn-beijing.aliyuncs.com")
OSS_BUCKET = os.getenv("OSS_BUCKET")


@router.get("/oss/presign")
def chat_endpoint(filename: str):
    """根据文件名生成 OSS 上传地址、访问地址和 Content-Type。"""
    content_type_map = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
        "webp": "image/webp",
    }
    ext = filename.split(".")[-1].lower() if "." in filename else "jpg"
    content_type = content_type_map.get(ext, "application/octet-stream")

    pre_result = client.presign(oss.PutObjectRequest(
        bucket=OSS_BUCKET,
        key=filename,
        content_type=content_type,
    ), expires=timedelta(seconds=3600))

    return {
        "uploadUrl": pre_result.url.strip('"'),
        "contentType": content_type,
        "accessUrl": f"https://{OSS_BUCKET}.{OSS_ENDPOINT}/{filename}"
    }
