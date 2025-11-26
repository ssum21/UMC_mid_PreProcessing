import os
import boto3
from botocore.exceptions import NoCredentialsError
from dotenv import load_dotenv

load_dotenv()

# Cloudflare R2 Configuration
R2_ENDPOINT_URL = os.environ.get("R2_ENDPOINT_URL")
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY")
R2_BUCKET_NAME = os.environ.get("R2_BUCKET_NAME")

def get_s3_client():
    return boto3.client(
        's3',
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY
    )

def upload_to_r2(file_path: str, object_name: str = None) -> str:
    """
    Uploads a file to Cloudflare R2 and returns the public URL (if configured) or the object key.
    """
    if object_name is None:
        object_name = os.path.basename(file_path)

    s3_client = get_s3_client()

    try:
        s3_client.upload_file(file_path, R2_BUCKET_NAME, object_name)
        print(f"Successfully uploaded {file_path} to R2 bucket {R2_BUCKET_NAME} as {object_name}")
        # Return the object name or a constructed URL if you have a custom domain
        # For now, we'll return the object name which is sufficient for internal reference
        return object_name 
    except FileNotFoundError:
        print(f"The file was not found: {file_path}")
        raise
    except NoCredentialsError:
        print("Credentials not available")
        raise
    except Exception as e:
        print(f"Error uploading to R2: {e}")
        raise

def download_from_r2(object_name: str, download_path: str):
    """
    Downloads a file from Cloudflare R2.
    """
    s3_client = get_s3_client()
    try:
        s3_client.download_file(R2_BUCKET_NAME, object_name, download_path)
        print(f"Successfully downloaded {object_name} to {download_path}")
    except Exception as e:
        print(f"Error downloading from R2: {e}")
        raise

def generate_presigned_url(object_name: str, expiration: int = 3600) -> str:
    """
    Generates a presigned URL for an R2 object.
    
    :param object_name: The name of the object (key) in the bucket
    :param expiration: Time in seconds for the URL to remain valid (default: 1 hour)
    :return: The presigned URL
    """
    try:
        response = s3_client.generate_presigned_url('get_object',
                                                    Params={'Bucket': R2_BUCKET_NAME,
                                                            'Key': object_name},
                                                    ExpiresIn=expiration)
        return response
    except Exception as e:
        print(f"Error generating presigned URL: {e}")
        return None

def generate_presigned_url(object_name: str, expiration=3600):
    """
    Generate a presigned URL to share an S3 object
    """
    s3_client = get_s3_client()
    try:
        response = s3_client.generate_presigned_url('get_object',
                                                    Params={'Bucket': R2_BUCKET_NAME,
                                                            'Key': object_name},
                                                    ExpiresIn=expiration)
    except Exception as e:
        print(e)
        return None
    return response
