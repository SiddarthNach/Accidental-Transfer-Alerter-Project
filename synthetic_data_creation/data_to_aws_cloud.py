import boto3

s3 = boto3.client('s3')

# File you want to upload
file_name = 'synthetic_transactions_1M.csv'
bucket_name = 'accidentaltransferproject'
object_name = 'data/synthetic_transactions_1M.csv'  

s3.upload_file(file_name, bucket_name, object_name)

print(f"Uploaded {file_name} to s3://{bucket_name}/{object_name}")
