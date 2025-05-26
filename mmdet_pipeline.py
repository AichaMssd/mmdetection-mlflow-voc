import kfp
from kfp import dsl

# Define the pipeline
@dsl.pipeline(
    name="MMDetection S3 Pipeline", 
    description="Pipeline to train MMDetection models using S3 data"
)
def mmdetection_pipeline(
    config_file: str = "pascal_voc/ssd300_voc0712.py"
):
    # Define a Kubernetes Pod to run the training job
    train = dsl.ContainerOp(
        name="mmdetection-train",
        image="aichamsd/mmdetection:v1",
        command=["/bin/bash", "-c"],
        arguments=[f"""
            # Add "-e" for error checking
            set -e

# Install AWS CLI
            apt-get update && apt-get install -y curl unzip
            curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
            unzip -q awscliv2.zip
            ./aws/install
            
            # Set AWS region
            export AWS_DEFAULT_REGION=eu-north-1
            
            # Create data directory
            mkdir -p /mmdetection/data
            
            # Download data from S3
            echo "Downloading data from S3"
            aws s3 cp --recursive s3://mmdet-training-data/data/VOCdevkit /mmdetection/data/VOCdevkit
            
            # Verify download
            ls -la /mmdetection/data/VOCdevkit/VOC2007/JPEGImages | head -5
            
            # Run training
            echo "Starting training with config: {config_file}"
            python /mmdetection/tools/train.py /mmdetection/configs/{config_file}


# Upload results back to S3
            echo "Uploading results to S3"
            aws s3 cp --recursive /mmdetection/work_dirs s3://mmdet-training-data/outputs/
            
            # Success message
            echo "Training completed successfully"
            mkdir -p /tmp/outputs
            echo "/mmdetection/work_dirs" > /tmp/outputs/model_path
        """],
        file_outputs={
            "model_path": "/tmp/outputs/model_path"
        }
    )
    
    # Try to set resources - different methods for different KFP versions
    try:
        # Method 1: Direct setting (newer KFP)
        train.container.resources = {
            'limits': {
                'nvidia.com/gpu': '1',
                'memory': '8Gi',
                'cpu': '4'
            },
            'requests': {
                'memory': '4Gi',
                'cpu': '2'
            }
        }

    except:
        try:
            # Method 2: Using add_resource methods (older KFP)
            train.add_resource_limit("nvidia.com/gpu", "1")
            train.add_resource_limit("memory", "8Gi")
            train.add_resource_limit("cpu", "4")
        except:
            print("Warning: Could not set resource limits")
    
    return train

# Compile the pipeline
if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=mmdetection_pipeline,
        package_path="mmdetection_s3_pipeline.yaml"
    )
    print("Pipeline compiled successfully!")



























































