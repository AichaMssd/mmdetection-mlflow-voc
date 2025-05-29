from kfp import dsl, compiler
from kfp.dsl import Input, Output, Dataset, Model, Metrics, Artifact

# Component 1: Download and prepare VOC dataset
@dsl.component(
    base_image='valohai/mmdetect-kubeflow'
)
def prepare_voc_dataset(
    output_dataset: Output[Dataset],
    dataset_name: str = 'voc2007'  # or 'voc2012' or 'voc0712' for both
):
    import os
    import subprocess
    from pathlib import Path
    import shutil
    
    # Create output directory
    output_path = Path(output_dataset.path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create a temporary data directory
    data_dir = Path('./data')
    data_dir.mkdir(exist_ok=True)
    
    print(f"Downloading VOC dataset: {dataset_name}")
    
    # Download using the MMDetection download script
    download_script = f'''
import os
import urllib.request
import tarfile
import zipfile
from pathlib import Path

def download_voc(dataset_name, save_dir):
    """Download VOC dataset"""
    urls = {{
        'voc2007': [
            ('http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar', 'VOCtrainval_06-Nov-2007.tar'),
            ('http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar', 'VOCtest_06-Nov-2007.tar'),
            ('http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar', 'VOCdevkit_08-Jun-2007.tar')
        ],
        'voc2012': [
            ('http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar', 'VOCtrainval_11-May-2012.tar')
        ]
    }}
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Download files
    if dataset_name == 'voc0712':
        download_files = urls['voc2007'] + urls['voc2012']
    else:
        download_files = urls.get(dataset_name, [])
    
    for url, filename in download_files:
        filepath = save_path / filename
        if not filepath.exists():
            print(f"Downloading {{filename}}...")
            urllib.request.urlretrieve(url, filepath)
        
        print(f"Extracting {{filename}}...")
        if filename.endswith('.tar'):
            with tarfile.open(filepath, 'r') as tar:
                tar.extractall(save_path)
        elif filename.endswith('.zip'):
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(save_path)
        os.remove(filepath)

    print(f"VOC dataset downloaded to {{save_path}}")

# Run download
download_voc('{dataset_name}', './data')
'''

    # Save and run the download script
    with open('download_voc.py', 'w') as f:
        f.write(download_script)
    
    subprocess.run(['python', 'download_voc.py'], check=True)
    
    # Copy the downloaded data to output
    voc_path = data_dir / 'VOCdevkit'
    if voc_path.exists():
        shutil.copytree(voc_path, output_path / 'VOCdevkit')
        print(f"Dataset copied to {output_path}")
    else:
        raise RuntimeError("VOCdevkit not found after download!")
    
    # Verify dataset structure
    print("Dataset structure:")
    for year in ['VOC2007', 'VOC2012']:
        year_path = output_path / 'VOCdevkit' / year
        if year_path.exists():
            print(f"  {year}: ✓")
            print(f"    - Annotations: {(year_path / 'Annotations').exists()}")
            print(f"    - JPEGImages: {(year_path / 'JPEGImages').exists()}")
            print(f"    - ImageSets: {(year_path / 'ImageSets').exists()}")

# Component 2: Train SSD model with MLflow tracking
@dsl.component(
    base_image='valohai/mmdetect-kubeflow'
)
def train_ssd_with_mlflow(
    dataset: Input[Dataset],
    model_output: Output[Model],
    metrics_output: Output[Metrics],
    config_output: Output[Artifact],
    github_repo: str = '',  # Your modified MMDetection repo
    config_path: str = 'configs/pascal_voc/ssd300_voc0712.py',
    mlflow_tracking_uri: str = '',  # MLflow server URI if you have one
    mlflow_experiment_name: str = 'ssd_voc_training',
    num_epochs: int = 24,
    batch_size: int = 8,
    learning_rate: float = 1e-3
):
    import subprocess
    import os
    from pathlib import Path
    import shutil
    import json
    import mlflow
    import mlflow.pytorch
    
    # Setup MLflow
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    mlflow.set_experiment(mlflow_experiment_name)
    
    # Clone your modified repo if provided
    if github_repo:
        print(f"Cloning custom repo from {github_repo}")
        subprocess.run(['git', 'clone', github_repo, 'mmdetection'], check=True)
        os.chdir('mmdetection')

        # Fix MKL threading conflict
        import numpy
        
        # Install in editable mode to use your modifications
        subprocess.run(['pip', 'install', '-e', '.'], check=True)
    
    # Create custom config with correct data paths
    data_root = Path(dataset.path) / 'VOCdevkit'
    
    custom_config = f'''
# Based on {config_path}
_base_ = [
    '../_base_/models/ssd300.py',
    '../_base_/datasets/voc0712.py',
    '../_base_/default_runtime.py'
]

# Data settings
data_root = '{data_root}/'
data = dict(
    samples_per_gpu={batch_size},
    workers_per_gpu=4,
    train=dict(
        _delete_=True,
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type='VOCDataset',
            data_root=data_root,
            ann_file=data_root + 'VOC2007/ImageSets/Main/trainval.txt',
            img_prefix=data_root + 'VOC2007/',
            pipeline=[
                dict(type='LoadImageFromFile', backend_args=None),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='Resize', img_scale=(300, 300), keep_ratio=False),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(type='PhotoMetricDistortion'),
                dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[1, 1, 1]),
                dict(type='Pad', size_divisor=1),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
            ]
        )
    ),
    val=dict(
        type='VOCDataset',
        data_root=data_root,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='Resize', img_scale=(300, 300), keep_ratio=False),
            dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[1, 1, 1]),
            dict(type='Pad', size_divisor=1),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]
    ),
    test=dict(
        type='VOCDataset',
        data_root=data_root,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='Resize', img_scale=(300, 300), keep_ratio=False),
            dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[1, 1, 1]),
            dict(type='Pad', size_divisor=1),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]
    )
)

# Model settings
model = dict(
    bbox_head=dict(
        num_classes=20  # VOC has 20 classes
    )
)

# Optimizer
optimizer = dict(type='SGD', lr={learning_rate}, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict()

# Learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])

# Runtime settings
runner = dict(type='EpochBasedRunner', max_epochs={num_epochs})
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

evaluation = dict(interval=1, metric='mAP')

# Work directory
work_dir = './work_dirs/ssd300_voc_kubeflow'

# For better performance on VOC
cudnn_benchmark = True
'''
    
    # Save custom config
    with open('custom_ssd_config.py', 'w') as f:
        f.write(custom_config)
    
    # Save config for later use
    shutil.copy('custom_ssd_config.py', config_output.path)
    
    # Start MLflow run
    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_param("model_type", "ssd300")
        mlflow.log_param("dataset", "voc2007")  # Changed to voc2007
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("optimizer", "SGD")
        
        # Run training
        print("Starting training with MLflow tracking...")
        
        # If you have your modified train.py with MLflow
        if github_repo or os.path.exists('tools/train.py'):
            # Use your train.py which should have MLflow integration
            cmd = [
                'python', 'tools/train.py',
                'custom_ssd_config.py',
                '--work-dir', './work_dirs/ssd300_voc_kubeflow'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            print(result.stdout)
            if result.returncode != 0:
                print(f"Error: {result.stderr}")
                raise RuntimeError(f"Training failed: {result.stderr}")
        
        # Find and save the best model
        work_dir = Path('./work_dirs/ssd300_voc_kubeflow')
        checkpoints = list(work_dir.glob('epoch_*.pth'))
        
        if checkpoints:
            # Get the latest checkpoint
            latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.stem.split('_')[1]))[-1]
            
            # Copy model to output
            shutil.copy(latest_checkpoint, model_output.path)
            print(f"Model saved to {model_output.path}")
            
            # Log model to MLflow
            mlflow.pytorch.log_model(
                pytorch_model=model_output.path,
                artifact_path="model",
                registered_model_name="ssd300_voc2007"  # Changed name
            )
        else:
            raise RuntimeError("No checkpoint found after training!")
        
        # Log final metrics
        metrics_output.log_metric('epochs_trained', num_epochs)
        print(f"Training completed! MLflow run ID: {run.info.run_id}")

# Define the pipeline
@dsl.pipeline(
    name='MMDetection SSD VOC2007 Training with MLflow',
    description='Train SSD300 on Pascal VOC 2007 with MLflow tracking'
)
def mmdet_voc_mlflow_pipeline(
    github_repo: str = '',  # Your GitHub repo with modified MMDetection
    mlflow_tracking_uri: str = '',  # MLflow server URI (optional)
    mlflow_experiment_name: str = 'ssd_voc_training',
    num_epochs: int = 24,
    batch_size: int = 8,
    learning_rate: float = 1e-3
):
    # Step 1: Download and prepare VOC dataset
    dataset_task = prepare_voc_dataset(
        dataset_name='voc2007'  # Download only VOC2007
    )
    
    # Step 2: Train SSD model with MLflow tracking
    train_task = train_ssd_with_mlflow(
        dataset=dataset_task.outputs['output_dataset'],
        github_repo=github_repo,
        config_path='configs/pascal_voc/ssd300_voc0712.py',
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )

# Compile the pipeline
if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=mmdet_voc_mlflow_pipeline,
        package_path='mmdet_voc_mlflow_pipeline.yaml'
    )
    print("Pipeline compiled to: mmdet_voc_mlflow_pipeline.yaml")
