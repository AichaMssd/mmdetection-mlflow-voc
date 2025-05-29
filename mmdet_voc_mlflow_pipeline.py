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
            
            # List some files for debugging
            if (year_path / 'ImageSets' / 'Main').exists():
                print(f"    - ImageSets/Main files: {list((year_path / 'ImageSets' / 'Main').glob('*.txt'))[:5]}")

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
    else:
        # Use local file store
        mlflow_dir = Path('./mlruns')
        mlflow_dir.mkdir(exist_ok=True)
        mlflow.set_tracking_uri(f'file://{mlflow_dir.absolute()}')
    
    # Create or get experiment
    try:
        experiment = mlflow.get_experiment_by_name(mlflow_experiment_name)
        if experiment is None:
            print(f"Creating new MLflow experiment: {mlflow_experiment_name}")
            experiment_id = mlflow.create_experiment(mlflow_experiment_name)
        else:
            print(f"Using existing MLflow experiment: {mlflow_experiment_name} (ID: {experiment.experiment_id})")
            experiment_id = experiment.experiment_id
    except Exception as e:
        print(f"Error with MLflow experiment setup: {e}")
        print("Using default experiment")
        experiment_id = "0"
    
    # Set the experiment
    mlflow.set_experiment(mlflow_experiment_name)
    
    # Clone your modified repo if provided
    if github_repo:
        print(f"Cloning custom repo from {github_repo}")
        subprocess.run(['git', 'clone', github_repo, 'mmdetection'], check=True)
        os.chdir('mmdetection')

        # Fix MKL threading conflict
        import numpy
        print(f"Numpy version: {numpy.__version__}")
        
        # Set MKL threading layer to avoid conflicts
        os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
        
        # Install in editable mode to use your modifications
        subprocess.run(['pip', 'install', '-e', '.'], check=True)
        
        # Install missing libraries for OpenCV
        subprocess.run(['apt-get', 'update'], check=True)
        subprocess.run(['apt-get', 'install', '-y', 'libgl1-mesa-glx', 'libglib2.0-0'], check=True)
    
    # CRITICAL FIX: Correctly set the data root
    # The dataset.path already contains the full path including VOCdevkit
    data_root = Path(dataset.path)
    
    # Debug: Check the actual structure
    print(f"Dataset path: {data_root}")
    print(f"Contents of dataset path: {list(data_root.iterdir())}")
    
    # Check if VOCdevkit is directly in data_root or if data_root IS VOCdevkit
    if (data_root / 'VOCdevkit').exists():
        # VOCdevkit is a subdirectory
        data_root = data_root / 'VOCdevkit'
        print(f"Found VOCdevkit at: {data_root}")
    elif data_root.name == 'VOCdevkit':
        # data_root is already VOCdevkit
        print(f"data_root is already VOCdevkit: {data_root}")
    else:
        # Check if VOC2007 is directly in data_root (in case of different structure)
        if (data_root / 'VOC2007').exists():
            print(f"Found VOC2007 directly in data_root: {data_root}")
        else:
            raise RuntimeError(f"Could not find VOCdevkit or VOC2007 in {data_root}")
    
    # Verify the expected files exist
    voc2007_path = data_root / 'VOC2007'
    if voc2007_path.exists():
        print(f"VOC2007 path: {voc2007_path}")
        print(f"VOC2007 contents: {list(voc2007_path.iterdir())[:10]}")
        
        trainval_file = voc2007_path / 'ImageSets' / 'Main' / 'trainval.txt'
        test_file = voc2007_path / 'ImageSets' / 'Main' / 'test.txt'
        
        if trainval_file.exists():
            print(f"✓ Found trainval.txt at: {trainval_file}")
        else:
            print(f"✗ trainval.txt not found at: {trainval_file}")
            print(f"  ImageSets/Main contents: {list((voc2007_path / 'ImageSets' / 'Main').glob('*.txt'))}")
        
        if test_file.exists():
            print(f"✓ Found test.txt at: {test_file}")
        else:
            print(f"✗ test.txt not found at: {test_file}")
    
    # Check if we can use the existing config
    existing_config_path = Path(config_path)
    if existing_config_path.exists():
        print(f"Using existing config: {existing_config_path}")
        # Read the existing config and modify it
        with open(existing_config_path, 'r') as f:
            config_content = f.read()
        
        # Create a modified version with our data path
        custom_config = f'''
# Based on {config_path}
_base_ = '{existing_config_path}'

# Override data settings with our paths
data_root = '{data_root}/'
data = dict(
    samples_per_gpu={batch_size},
    workers_per_gpu=4,
    train=dict(
        type='VOCDataset',
        data_root='{data_root}/',
        ann_file='VOC2007/ImageSets/Main/trainval.txt',
        data_prefix=dict(sub_data_root='VOC2007/'),
    ),
    val=dict(
        type='VOCDataset',
        data_root='{data_root}/',
        ann_file='VOC2007/ImageSets/Main/test.txt',
        data_prefix=dict(sub_data_root='VOC2007/'),
    ),
    test=dict(
        type='VOCDataset',
        data_root='{data_root}/',
        ann_file='VOC2007/ImageSets/Main/test.txt',
        data_prefix=dict(sub_data_root='VOC2007/'),
    )
)

# Override training settings
optimizer = dict(lr={learning_rate})
runner = dict(max_epochs={num_epochs})
'''
    else:
        print(f"Config not found at {existing_config_path}, creating self-contained config")
        # Create a self-contained config if the base doesn't exist
        custom_config = f'''
# Self-contained SSD300 VOC configuration
_base_ = [
    '../_base_/models/ssd300.py',
    '../_base_/schedules/schedule_2x.py', 
    '../_base_/default_runtime.py'
]

# Model settings
model = dict(
    bbox_head=dict(
        num_classes=20  # VOC has 20 classes
    )
)

# Dataset settings
dataset_type = 'VOCDataset'
data_root = '{data_root}/'
backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(300, 300), keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='Expand',
        mean=[123.675, 116.28, 103.53],
        to_rgb=True,
        ratio_range=(1, 4)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[1, 1, 1]),
    dict(type='Pad', size_divisor=1),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(300, 300), keep_ratio=False),
    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[1, 1, 1]),
    dict(type='Pad', size_divisor=1),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

train_dataloader = dict(
    batch_size={batch_size},
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='VOC2007/ImageSets/Main/trainval.txt',
            data_prefix=dict(sub_data_root='VOC2007/'),
            filter_cfg=dict(
                filter_empty_gt=True, min_size=32, bbox_min_size=32),
            pipeline=train_pipeline,
            backend_args=backend_args)))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='VOC2007/ImageSets/Main/test.txt',
        data_prefix=dict(sub_data_root='VOC2007/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

test_dataloader = val_dataloader

# Evaluation
val_evaluator = dict(
    type='VOCMetric',
    metric='mAP',
    eval_mode='11points')
test_evaluator = val_evaluator

# Optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr={learning_rate}, momentum=0.9, weight_decay=5e-4))

# Learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end={num_epochs},
        by_epoch=True,
        milestones=[16, 22],
        gamma=0.1)
]

# Training schedule
train_cfg = dict(
    type='EpochBasedTrainLoop', 
    max_epochs={num_epochs}, 
    val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Runtime settings
default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False

# Work directory
work_dir = './work_dirs/ssd300_voc_kubeflow'
'''
    
    # Save custom config
    with open('custom_ssd_config.py', 'w') as f:
        f.write(custom_config)
    
    # Also save it in the configs directory to ensure proper resolution
    os.makedirs('configs/custom', exist_ok=True)
    shutil.copy('custom_ssd_config.py', 'configs/custom/custom_ssd_config.py')
    
    # Save config for later use
    shutil.copy('custom_ssd_config.py', config_output.path)
    
    # Start MLflow run
    try:
        with mlflow.start_run() as run:
            # Log parameters
            mlflow.log_param("model_type", "ssd300")
            mlflow.log_param("dataset", "voc2007")
            mlflow.log_param("num_epochs", num_epochs)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("optimizer", "SGD")
            mlflow.log_param("config_path", config_path)
            
            # Log the config file
            mlflow.log_artifact('custom_ssd_config.py')
            
            # Run training
            print("Starting training with MLflow tracking...")
            print(f"MLflow Run ID: {run.info.run_id}")
            print(f"MLflow Experiment ID: {run.info.experiment_id}")
            
            # If you have your modified train.py with MLflow
            if github_repo and os.path.exists('tools/train.py'):
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
            else:
                # Fallback: use standard MMDetection training without your modifications
                print("Warning: No custom train.py found. Using standard MMDetection training.")
                print("MLflow metrics logging may be limited.")
                
                # Import and use MMDetection's train API directly
                try:
                    from mmdet.apis import train_detector
                    from mmdet.models import build_detector
                    from mmcv import Config
                    from mmdet.datasets import build_dataset
                    
                    cfg = Config.fromfile('custom_ssd_config.py')
                    
                    # Build dataset
                    datasets = [build_dataset(cfg.data.train)]
                    
                    # Build model
                    model = build_detector(
                        cfg.model,
                        train_cfg=cfg.get('train_cfg'),
                        test_cfg=cfg.get('test_cfg'))
                    model.init_weights()
                    
                    # Train
                    train_detector(
                        model,
                        datasets,
                        cfg,
                        distributed=False,
                        validate=True)
                        
                except ImportError:
                    print("MMDetection API not available. Running train.py script.")
                    cmd = [
                        'python', '-m', 'mmdet.tools.train',
                        'custom_ssd_config.py',
                        '--work-dir', './work_dirs/ssd300_voc_kubeflow'
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    print(result.stdout)
                    if result.returncode != 0:
                        print(f"Error: {result.stderr}")
            
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
                try:
                    mlflow.pytorch.log_model(
                        pytorch_model=model_output.path,
                        artifact_path="model",
                        registered_model_name="ssd300_voc2007"
                    )
                except Exception as e:
                    print(f"Warning: Could not log model to MLflow registry: {e}")
                    # Just log as artifact instead
                    mlflow.log_artifact(str(latest_checkpoint), "model")
                
                # Log final metrics
                mlflow.log_metric("final_epoch", int(latest_checkpoint.stem.split('_')[1]))
                
            else:
                raise RuntimeError("No checkpoint found after training!")
            
            # Log metrics to KFP
            metrics_output.log_metric('epochs_trained', num_epochs)
            metrics_output.log_metric('mlflow_run_id', run.info.run_id)
            
            print(f"Training completed! MLflow run ID: {run.info.run_id}")
            
    except Exception as e:
        print(f"Error during MLflow tracking: {e}")
        print("Continuing without MLflow tracking...")
        
        # Run training without MLflow if it fails
        if github_repo and os.path.exists('tools/train.py'):
            cmd = [
                'python', 'tools/train.py',
                'custom_ssd_config.py',
                '--work-dir', './work_dirs/ssd300_voc_kubeflow'
            ]
        else:
            cmd = [
                'python', '-m', 'mmdet.tools.train',
                'custom_ssd_config.py',
                '--work-dir', './work_dirs/ssd300_voc_kubeflow'
            ]
            
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            raise RuntimeError(f"Training failed: {result.stderr}")
        
        # Save model even if MLflow fails
        work_dir = Path('./work_dirs/ssd300_voc_kubeflow')
        checkpoints = list(work_dir.glob('epoch_*.pth'))
        if checkpoints:
            latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.stem.split('_')[1]))[-1]
            shutil.copy(latest_checkpoint, model_output.path)
            print(f"Model saved to {model_output.path}")
            metrics_output.log_metric('epochs_trained', num_epochs)
        else:
            raise RuntimeError("No checkpoint found after training!")

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