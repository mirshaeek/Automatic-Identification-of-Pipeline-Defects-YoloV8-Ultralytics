from ultralytics import YOLO
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# Load a model
if __name__ == '__main__':
    model = YOLO('D:/yolov8/runs/detect/train8/weights/last.pt')  # load a partially trained model
    dataset = 'D:/yolov8/config.yaml'
    
    # Define training parameters
    epochs = 500
    batch_size = -1  # Explicit batch size
    initial_lr = 1e-4  # Lower learning rate for fine-tuning

    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr)

    # Define scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Training loop
    for epoch in range(epochs):
        # Train for one epoch
        results = model.train(
            data=dataset,
            epochs=1,  # Train one epoch at a time
            pretrained=True,
            save_period=1,
            batch=batch_size,
            augment=True,
            save=True,
            plots=True,
            val=True,
            imgsz=640,
            workers=8,
            project=None,
            name='train150',
            exist_ok=False,
            verbose=True,
            seed=0,
            deterministic=True,
            single_cls=False,
            rect=False,
            cos_lr=True,
            close_mosaic=10,
            amp=True,
            fraction=1.0,
            profile=False,
            multi_scale=False,
            overlap_mask=True,
            mask_ratio=4,
            dropout=0.0,
            split='val',
            save_json=False,
            save_hybrid=False,
            conf=0.25,
            iou=0.7,
            max_det=300,
            half=False,
            dnn=False,
            visualize=True,
            agnostic_nms=False,
            classes=None,
            retina_masks=False,
            embed=None,
            show=False,
            save_frames=False,
            save_txt=False,
            save_conf=False,
            save_crop=False,
            show_labels=True,
            show_conf=True,
            show_boxes=True,
            format='torchscript',
            keras=False,
            optimize=False,
            int8=False,
            dynamic=False,
            simplify=False,
            opset=None,
            workspace=4,
            nms=False,
            lr0=initial_lr,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            pose=12.0,
            kobj=1.0,
            label_smoothing=0.0,
            nbs=64,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            bgr=0.0,
            mosaic=1.0,
            mixup=0.0,
            copy_paste=0.0,
            auto_augment='randaugment',
            erasing=0.4,
            crop_fraction=1.0,
            cfg=None,
            tracker='botsort.yaml',
            save_dir='runs/detect/train150'
        )

        # Step the scheduler
        scheduler.step()
