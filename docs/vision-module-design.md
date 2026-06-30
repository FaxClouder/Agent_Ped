# 视觉模块详细设计

## 一、模块概述

视觉模块负责从视频中提取行人轨迹数据，作为可插拔组件接入分析管道。

核心流程：
```
视频输入 → 行人检测 (YOLO26) → 多目标跟踪 (ByteTrack) 
→ 坐标变换 (像素→世界) → 轨迹后处理 → TrajectoryData 输出
```

---

## 二、检测模型：YOLO26

### 2.1 选型说明

YOLO26 是 Ultralytics YOLO 系列模型，通过 `ultralytics` 包统一管理。官方 YOLO26 默认走 end-to-end / NMS-free 推理路径；如需兼容传统 one-to-many/NMS 行为，可在推理时显式关闭 `end2end`。

| 模型 | 定位 | 适用场景 |
|------|------|---------|
| yolo26n | nano | 实时/边缘设备 |
| yolo26s | small | 轻量级部署 |
| yolo26m | medium | 平衡精度/速度 |
| yolo26l | large | 高精度离线分析 |
| yolo26x | extra-large | 最高精度离线分析（推荐） |

**本项目推荐 yolo26x**：行人流场景中遮挡、密集情况频繁，需要最高检测精度。离线分析场景对实时性要求不高。

### 2.2 检测器封装

```python
from ultralytics import YOLO
from pathlib import Path
import numpy as np

class PedestrianDetector:
    """YOLO26 行人检测器"""
    
    def __init__(self, config: dict):
        self.model_path = config.get('model', 'yolo26x.pt')
        self.confidence = config.get('confidence', 0.5)
        self.iou_threshold = config.get('iou_threshold', 0.45)
        self.classes = config.get('classes', [0])  # person
        self.input_size = config.get('input_size', 1280)
        self.device = config.get('device', 'cuda:0')
        self.half = config.get('half_precision', True)
        self.end2end = config.get('end2end', True)
        
        self.model = YOLO(self.model_path)
        
    def detect(self, frame: np.ndarray) -> list:
        """
        检测单帧中的行人
        
        Returns:
            List[Detection]: 检测结果列表
        """
        results = self.model(
            frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            classes=self.classes,
            imgsz=self.input_size,
            device=self.device,
            half=self.half,
            end2end=self.end2end,
            verbose=False,
        )
        
        detections = []
        if results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                det = Detection(
                    bbox=boxes.xyxy[i].cpu().numpy(),  # [x1, y1, x2, y2]
                    confidence=float(boxes.conf[i]),
                    class_id=int(boxes.cls[i]),
                )
                detections.append(det)
        
        return detections
    
    def detect_batch(self, frames: list) -> list:
        """批量检测"""
        results = self.model(
            frames,
            conf=self.confidence,
            iou=self.iou_threshold,
            classes=self.classes,
            imgsz=self.input_size,
            device=self.device,
            half=self.half,
            verbose=False,
            stream=True,
        )
        return [self._parse_results(r) for r in results]
```

---

## 三、多目标跟踪

### 3.1 跟踪器选择

| 跟踪器 | 特点 | 适用场景 |
|--------|------|---------|
| **ByteTrack** | 利用低分框，高召回 | 密集行人流 (推荐) |
| DeepSORT | ReID 外观特征 | 遮挡严重/重识别场景 |
| BoT-SORT | ByteTrack + 外观 + 相机补偿 | 运动相机 |

**推荐 ByteTrack**：行人流场景通常是固定相机，密集度高，ByteTrack 对低置信度目标的利用能有效提高跟踪连续性。

### 3.2 跟踪器封装

```python
from boxmot import ByteTrack, DeepSORT

class PedestrianTracker:
    """多目标跟踪器"""
    
    def __init__(self, config: dict):
        self.config = config
        self.algorithm = config.get('algorithm', 'bytetrack')
        self.tracker = self._build_tracker()

    def _build_tracker(self):
        if self.algorithm == 'bytetrack':
            self.tracker = ByteTrack(
                track_thresh=self.config.get('match_thresh', 0.8),
                track_buffer=self.config.get('track_buffer', 30),
                match_thresh=0.8,
            )
        elif self.algorithm == 'deepsort':
            self.tracker = DeepSORT(
                reid_weights=Path(self.config.get('reid_model', 'osnet_x0_25_msmt17.pt')),
                max_cosine_distance=self.config.get('max_cosine_distance', 0.3),
                max_age=self.config.get('track_buffer', 30),
            )
        else:
            raise ValueError(f"Unknown tracker: {self.algorithm}")

        return self.tracker
    
    def update(self, detections: list, frame: np.ndarray) -> np.ndarray:
        """
        更新跟踪状态
        
        Args:
            detections: 当前帧检测结果
            frame: 当前帧图像 (DeepSORT 需要用于 ReID)
            
        Returns:
            tracks: (N, 6) array [x1, y1, x2, y2, track_id, confidence]
        """
        if not detections:
            dets = np.empty((0, 5))
        else:
            dets = np.array([[*d.bbox, d.confidence] for d in detections])
        
        tracks = self.tracker.update(dets, frame)
        return tracks
    
    def reset(self):
        """重置跟踪器状态 (新视频时调用)"""
        self.tracker = self._build_tracker()
```

---

## 四、坐标变换

### 4.1 变换方法

| 方法 | 精度 | 要求 | 适用场景 |
|------|------|------|---------|
| **单应矩阵 (Homography)** | 高 | 4+ 对应点标定 | 平面场景 |
| 简易缩放 (Scale) | 中 | 已知像素/米比 | 正俯视相机 |
| 无变换 (None) | - | 无 | 仅像素分析 |

### 4.2 实现

```python
import cv2

class CoordinateTransformer:
    """像素坐标 → 世界坐标变换"""
    
    def __init__(self, config: dict):
        self.method = config.get('method', 'none')
        self.homography_matrix = None
        self.pixel_per_meter = config.get('pixel_per_meter', None)
        
        if self.method == 'homography':
            calib_file = config.get('calibration_file')
            if calib_file:
                self.homography_matrix = self._load_homography(calib_file)
    
    def transform(self, pixel_points: np.ndarray) -> np.ndarray:
        """
        将像素坐标转换为世界坐标 (米)
        
        Args:
            pixel_points: (N, 2) 像素坐标 [px_x, px_y]
            
        Returns:
            world_points: (N, 2) 世界坐标 [m_x, m_y]
        """
        if self.method == 'none':
            return pixel_points
        
        elif self.method == 'scale':
            return pixel_points / self.pixel_per_meter
        
        elif self.method == 'homography':
            if self.homography_matrix is None:
                raise ValueError("Homography matrix not loaded")
            
            # OpenCV perspectiveTransform 要求 (N, 1, 2) 形状
            pts = pixel_points.reshape(-1, 1, 2).astype(np.float64)
            world_pts = cv2.perspectiveTransform(pts, self.homography_matrix)
            return world_pts.reshape(-1, 2)
    
    def transform_bbox_center(self, bboxes: np.ndarray) -> np.ndarray:
        """将 bbox 底边中点转为世界坐标 (行人脚部位置)"""
        # 底边中点: ((x1+x2)/2, y2)
        centers = np.column_stack([
            (bboxes[:, 0] + bboxes[:, 2]) / 2,  # x center
            bboxes[:, 3]  # y bottom (脚部)
        ])
        return self.transform(centers)
    
    @staticmethod
    def calibrate_from_points(pixel_points: np.ndarray, 
                              world_points: np.ndarray) -> np.ndarray:
        """
        从对应点计算单应矩阵
        
        Args:
            pixel_points: (N, 2) 像素坐标 (至少4个点)
            world_points: (N, 2) 世界坐标
            
        Returns:
            H: (3, 3) 单应矩阵
        """
        H, mask = cv2.findHomography(
            pixel_points.astype(np.float64),
            world_points.astype(np.float64),
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0
        )
        return H
    
    @staticmethod
    def _load_homography(filepath: str) -> np.ndarray:
        """从文件加载单应矩阵"""
        import json
        with open(filepath) as f:
            data = json.load(f)
        return np.array(data['homography_matrix'])
```

### 4.3 标定文件格式

```json
{
  "calibration_method": "homography",
  "pixel_points": [[100, 200], [500, 200], [500, 600], [100, 600]],
  "world_points": [[0, 0], [10, 0], [10, 15], [0, 15]],
  "homography_matrix": [
    [0.02, 0.001, -1.5],
    [0.0, 0.025, -3.2],
    [0.0, 0.0, 1.0]
  ],
  "reprojection_error_px": 2.3,
  "unit": "meters"
}
```

---

## 五、轨迹后处理

```python
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

class TrajectoryPostProcessor:
    """轨迹后处理：平滑、插值、过滤"""
    
    def __init__(self, config: dict):
        self.min_track_length = config.get('min_track_length', 10)
        self.smoothing = config.get('smoothing', 'savgol')
        self.smoothing_window = config.get('smoothing_window', 5)
        self.interpolation = config.get('interpolation', True)
        self.fps = config.get('fps', 25.0)
    
    def process(self, raw_tracks: dict) -> List[PedestrianTrack]:
        """
        完整后处理管道
        
        Args:
            raw_tracks: {track_id: [(frame, x, y, conf), ...]}
            
        Returns:
            List[PedestrianTrack]
        """
        processed = []
        
        for track_id, points in raw_tracks.items():
            # 1. 长度过滤
            if len(points) < self.min_track_length:
                continue
            
            frames = np.array([p[0] for p in points])
            positions = np.array([[p[1], p[2]] for p in points])
            confidences = np.array([p[3] for p in points])
            
            # 2. 插值缺失帧
            if self.interpolation:
                frames, positions, confidences = self._interpolate(
                    frames, positions, confidences
                )
            
            # 3. 平滑
            if self.smoothing != 'none':
                positions = self._smooth(positions)
            
            # 4. 异常值过滤 (速度突变)
            mask = self._filter_outliers(positions, frames)
            frames = frames[mask]
            positions = positions[mask]
            confidences = confidences[mask]
            
            if len(frames) < self.min_track_length:
                continue
            
            # 构建输出
            track = PedestrianTrack(
                track_id=int(track_id),
                frames=frames.tolist(),
                positions=[Position(x=p[0], y=p[1]) for p in positions],
                timestamps=(frames / self.fps).tolist(),
                confidence=confidences.tolist(),
            )
            processed.append(track)
        
        return processed
    
    def _interpolate(self, frames, positions, confidences):
        """线性插值缺失帧"""
        full_frames = np.arange(frames[0], frames[-1] + 1)
        
        fx = interp1d(frames, positions[:, 0], kind='linear', fill_value='extrapolate')
        fy = interp1d(frames, positions[:, 1], kind='linear', fill_value='extrapolate')
        fc = interp1d(frames, confidences, kind='nearest', fill_value='extrapolate')
        
        new_positions = np.column_stack([fx(full_frames), fy(full_frames)])
        new_confidences = fc(full_frames)
        
        return full_frames, new_positions, new_confidences
    
    def _smooth(self, positions: np.ndarray) -> np.ndarray:
        """轨迹平滑"""
        if len(positions) < self.smoothing_window:
            return positions
        
        if self.smoothing == 'savgol':
            window = min(self.smoothing_window, len(positions))
            if window % 2 == 0:
                window -= 1
            if window < 3:
                return positions
            smoothed_x = savgol_filter(positions[:, 0], window, polyorder=2)
            smoothed_y = savgol_filter(positions[:, 1], window, polyorder=2)
            return np.column_stack([smoothed_x, smoothed_y])
        
        elif self.smoothing == 'moving_avg':
            kernel = np.ones(self.smoothing_window) / self.smoothing_window
            smoothed_x = np.convolve(positions[:, 0], kernel, mode='same')
            smoothed_y = np.convolve(positions[:, 1], kernel, mode='same')
            return np.column_stack([smoothed_x, smoothed_y])
        
        return positions
    
    def _filter_outliers(self, positions: np.ndarray, frames: np.ndarray,
                         max_speed: float = 5.0) -> np.ndarray:
        """
        过滤速度异常点 (行人最大速度约 2-3 m/s，留余量 5 m/s)
        """
        if len(positions) < 2:
            return np.ones(len(positions), dtype=bool)
        
        displacements = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        dt = np.diff(frames) / self.fps
        speeds = displacements / np.maximum(dt, 1e-6)
        
        # 首点始终保留
        valid = np.concatenate([[True], speeds < max_speed])
        return valid
```

---

## 六、视觉管道主类

```python
import cv2
from pathlib import Path

class VisionPipeline:
    """视觉处理完整管道"""
    
    def __init__(self, config: dict):
        self.config = config
        self.detector = PedestrianDetector(config.get('detector', {}))
        self.tracker = PedestrianTracker(config.get('tracker', {}))
        self.transformer = CoordinateTransformer(config.get('coordinate_transform', {}))
        self.postprocessor = TrajectoryPostProcessor(config.get('postprocessing', {}))
        self.skip_frames = config.get('preprocessing', {}).get('skip_frames', 1)
    
    def process_video(self, video_path: str, 
                      roi: dict = None) -> TrajectoryData:
        """
        处理完整视频，输出标准化轨迹数据
        
        Args:
            video_path: 视频文件路径
            roi: 感兴趣区域 {"type": "polygon", "points": [...]}
        """
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.postprocessor.fps = fps
        
        # 收集原始轨迹
        raw_tracks = {}  # {track_id: [(frame, x, y, conf), ...]}
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 跳帧
            if frame_idx % self.skip_frames != 0:
                frame_idx += 1
                continue
            
            # ROI 裁剪 (可选)
            if roi:
                frame = self._apply_roi(frame, roi)
            
            # 检测
            detections = self.detector.detect(frame)
            
            # 跟踪
            tracks = self.tracker.update(detections, frame)
            
            # 收集轨迹点 (像素坐标)
            if len(tracks) > 0:
                for track in tracks:
                    x1, y1, x2, y2, track_id, conf = track[:6]
                    # 取底边中点作为行人位置
                    px = (x1 + x2) / 2
                    py = y2  # 脚部
                    
                    if track_id not in raw_tracks:
                        raw_tracks[track_id] = []
                    raw_tracks[track_id].append((frame_idx, px, py, conf))
            
            frame_idx += 1
        
        cap.release()
        
        # 坐标变换 (像素 → 世界)
        for track_id in raw_tracks:
            points = raw_tracks[track_id]
            pixel_coords = np.array([[p[1], p[2]] for p in points])
            world_coords = self.transformer.transform(pixel_coords)
            
            raw_tracks[track_id] = [
                (p[0], w[0], w[1], p[3]) 
                for p, w in zip(points, world_coords)
            ]
        
        # 后处理
        processed_tracks = self.postprocessor.process(raw_tracks)
        
        # 构建标准化输出
        return TrajectoryData(
            video_meta=VideoMetadata(
                source=str(video_path),
                fps=fps,
                total_frames=total_frames,
                resolution=(width, height),
                duration=total_frames / fps,
            ),
            tracks=processed_tracks,
        )
    
    def _apply_roi(self, frame: np.ndarray, roi: dict) -> np.ndarray:
        """应用 ROI 遮罩"""
        if roi.get('type') == 'polygon':
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            pts = np.array(roi['points'], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
            frame = cv2.bitwise_and(frame, frame, mask=mask)
        return frame
```

---

## 七、Plugin 注册与发现

```python
from typing import Protocol, runtime_checkable, Dict, Type
import importlib.metadata

@runtime_checkable
class VisionBackend(Protocol):
    """视觉后端协议接口"""
    def configure(self, config: dict) -> None: ...
    def process_video(self, video_path: str, roi: dict = None) -> TrajectoryData: ...
    def process_frame(self, frame: np.ndarray) -> list: ...
    
    @property
    def capabilities(self) -> set: ...

class VisionRegistry:
    """插件注册表"""
    
    _backends: Dict[str, Type] = {}
    
    @classmethod
    def register(cls, name: str):
        """装饰器注册"""
        def decorator(backend_class):
            cls._backends[name] = backend_class
            return backend_class
        return decorator
    
    @classmethod
    def discover(cls):
        """通过 entry_points 发现插件"""
        try:
            eps = importlib.metadata.entry_points(group='ped_agent.vision')
            for ep in eps:
                cls._backends[ep.name] = ep.load()
        except Exception:
            pass
    
    @classmethod
    def get(cls, name: str, config: dict) -> VisionBackend:
        """获取并初始化指定后端"""
        cls.discover()
        
        if name not in cls._backends:
            available = list(cls._backends.keys())
            raise ValueError(f"Unknown backend '{name}'. Available: {available}")
        
        backend = cls._backends[name](config)
        return backend
    
    @classmethod
    def list_available(cls) -> list:
        cls.discover()
        return list(cls._backends.keys())

# 内置插件注册
@VisionRegistry.register("yolo26_bytetrack")
class YOLO26ByteTrackBackend:
    """YOLO26 + ByteTrack 默认后端"""
    
    def __init__(self, config: dict):
        self.pipeline = VisionPipeline(config)
    
    def configure(self, config: dict) -> None:
        self.pipeline = VisionPipeline(config)
    
    def process_video(self, video_path: str, roi: dict = None) -> TrajectoryData:
        return self.pipeline.process_video(video_path, roi)
    
    def process_frame(self, frame: np.ndarray) -> list:
        return self.pipeline.detector.detect(frame)
    
    @property
    def capabilities(self) -> set:
        return {"detection", "tracking", "coordinate_transform"}
```

---

## 八、与分析管道对接

```python
# 使用示例：从视频到分析结果

from ped_agent.vision.registry import VisionRegistry
from ped_agent.analysis.pipeline import AnalysisPipeline

# 1. 视觉提取
vision = VisionRegistry.get("yolo26_bytetrack", config=vision_config)
trajectory_data = vision.process_video("./data/videos/station_exit.mp4")

# 2. 构建分析输入
scenario = ScenarioInput(
    metadata=ScenarioMetadata(
        scenario_id="station_exit_001",
        fps=trajectory_data.video_meta.fps,
        duration=trajectory_data.video_meta.duration,
        area_definition={"type": "rectangle", "bounds": [0, 20, 0, 15]},
    ),
    trajectories=trajectory_data.tracks,
    area_m2=20 * 15,  # 300 m²
)

# 3. 分析
pipeline = AnalysisPipeline(config=analysis_config)
result = pipeline.analyze_scenario(scenario)

# result 包含密度、速度、流量、OD矩阵、基本图等
```

---

## 九、ROI 配置文件格式

```json
{
  "roi_id": "station_platform_01",
  "type": "polygon",
  "points": [
    [120, 80], [900, 80], [900, 650], [120, 650]
  ],
  "coordinate_system": "pixel",
  "description": "车站站台候车区域",
  "cross_sections": [
    {
      "name": "entrance",
      "start": [120, 400],
      "end": [900, 400],
      "direction": "inbound"
    }
  ],
  "zones": [
    {
      "zone_id": "waiting_area",
      "polygon": [[120, 80], [500, 80], [500, 400], [120, 400]]
    },
    {
      "zone_id": "exit_area",
      "polygon": [[500, 400], [900, 400], [900, 650], [500, 650]]
    }
  ]
}
```

---

## 十、模块文件结构

```
src/ped_agent/vision/
├── __init__.py
├── interface.py           # VisionBackend Protocol 定义
├── registry.py            # VisionRegistry 插件注册
├── schemas.py             # TrajectoryData, VideoMetadata 等
├── pipeline.py            # VisionPipeline 主管道
├── detector.py            # PedestrianDetector (YOLO26)
├── tracker.py             # PedestrianTracker (ByteTrack/DeepSORT)
├── transform.py           # CoordinateTransformer 坐标变换
├── postprocess.py         # TrajectoryPostProcessor 后处理
└── plugins/
    ├── __init__.py
    ├── yolo26_bytetrack.py   # 默认: YOLO26 + ByteTrack
    └── yolo26_deepsort.py    # 备选: YOLO26 + DeepSORT
```

---

## 十一、可选依赖打包

```toml
# pyproject.toml
[project.optional-dependencies]
vision = [
    "ultralytics>=8.2",
    "boxmot>=10.0",
    "opencv-python>=4.8",
    "lap>=0.4",
]

[project.entry-points."ped_agent.vision"]
yolo26_bytetrack = "ped_agent.vision.plugins.yolo26_bytetrack:YOLO26ByteTrackBackend"
yolo26_deepsort = "ped_agent.vision.plugins.yolo26_deepsort:YOLO26DeepSORTBackend"
```

安装方式：
```bash
# 核心功能 (无视觉)
pip install ped-agent

# 含视觉模块
pip install ped-agent[vision]
```
