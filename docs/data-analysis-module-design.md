# 数据分析模块详细设计

## 一、模块概述

数据分析模块负责对行人流场景数据进行结构化分析，包括：
- 基础指标计算（密度、速度、流量）
- OD 矩阵构建
- 基本图关系分析
- 可视化输出

---

## 二、数据模型

### 2.1 输入数据结构

```python
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np

class Position(BaseModel):
    """二维位置"""
    x: float
    y: float
    
class PedestrianTrack(BaseModel):
    """单个行人轨迹"""
    track_id: int
    frames: List[int]  # 帧序号
    positions: List[Position]  # 位置序列 (世界坐标)
    timestamps: List[float]  # 时间戳 (秒)
    confidence: Optional[List[float]] = None  # 检测置信度
    
class ScenarioMetadata(BaseModel):
    """场景元数据"""
    scenario_id: str
    fps: float = 25.0  # 视频帧率
    duration: float  # 总时长 (秒)
    area_definition: dict  # 区域定义 {"type": "polygon", "vertices": [...]}
    
class ScenarioInput(BaseModel):
    """场景输入数据"""
    metadata: ScenarioMetadata
    trajectories: List[PedestrianTrack]
    area_m2: float  # 分析区域面积 (平方米)
```

### 2.2 输出数据结构

```python
class DensityMetrics(BaseModel):
    """密度指标"""
    mean_density: float  # 平均密度 (人/m²)
    max_density: float
    min_density: float
    std_density: float
    density_time_series: List[float]  # 时间序列
    
class VelocityMetrics(BaseModel):
    """速度指标"""
    mean_speed: float  # 平均速度 (m/s)
    max_speed: float
    min_speed: float
    std_speed: float
    speed_distribution: dict  # {"bins": [...], "counts": [...]}
    
class FlowMetrics(BaseModel):
    """流量指标"""
    flow_rate: float  # 流量 (人/秒)
    cross_section_id: str
    direction: str  # "inbound" | "outbound" | "bidirectional"
    
class ODMatrix(BaseModel):
    """起终点矩阵"""
    zones: List[str]  # 区域名称
    matrix: List[List[int]]  # OD 矩阵
    total_trips: int
    
class FundamentalDiagram(BaseModel):
    """基本图数据"""
    density_bins: List[float]  # 密度分箱
    flow_values: List[float]  # 对应流量
    speed_values: List[float]  # 对应速度
    model_params: dict  # 拟合模型参数
    r_squared: float  # 拟合优度
    
class AnalysisResult(BaseModel):
    """完整分析结果"""
    scenario_id: str
    density: DensityMetrics
    velocity: VelocityMetrics
    flows: List[FlowMetrics]
    od_matrix: Optional[ODMatrix] = None
    fundamental_diagram: Optional[FundamentalDiagram] = None
    visualizations: List[str]  # 图表文件路径
```

---

## 三、核心算法

### 3.1 密度计算

**Voronoi 方法** (局部密度)

```python
from scipy.spatial import Voronoi
import numpy as np

def compute_density_voronoi(positions: np.ndarray, area_bounds: tuple) -> np.ndarray:
    """
    使用 Voronoi 图计算局部密度
    
    Args:
        positions: (N, 2) 行人位置
        area_bounds: (xmin, xmax, ymin, ymax) 区域边界
        
    Returns:
        densities: (N,) 每个行人位置的局部密度 (人/m²)
    """
    if len(positions) < 3:
        return np.zeros(len(positions))
    
    # 构建 Voronoi 图
    vor = Voronoi(positions)
    
    densities = []
    for i, point_idx in enumerate(vor.point_region):
        region_idx = vor.regions[point_idx]
        
        # 跳过无界区域
        if -1 in region_idx or len(region_idx) == 0:
            densities.append(0.0)
            continue
        
        # 计算 Voronoi cell 面积
        vertices = vor.vertices[region_idx]
        area = polygon_area(vertices)
        
        # 密度 = 1 / 面积
        densities.append(1.0 / area if area > 0 else 0.0)
    
    return np.array(densities)

def polygon_area(vertices: np.ndarray) -> float:
    """Shoelace 公式计算多边形面积"""
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
```

**网格方法** (全局密度场)

```python
def compute_density_grid(positions: np.ndarray, 
                         grid_size: tuple = (50, 50),
                         area_bounds: tuple = None) -> np.ndarray:
    """
    网格化密度场计算
    
    Args:
        positions: (N, 2) 行人位置
        grid_size: (rows, cols) 网格分辨率
        area_bounds: (xmin, xmax, ymin, ymax)
        
    Returns:
        density_field: (rows, cols) 密度场 (人/m²)
    """
    if area_bounds is None:
        xmin, xmax = positions[:, 0].min(), positions[:, 0].max()
        ymin, ymax = positions[:, 1].min(), positions[:, 1].max()
    else:
        xmin, xmax, ymin, ymax = area_bounds
    
    # 计算每个网格的行人数量
    hist, xedges, yedges = np.histogram2d(
        positions[:, 0], positions[:, 1],
        bins=grid_size,
        range=[[xmin, xmax], [ymin, ymax]]
    )
    
    # 网格单元面积
    cell_width = (xmax - xmin) / grid_size[1]
    cell_height = (ymax - ymin) / grid_size[0]
    cell_area = cell_width * cell_height
    
    # 密度 = 数量 / 面积
    density_field = hist / cell_area
    
    return density_field.T  # 转置以匹配图像坐标系
```

### 3.2 速度场计算

```python
def compute_velocity_field(trajectories: List[PedestrianTrack],
                           grid_size: tuple = (50, 50),
                           area_bounds: tuple = None) -> tuple:
    """
    计算速度场 (向量场)
    
    Returns:
        velocity_x: (rows, cols) x 方向速度分量
        velocity_y: (rows, cols) y 方向速度分量
        speed_magnitude: (rows, cols) 速度大小
    """
    # 初始化累加器
    vx_sum = np.zeros(grid_size)
    vy_sum = np.zeros(grid_size)
    count = np.zeros(grid_size)
    
    for track in trajectories:
        positions = np.array([[p.x, p.y] for p in track.positions])
        timestamps = np.array(track.timestamps)
        
        if len(positions) < 2:
            continue
        
        # 计算瞬时速度 (中心差分)
        velocities = np.diff(positions, axis=0) / np.diff(timestamps)[:, None]
        
        # 位置对应到速度中点
        mid_positions = (positions[:-1] + positions[1:]) / 2
        
        # 分配到网格
        for pos, vel in zip(mid_positions, velocities):
            i, j = position_to_grid(pos, area_bounds, grid_size)
            if 0 <= i < grid_size[0] and 0 <= j < grid_size[1]:
                vx_sum[i, j] += vel[0]
                vy_sum[i, j] += vel[1]
                count[i, j] += 1
    
    # 平均速度
    with np.errstate(divide='ignore', invalid='ignore'):
        velocity_x = np.where(count > 0, vx_sum / count, 0)
        velocity_y = np.where(count > 0, vy_sum / count, 0)
    
    speed_magnitude = np.sqrt(velocity_x**2 + velocity_y**2)
    
    return velocity_x, velocity_y, speed_magnitude
```

### 3.3 流量计算

```python
class CrossSection:
    """横截线定义"""
    def __init__(self, start: tuple, end: tuple, name: str = ""):
        self.start = np.array(start)
        self.end = np.array(end)
        self.name = name
        self.length = np.linalg.norm(self.end - self.start)
        
def compute_flow_rate(trajectories: List[PedestrianTrack],
                      cross_section: CrossSection,
                      time_window: float = None) -> FlowMetrics:
    """
    计算通过横截线的流量
    
    Args:
        trajectories: 轨迹列表
        cross_section: 横截线定义
        time_window: 统计时间窗口 (秒)，None 表示全程
        
    Returns:
        FlowMetrics
    """
    crossings = []
    
    for track in trajectories:
        positions = np.array([[p.x, p.y] for p in track.positions])
        timestamps = np.array(track.timestamps)
        
        # 检测轨迹与截线的交点
        for i in range(len(positions) - 1):
            p1, p2 = positions[i], positions[i+1]
            
            if line_segment_intersect(p1, p2, cross_section.start, cross_section.end):
                # 线性插值交点时间
                t_cross = timestamps[i]  # 简化：取起点时间
                
                # 判断方向 (叉积)
                direction = np.cross(cross_section.end - cross_section.start, p2 - p1)
                
                crossings.append({
                    'time': t_cross,
                    'track_id': track.track_id,
                    'direction': 'inbound' if direction > 0 else 'outbound'
                })
    
    # 统计流量
    if time_window is None:
        time_window = max(t.timestamps[-1] for t in trajectories)
    
    flow_rate = len(crossings) / time_window
    
    # 方向统计
    inbound = sum(1 for c in crossings if c['direction'] == 'inbound')
    outbound = len(crossings) - inbound
    
    return FlowMetrics(
        flow_rate=flow_rate,
        cross_section_id=cross_section.name,
        direction='bidirectional' if inbound > 0 and outbound > 0 else 
                  ('inbound' if inbound > outbound else 'outbound')
    )

def line_segment_intersect(p1, p2, q1, q2) -> bool:
    """判断两线段是否相交"""
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    
    return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)
```

---

## 四、OD 矩阵构建

```python
from shapely.geometry import Point, Polygon

class Zone:
    """空间区域定义"""
    def __init__(self, zone_id: str, polygon: Polygon):
        self.zone_id = zone_id
        self.polygon = polygon
    
    def contains(self, point: tuple) -> bool:
        return self.polygon.contains(Point(point))

def build_od_matrix(trajectories: List[PedestrianTrack],
                    zones: List[Zone]) -> ODMatrix:
    """
    构建起终点矩阵
    
    算法：
    1. 识别每条轨迹的起点和终点所在区域
    2. 统计 (origin_zone, destination_zone) 对的数量
    """
    od_counts = np.zeros((len(zones), len(zones)), dtype=int)
    zone_names = [z.zone_id for z in zones]
    
    for track in trajectories:
        if len(track.positions) < 2:
            continue
        
        origin = (track.positions[0].x, track.positions[0].y)
        destination = (track.positions[-1].x, track.positions[-1].y)
        
        # 查找所属区域
        origin_idx = None
        dest_idx = None
        
        for i, zone in enumerate(zones):
            if zone.contains(origin):
                origin_idx = i
            if zone.contains(destination):
                dest_idx = i
        
        # 更新 OD 矩阵
        if origin_idx is not None and dest_idx is not None:
            od_counts[origin_idx, dest_idx] += 1
    
    return ODMatrix(
        zones=zone_names,
        matrix=od_counts.tolist(),
        total_trips=int(od_counts.sum())
    )
```

---

## 五、基本图关系分析

```python
from scipy.optimize import curve_fit

def compute_fundamental_diagram(trajectories: List[PedestrianTrack],
                                area_m2: float,
                                time_bin: float = 1.0) -> FundamentalDiagram:
    """
    计算流量-密度基本图
    
    方法：
    1. 按时间分箱
    2. 每个时间窗计算瞬时密度和流量
    3. 拟合 Weidmann 模型或线性模型
    """
    # 时间范围
    t_max = max(max(t.timestamps) for t in trajectories if t.timestamps)
    time_bins = np.arange(0, t_max, time_bin)
    
    densities = []
    flows = []
    speeds = []
    
    for t_start in time_bins:
        t_end = t_start + time_bin
        
        # 统计该时间窗内的行人
        pedestrians_in_window = []
        velocities_in_window = []
        
        for track in trajectories:
            for i, t in enumerate(track.timestamps):
                if t_start <= t < t_end:
                    pedestrians_in_window.append(track.track_id)
                    
                    # 计算速度
                    if i > 0:
                        dx = track.positions[i].x - track.positions[i-1].x
                        dy = track.positions[i].y - track.positions[i-1].y
                        dt = track.timestamps[i] - track.timestamps[i-1]
                        v = np.sqrt(dx**2 + dy**2) / dt if dt > 0 else 0
                        velocities_in_window.append(v)
                    break
        
        # 瞬时密度 = 行人数 / 面积
        n_peds = len(set(pedestrians_in_window))
        density = n_peds / area_m2
        
        # 瞬时流量 = 密度 × 平均速度
        avg_speed = np.mean(velocities_in_window) if velocities_in_window else 0
        flow = density * avg_speed
        
        if density > 0:  # 过滤空窗口
            densities.append(density)
            flows.append(flow)
            speeds.append(avg_speed)
    
    densities = np.array(densities)
    flows = np.array(flows)
    speeds = np.array(speeds)
    
    # 拟合 Weidmann 模型: v = v_free * (1 - exp(-gamma * (1/rho - 1/rho_max)))
    # 简化为线性: flow = a * density + b
    def linear_model(rho, a, b):
        return a * rho + b
    
    try:
        params, _ = curve_fit(linear_model, densities, flows, p0=[1.0, 0])
        r_squared = 1 - (np.sum((flows - linear_model(densities, *params))**2) / 
                         np.sum((flows - flows.mean())**2))
    except:
        params = [0, 0]
        r_squared = 0
    
    return FundamentalDiagram(
        density_bins=densities.tolist(),
        flow_values=flows.tolist(),
        speed_values=speeds.tolist(),
        model_params={'a': params[0], 'b': params[1]},
        r_squared=r_squared
    )
```

---

## 六、分析管道编排

```python
class AnalysisPipeline:
    """分析管道主类"""
    
    def __init__(self, config: dict):
        self.config = config
        self.visualizer = Visualizer(config.get('visualization', {}))
    
    def analyze_scenario(self, scenario: ScenarioInput) -> AnalysisResult:
        """完整场景分析"""
        
        # 1. 密度分析
        density_metrics = self._compute_density(scenario)
        
        # 2. 速度分析
        velocity_metrics = self._compute_velocity(scenario)
        
        # 3. 流量分析
        flow_metrics = self._compute_flows(scenario)
        
        # 4. OD 矩阵 (可选)
        od_matrix = None
        if 'zones' in scenario.metadata.area_definition:
            od_matrix = self._compute_od_matrix(scenario)
        
        # 5. 基本图 (可选)
        fundamental_diagram = None
        if self.config.get('compute_fundamental_diagram', True):
            fundamental_diagram = compute_fundamental_diagram(
                scenario.trajectories,
                scenario.area_m2
            )
        
        # 6. 生成可视化
        viz_paths = self.visualizer.generate_all(
            scenario, density_metrics, velocity_metrics, flow_metrics, fundamental_diagram
        )
        
        return AnalysisResult(
            scenario_id=scenario.metadata.scenario_id,
            density=density_metrics,
            velocity=velocity_metrics,
            flows=flow_metrics,
            od_matrix=od_matrix,
            fundamental_diagram=fundamental_diagram,
            visualizations=viz_paths
        )
```

---

## 七、可视化设计

### 7.1 图表类型规划

| 图表类型 | 用途 | 库 | 输出格式 |
|---------|------|------|---------|
| 密度热力图 | 空间密度分布 | Plotly | HTML (交互) |
| 速度矢量场 | 行人流方向与速度 | Matplotlib | PNG/SVG |
| 基本图散点图 | 流量-密度/速度-密度关系 | Plotly | HTML |
| OD 矩阵桑基图 | 起终点流量分布 | Plotly | HTML |
| 时间序列折线图 | 密度/流量随时间变化 | Plotly | HTML |
| 轨迹叠加图 | 行人运动路径 | Matplotlib | PNG/SVG |
| 速度分布直方图 | 速度统计分布 | Plotly | HTML |

### 7.2 可视化器实现

```python
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

class Visualizer:
    """行人流数据可视化"""
    
    def __init__(self, config: dict):
        self.output_dir = Path(config.get('output_dir', './outputs/figures'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.theme = config.get('theme', 'plotly_white')
        self.dpi = config.get('dpi', 150)
    
    def generate_all(self, scenario, density, velocity, flows, fd) -> List[str]:
        """生成全套可视化"""
        paths = []
        paths.append(self.density_heatmap(scenario, density))
        paths.append(self.velocity_field(scenario))
        paths.append(self.trajectory_overlay(scenario))
        paths.append(self.time_series(density, velocity))
        paths.append(self.speed_distribution(velocity))
        if fd:
            paths.append(self.fundamental_diagram_plot(fd))
        return [str(p) for p in paths if p]
    
    def density_heatmap(self, scenario: ScenarioInput, 
                        density: DensityMetrics) -> Path:
        """密度热力图 (Plotly 交互式)"""
        positions = []
        for track in scenario.trajectories:
            for pos in track.positions:
                positions.append([pos.x, pos.y])
        positions = np.array(positions)
        
        fig = go.Figure(data=go.Histogram2d(
            x=positions[:, 0],
            y=positions[:, 1],
            colorscale='YlOrRd',
            colorbar=dict(title='密度 (人/m²)'),
            nbinsx=50,
            nbinsy=50,
        ))
        
        fig.update_layout(
            title=f'场景 {scenario.metadata.scenario_id} - 密度分布',
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            template=self.theme,
            width=800, height=600,
        )
        
        path = self.output_dir / f'{scenario.metadata.scenario_id}_density.html'
        fig.write_html(str(path))
        return path
    
    def velocity_field(self, scenario: ScenarioInput) -> Path:
        """速度矢量场 (Matplotlib)"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=self.dpi)
        
        vx, vy, mag = compute_velocity_field(
            scenario.trajectories, grid_size=(20, 20)
        )
        
        # 网格坐标
        rows, cols = vx.shape
        x = np.linspace(0, 1, cols)
        y = np.linspace(0, 1, rows)
        X, Y = np.meshgrid(x, y)
        
        # 绘制矢量场
        quiver = ax.quiver(X, Y, vx, vy, mag, cmap='coolwarm', scale=20)
        ax.set_title(f'场景 {scenario.metadata.scenario_id} - 速度矢量场')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        plt.colorbar(quiver, ax=ax, label='速度 (m/s)')
        
        path = self.output_dir / f'{scenario.metadata.scenario_id}_velocity_field.png'
        fig.savefig(str(path), bbox_inches='tight')
        plt.close(fig)
        return path
    
    def trajectory_overlay(self, scenario: ScenarioInput) -> Path:
        """轨迹叠加图 (Matplotlib)"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=self.dpi)
        
        cmap = plt.cm.get_cmap('tab20', min(len(scenario.trajectories), 20))
        
        for i, track in enumerate(scenario.trajectories):
            positions = np.array([[p.x, p.y] for p in track.positions])
            color = cmap(i % 20)
            ax.plot(positions[:, 0], positions[:, 1], 
                    color=color, alpha=0.6, linewidth=0.8)
            # 起点标记
            ax.scatter(positions[0, 0], positions[0, 1], 
                       color=color, marker='o', s=15, zorder=5)
        
        ax.set_title(f'场景 {scenario.metadata.scenario_id} - 行人轨迹')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_aspect('equal')
        
        path = self.output_dir / f'{scenario.metadata.scenario_id}_trajectories.png'
        fig.savefig(str(path), bbox_inches='tight')
        plt.close(fig)
        return path
    
    def fundamental_diagram_plot(self, fd: FundamentalDiagram) -> Path:
        """基本图 (Plotly 交互式)"""
        from plotly.subplots import make_subplots
        
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=('流量-密度关系', '速度-密度关系'))
        
        # 流量-密度
        fig.add_trace(go.Scatter(
            x=fd.density_bins, y=fd.flow_values,
            mode='markers', name='实测数据',
            marker=dict(size=5, opacity=0.6)
        ), row=1, col=1)
        
        # 拟合曲线
        rho_fit = np.linspace(min(fd.density_bins), max(fd.density_bins), 100)
        flow_fit = fd.model_params['a'] * rho_fit + fd.model_params['b']
        fig.add_trace(go.Scatter(
            x=rho_fit.tolist(), y=flow_fit.tolist(),
            mode='lines', name=f'拟合 (R²={fd.r_squared:.3f})',
            line=dict(color='red', width=2)
        ), row=1, col=1)
        
        # 速度-密度
        fig.add_trace(go.Scatter(
            x=fd.density_bins, y=fd.speed_values,
            mode='markers', name='速度',
            marker=dict(size=5, opacity=0.6, color='green')
        ), row=1, col=2)
        
        fig.update_xaxes(title_text='密度 (人/m²)', row=1, col=1)
        fig.update_yaxes(title_text='流量 (人/m·s)', row=1, col=1)
        fig.update_xaxes(title_text='密度 (人/m²)', row=1, col=2)
        fig.update_yaxes(title_text='速度 (m/s)', row=1, col=2)
        fig.update_layout(template=self.theme, width=1200, height=500)
        
        path = self.output_dir / 'fundamental_diagram.html'
        fig.write_html(str(path))
        return path
    
    def time_series(self, density: DensityMetrics, 
                    velocity: VelocityMetrics) -> Path:
        """密度/速度时间序列"""
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=('密度变化', '速度变化'))
        
        t = list(range(len(density.density_time_series)))
        
        fig.add_trace(go.Scatter(
            x=t, y=density.density_time_series,
            mode='lines', name='密度',
            line=dict(color='orange')
        ), row=1, col=1)
        
        fig.update_xaxes(title_text='时间 (s)', row=2, col=1)
        fig.update_yaxes(title_text='密度 (人/m²)', row=1, col=1)
        fig.update_yaxes(title_text='速度 (m/s)', row=2, col=1)
        fig.update_layout(template=self.theme, height=600)
        
        path = self.output_dir / 'time_series.html'
        fig.write_html(str(path))
        return path
    
    def speed_distribution(self, velocity: VelocityMetrics) -> Path:
        """速度分布直方图"""
        fig = go.Figure(data=go.Bar(
            x=velocity.speed_distribution.get('bins', []),
            y=velocity.speed_distribution.get('counts', []),
            marker_color='steelblue'
        ))
        
        fig.update_layout(
            title='速度分布',
            xaxis_title='速度 (m/s)',
            yaxis_title='频次',
            template=self.theme,
        )
        
        path = self.output_dir / 'speed_distribution.html'
        fig.write_html(str(path))
        return path
    
    def od_sankey(self, od_matrix: ODMatrix) -> Path:
        """OD 矩阵桑基图"""
        sources, targets, values = [], [], []
        n = len(od_matrix.zones)
        
        for i in range(n):
            for j in range(n):
                if od_matrix.matrix[i][j] > 0:
                    sources.append(i)
                    targets.append(n + j)
                    values.append(od_matrix.matrix[i][j])
        
        labels = [f'O: {z}' for z in od_matrix.zones] + \
                 [f'D: {z}' for z in od_matrix.zones]
        
        fig = go.Figure(data=go.Sankey(
            node=dict(label=labels),
            link=dict(source=sources, target=targets, value=values)
        ))
        
        fig.update_layout(title='起终点流量分布 (OD 桑基图)')
        
        path = self.output_dir / 'od_sankey.html'
        fig.write_html(str(path))
        return path
```

---

## 八、视觉模块说明

### 8.1 目标检测模型选型

本项目视觉模块计划使用 **YOLO26** 作为行人检测模型。

> YOLO26 是 Ultralytics YOLO 系列的 end-to-end / NMS-free 模型路线。行人流场景以离线高精度分析为主，默认选择 `yolo26x.pt`；实时或边缘部署可降级为 `yolo26n/s/m`。

**依赖安装**：
```bash
pip install ultralytics  # YOLO26 通过 ultralytics 包发布
```

**模型使用**：
```python
from ultralytics import YOLO

# 加载 YOLO26 模型
model = YOLO('yolo26x.pt')  # x 版本 - 最高精度

# 行人检测 (COCO class 0 = person)
results = model(frame, classes=[0], conf=0.5, end2end=True)
detections = results[0].boxes  # xyxy, conf, cls
```

### 8.2 与分析管道的衔接

```
视频输入 → YOLO26 检测 → ByteTrack/DeepSORT 跟踪 
→ 坐标变换 (像素→世界) → TrajectoryData 
→ AnalysisPipeline.analyze_scenario()
→ 可视化输出
```

视觉模块输出标准化的 `TrajectoryData` 对象后，分析管道无需关心轨迹来源（可以是视频提取，也可以是手动标注或仿真数据）。

### 8.3 配置示例

```yaml
# config/vision.yaml
vision:
  enabled: true
  detector:
    model: "yolo26x.pt"        # YOLO26 模型权重
    confidence: 0.5            # 检测置信度阈值
    iou_threshold: 0.45        # NMS IoU 阈值
    classes: [0]               # 仅检测行人 (COCO person)
    input_size: 1280           # 推理分辨率
    device: "cuda:0"           # 推理设备
    end2end: true              # YOLO26 默认 end-to-end / NMS-free 推理
  tracker:
    algorithm: "bytetrack"     # "bytetrack" | "deepsort"
    track_buffer: 30           # 丢失目标保持帧数
    match_thresh: 0.8          # 匹配阈值
  coordinate_transform:
    method: "homography"       # "homography" | "none"
    calibration_file: null     # 标定文件路径
    pixel_per_meter: null      # 简易缩放比 (无标定时)
  output:
    save_video: false          # 是否保存标注视频
    trajectory_format: "json"  # 轨迹输出格式
```

---

## 九、统计分析工具

### 9.1 分布拟合

```python
from scipy import stats

def fit_speed_distribution(speeds: np.ndarray) -> dict:
    """
    拟合速度分布
    
    行人自由流速度通常服从正态分布：
    v ~ N(mu, sigma²), 典型值 mu=1.34 m/s, sigma=0.26 m/s (Weidmann 1993)
    """
    # 正态分布拟合
    mu, sigma = stats.norm.fit(speeds)
    
    # K-S 检验
    ks_stat, p_value = stats.kstest(speeds, 'norm', args=(mu, sigma))
    
    # 对数正态 (适合混合人群)
    shape, loc, scale = stats.lognorm.fit(speeds, floc=0)
    
    return {
        'normal': {'mu': mu, 'sigma': sigma, 'ks_p': p_value},
        'lognormal': {'shape': shape, 'scale': scale},
        'percentiles': {
            'p15': np.percentile(speeds, 15),
            'p50': np.percentile(speeds, 50),
            'p85': np.percentile(speeds, 85),
        }
    }
```

### 9.2 服务水平 (Level of Service)

```python
def compute_los(density: float) -> str:
    """
    根据密度计算行人服务水平 (HCM 标准)
    
    | LOS | 密度范围 (人/m²) | 描述 |
    |-----|------------------|------|
    | A   | < 0.27           | 自由流 |
    | B   | 0.27 - 0.43      | 轻微限制 |
    | C   | 0.43 - 0.72      | 限制 |
    | D   | 0.72 - 1.08      | 严重限制 |
    | E   | 1.08 - 2.17      | 拥挤 |
    | F   | > 2.17           | 拥堵/停滞 |
    """
    if density < 0.27:
        return 'A'
    elif density < 0.43:
        return 'B'
    elif density < 0.72:
        return 'C'
    elif density < 1.08:
        return 'D'
    elif density < 2.17:
        return 'E'
    else:
        return 'F'
```

---

## 十、模块文件结构总结

```
src/ped_agent/analysis/
├── __init__.py
├── pipeline.py              # AnalysisPipeline 主编排
├── metrics.py               # 密度/速度/流量计算
├── od_matrix.py             # OD 矩阵构建
├── fundamental_diagram.py   # 基本图拟合 (Weidmann 等模型)
├── statistics.py            # 分布拟合、假设检验、LOS 评定
├── visualizer.py            # Plotly/Matplotlib 图表生成
└── schemas.py               # 本模块 Pydantic 数据模型
```
