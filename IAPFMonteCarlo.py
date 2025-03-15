# 蒙特卡洛实验
import numpy as np
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.spatial import KDTree
import matplotlib.gridspec as gridspec
from tqdm import tqdm


# ====================== 碰撞管理类 ======================
class CollisionManager:
    def __init__(self, display_duration=2.0):
        self.collisions = []
        self.display_duration = display_duration

    def add_collision(self, position, time):
        self.collisions.append({
            'position': np.array(position),
            'time': time,
            'marker': None
        })

    def update(self, current_time, ax):
        to_remove = []
        for coll in self.collisions:
            if current_time - coll['time'] > self.display_duration:
                if coll['marker'] is not None:
                    coll['marker'].remove()
                to_remove.append(coll)
            elif coll['marker'] is None:
                coll['marker'] = ax.scatter(*coll['position'], c='red',
                                            marker='*', s=100, alpha=0.8)
        for coll in to_remove:
            self.collisions.remove(coll)


# ====================== 势场核心函数 ======================
def AttractiveF(coord, goal, zeta):
    return -zeta * (coord - goal)

def RepulsiveF(coord, obst, velocity=None, rho=0, eta=0, disturb_gain=0.3):  # 新增参数
    if velocity is None:
        velocity = np.zeros(3)

    coord = np.asarray(coord).flatten()
    obst = np.asarray(obst).flatten()
    velocity = np.asarray(velocity).flatten()

    dist_vec = coord - obst
    dist = np.linalg.norm(dist_vec)

    speed_rep = np.zeros(3)
    dir_rep = np.zeros(3)

    if dist < rho and dist > 1e-3:
        dir_normal = dist_vec / dist
        velocity_norm = np.linalg.norm(velocity)

        if velocity_norm > 1e-3:
            speed_dir = velocity / velocity_norm
        else:
            speed_dir = np.zeros(3)

        decay_factor = np.exp(-dist / rho) - np.exp(-1)
        factor = eta * decay_factor

        # ============= 新增共线检测逻辑 =============
        is_collinear = False
        if velocity_norm > 1e-3 and np.linalg.norm(dir_normal) > 1e-3:
            # 计算路径方向与障碍物方向的余弦相似度
            cos_sim = np.dot(speed_dir, dir_normal)
            is_collinear = abs(cos_sim) > 0.95  # 阈值设为0.95

        if is_collinear:
            # 生成横向扰动方向（垂直于速度方向）
            if np.linalg.norm(speed_dir) > 1e-3:
                lateral_dir = np.cross(speed_dir, [0, 0, 1])  # 假设在XY平面运动
                lateral_dir /= np.linalg.norm(lateral_dir)
                dir_rep = disturb_gain * (1 - dist / rho) * factor * lateral_dir
            speed_rep = np.zeros(3)  # 抑制原有速度方向斥力
        else:
            # 原有计算逻辑
            speed_ratio = np.dot(dir_normal, speed_dir)
            speed_rep = -factor * speed_dir * abs(speed_ratio)

            if velocity_norm > 1e-3:
                tangent_dir = dir_normal - np.dot(dir_normal, speed_dir) * speed_dir
                tangent_norm = np.linalg.norm(tangent_dir)
                if tangent_norm > 1e-3:
                    tangent_dir = tangent_dir / tangent_norm
                else:
                    tangent_dir = np.zeros(3)
                dir_rep = factor * tangent_dir

    return np.asarray(speed_rep).flatten(), np.asarray(dir_rep).flatten()

# ====================== 三维向量类 ======================
class Vector3d:
    def __init__(self, x, y, z):
        self.deltaX = x
        self.deltaY = y
        self.deltaZ = z
        self.length = -1
        self.direction = [0, 0, 0]
        self.update()

    def update(self):
        self.length = math.sqrt(self.deltaX ** 2 + self.deltaY ** 2 + self.deltaZ ** 2)
        if self.length > 0:
            self.direction = [
                self.deltaX / self.length,
                self.deltaY / self.length,
                self.deltaZ / self.length
            ]
        else:
            self.direction = None

    def __add__(self, other):
        return Vector3d(self.deltaX + other.deltaX,
                        self.deltaY + other.deltaY,
                        self.deltaZ + other.deltaZ)

    def __sub__(self, other):
        return Vector3d(self.deltaX - other.deltaX,
                        self.deltaY - other.deltaY,
                        self.deltaZ - other.deltaZ)

    def __mul__(self, scalar):
        return Vector3d(self.deltaX * scalar,
                        self.deltaY * scalar,
                        self.deltaZ * scalar)

    def __truediv__(self, scalar):
        return self.__mul__(1.0 / scalar)

    def __repr__(self):
        return f"Vector3d({self.deltaX}, {self.deltaY}, {self.deltaZ}, length={self.length})"


# ====================== 编队生成函数 ======================
def formation(x, y, z, num):
    center = np.array([x, y, z])
    angle = 2 * np.pi / num
    radius = 2.0
    positions = []
    for i in range(num):
        theta = i * angle
        dx = radius * np.cos(theta)
        dy = radius * np.sin(theta)
        positions.append(center + np.array([dx, dy, 0]))
    return center, positions


# ====================== 路径规划类 ======================
class IAPT3D:
    def __init__(self, start, goal, k_att, k_rep, rr, step_size, max_iters, goal_threshold):
        self.start = Vector3d(*start)
        self.goal = Vector3d(*goal)
        self.k_att = k_att
        self.k_rep = k_rep
        self.rr = rr
        self.step_size = step_size
        self.max_iters = max_iters
        self.goal_threshold = goal_threshold
        self.path = []
        self.is_path_plan_success = False

    def attractive(self, position):
        return (self.goal - position) * self.k_att

    def repulsion(self, position, obstacles):
        total_repulsion = Vector3d(0, 0, 0)
        for ob in obstacles:
            obst_pos = Vector3d(*ob[:3])
            obst_radius = ob[3]
            obs_to_pos_vec = position - obst_pos
            dist_to_center = obs_to_pos_vec.length
            effective_dist = dist_to_center - obst_radius
            if effective_dist < self.rr and effective_dist > 1e-3:
                pos_to_goal = self.goal - position
                obs_to_pos = obs_to_pos_vec * (effective_dist / dist_to_center)
                rep_1 = (obs_to_pos * (1.0 / effective_dist - 1.0 / self.rr) /
                         (effective_dist ** 2)) * (pos_to_goal.length ** 2)
                rep_2 = (pos_to_goal * ((1.0 / effective_dist - 1.0 / self.rr) ** 2)) * pos_to_goal.length
                total_repulsion += rep_1 + rep_2
        return total_repulsion * self.k_rep

    def compute_next_step(self, current_position, obstacles):
        position = Vector3d(*current_position)
        f_att = self.attractive(position)
        f_rep = self.repulsion(position, obstacles)
        f_total = f_att + f_rep
        if f_total.length == 0:
            return (position.deltaX, position.deltaY, position.deltaZ)
        next_position = position + (f_total / f_total.length) * self.step_size
        return (next_position.deltaX, next_position.deltaY, next_position.deltaZ)


# ====================== 编队控制类 ======================
class FormationControl:
    def __init__(self, num_followers, radius, params):
        self.num_followers = num_followers
        self.formation_radius = radius
        self.params = params
        self.followers_paths = [[] for _ in range(num_followers)]
        self.obst_tree = None
        self.formation_snapshots = []
        self.last_formation_center = None
        self.save_interval = params.get('save_interval', 2.0)
        self.current_time = 0.0

        # ========== 新增属性 ==========
        self.formation_stable = True  # 编队稳定性标志
        self.total_distance = np.zeros(num_followers)  # 各跟随者累计移动距离
        self.last_positions = [None] * num_followers  # 记录上一时刻位置

    def calculate_target_positions(self, leader_pos):
        f_center, form_coord = formation(
            leader_pos[0], leader_pos[1], leader_pos[2],
            self.num_followers
        )
        return f_center, form_coord

    def calculate_repulsions(self, current_pos, velocity, obstacles, idx, followers):
        speed_rep_sum = np.zeros(3)
        dir_rep_sum = np.zeros(3)

        near_obs = self.obst_tree.query_ball_point(current_pos, self.params['rho'])
        for k in near_obs:
            s_rep, d_rep = RepulsiveF(current_pos, obstacles[k], velocity,
                                      self.params['rho'], self.params['eta'],
                                      self.params.get('disturb_gain', 0.3))
            speed_rep_sum += s_rep
            dir_rep_sum += d_rep

        for j in range(self.num_followers):
            if j != idx and np.linalg.norm(current_pos - followers[j]) < self.params['rho_rob']:
                s_rep, d_rep = RepulsiveF(current_pos, followers[j], velocity,
                                          self.params['rho_rob'], self.params['eta_rob'])
                speed_rep_sum += s_rep
                dir_rep_sum += d_rep

        return speed_rep_sum, dir_rep_sum

    def update_followers(self, leader_pos, followers_positions, obstacles, is_extension=False):
        _, targets = self.calculate_target_positions(leader_pos)
        self.obst_tree = KDTree(obstacles)

        alpha = self.params.get('alpha', 1.0)
        w_min = self.params.get('w_min', 0.1)
        N = self.num_followers

        deviations = [np.linalg.norm(followers_positions[i] - targets[i]) for i in range(N)]
        weights = np.array([(d ** -alpha if d > 1e-3 else 0) for d in deviations])
        total_weight = np.sum(weights)

        if total_weight > 0:
            weights = np.maximum(weights / total_weight, w_min)
            weights /= np.sum(weights)
        else:
            weights = np.ones(N) / N

        Cf = np.zeros(3)
        for i in range(N):
            Cf += weights[i] * followers_positions[i]
        self.last_formation_center = Cf

        # ========== 新增编队稳定性检测 ==========
        for i in range(self.num_followers):
            current_pos = followers_positions[i]
            # 检测到编队中心的距离
            distance_to_cf = np.linalg.norm(current_pos - Cf)
            if distance_to_cf > 6.0:  # 超过6米判定为编队失稳
                self.formation_stable = False

            # 计算跟随者移动距离
            if self.last_positions[i] is not None:
                delta = np.linalg.norm(current_pos - self.last_positions[i])
                self.total_distance[i] += delta
            self.last_positions[i] = current_pos.copy()

        # ========== 原有逻辑继续执行 ==========
        for i in range(self.num_followers):
            current_pos = followers_positions[i]
            if len(self.followers_paths[i]) >= 2:
                prev_pos = self.followers_paths[i][-2]
                velocity = (current_pos - prev_pos) / self.params['dt']
            else:
                velocity = np.zeros(3)

            F_att_target = AttractiveF(current_pos, targets[i], self.params['zeta_robot'])
            F_att_formation = AttractiveF(current_pos, Cf, self.params['zeta_formation'] * 0.5)
            F_att = F_att_target + F_att_formation

            speed_rep_sum, dir_rep_sum = self.calculate_repulsions(current_pos, velocity, obstacles, i,
                                                                   followers_positions)

            z_error = leader_pos[2] - current_pos[2]
            F_z_constraint = np.array([0, 0, self.params['zeta_z'] * z_error])

            F_total = (F_att +
                       self.params['speed_rep_weight'] * speed_rep_sum +
                       self.params['dir_rep_weight'] * dir_rep_sum +
                       F_z_constraint)

            F_norm = np.linalg.norm(F_total)
            if F_norm > self.params['max_force']:
                F_total = F_total / F_norm * self.params['max_force']

            new_pos = current_pos + F_total * self.params['dt']
            followers_positions[i] = new_pos
            self.followers_paths[i].append(new_pos.copy())

        self.current_time += self.params['dt']
        if self.current_time >= self.save_interval or is_extension:
            snapshot = {
                'leader': np.array(leader_pos),
                'followers': [np.array(pos).copy() for pos in followers_positions]
            }
            self.formation_snapshots.append(snapshot)
            self.current_time = 0.0


class EnhancedVisualizer:
    def __init__(self, leader_path, followers_paths, formation_snapshots,
                 dt, static_obstacles, dynamic_config, collision_mgr):
        # 转换路径为NumPy数组（确保后续操作正确）
        self.leader_path = np.array(leader_path)
        self.followers_paths = [np.array(p) for p in followers_paths]

        # 转换编队快照数据结构
        self.formation_snapshots = [{
            'leader': np.array(snap['leader']),
            'followers': [np.array(f) for f in snap['followers']]
        } for snap in formation_snapshots]

        self.dt = dt
        self.static_obstacles = static_obstacles
        self.dynamic_config = dynamic_config
        self.collision_mgr = collision_mgr

        # 动态轨迹数据存储
        self.dynamic_trails = self._generate_dynamic_trails()

        # 动态可视化初始化
        self.dynamic_fig = plt.figure(figsize=(15, 10))
        self.dynamic_ax = self.dynamic_fig.add_subplot(111, projection='3d')
        self._init_dynamic_plot()

    def _generate_dynamic_trails(self):
        """生成动态障碍物轨迹数据"""
        trails = []
        total_time = len(self.leader_path) * self.dt

        for cfg in self.dynamic_config:
            obstacle_trail = []
            time_points = np.arange(0, total_time + 0.1, 0.1)
            time_points = np.append(time_points, total_time)

            prev_pos = None
            min_dist = cfg['obstacle_radius'] * 0.6  # 最小绘制间距

            for t in np.unique(time_points):
                pos = self._get_obstacle_position(cfg, t)
                current_pos = np.array(pos[:3])

                # 过滤相近点
                if prev_pos is None or np.linalg.norm(current_pos - prev_pos) >= min_dist:
                    alpha = 0.8 if np.isclose(t, total_time) else 0.3
                    obstacle_trail.append({
                        'position': current_pos,
                        'radius': cfg['obstacle_radius'],
                        'color': self._get_obstacle_color(cfg['type']),
                        'alpha': alpha
                    })
                    prev_pos = current_pos

            trails.append(obstacle_trail)
        return trails

    def _get_obstacle_color(self, obs_type):
        """获取障碍物颜色"""
        color_map = {
            'circular': 'blue',
            'linear': 'orange',
            'reciprocating': 'purple'
        }
        return color_map.get(obs_type, 'gray')

    def _get_obstacle_position(self, cfg, t):
        """计算单个障碍物位置"""
        if cfg['type'] == 'circular':
            theta = cfg['speed'] * t
            x = cfg['center'][0] + cfg['radius'] * np.cos(theta)
            y = cfg['center'][1] + cfg['radius'] * np.sin(theta)
            z = cfg['center'][2]
            return (x, y, z)

        elif cfg['type'] == 'linear':
            start = np.array(cfg['start'])
            end = np.array(cfg['end'])
            direction = (end - start) / np.linalg.norm(end - start)
            distance = min(cfg['speed'] * t, np.linalg.norm(end - start))
            return start + direction * distance

        elif cfg['type'] == 'reciprocating':
            center = np.array(cfg['center'])
            total_dist = cfg['speed'] * t
            cycle_dist = cfg['amplitude'] * 2
            mod_dist = total_dist % cycle_dist

            if mod_dist <= cfg['amplitude']:
                displacement = mod_dist
            else:
                displacement = cycle_dist - mod_dist
            displacement -= cfg['amplitude']

            if cfg['axis'] == 'x':
                return (center[0] + displacement, center[1], center[2])
            else:
                return (center[0], center[1] + displacement, center[2])

        return (0, 0, 0)

    def _plot_static_3d(self, ax):
        """3D静态视图"""
        # 绘制静态障碍物
        for ob in self.static_obstacles:
            self._plot_sphere(ax, ob[:3], ob[3], 'gray', alpha=0.4)

        # 绘制动态轨迹
        for trail in self.dynamic_trails:
            for point in trail:
                self._plot_sphere(ax, point['position'], point['radius'],
                                  point['color'], point['alpha'])

        # 绘制编队连线
        for snapshot in self.formation_snapshots:
            points = np.array([snapshot['leader']] + snapshot['followers'])
            for conn in [[0, 1], [0, 2], [0, 3], [1, 2], [2, 3], [3, 1]]:
                ax.plot(points[conn, 0], points[conn, 1], points[conn, 2],
                        color='lime', alpha=0.6, lw=0.8)

        # 绘制路径
        ax.plot(self.leader_path[:, 0], self.leader_path[:, 1], self.leader_path[:, 2],
                'r-', lw=2, label='Leader')
        for i, path in enumerate(self.followers_paths):
            ax.plot(path[:, 0], path[:, 1], path[:, 2], 'b--', alpha=0.5,
                    label=f'Follower {i + 1}' if i == 0 else "")

        ax.scatter(0, 0, 0, c='g', s=100, marker='*')
        ax.scatter(15, 15, 15, c='orange', s=100, marker='*')
        ax.set(xlim=[-2.5, 20], ylim=[-2.5, 20], zlim=[-2.5, 20],  # 修改这里
               xlabel='X', ylabel='Y', zlabel='Z')
        ax.legend()

    def _plot_static_2d(self, ax, projection='xy'):
        """二维投影视图"""
        # 绘制静态障碍物
        for ob in self.static_obstacles:
            x, y, z, r = ob
            circle = plt.Circle((x, y if projection == 'xy' else z), r,
                                color='gray', alpha=0.4)
            ax.add_patch(circle)

        # 绘制动态轨迹
        for trail in self.dynamic_trails:
            for point in trail:
                x, y, z = point['position']
                radius = point['radius']
                if projection == 'xy':
                    pos = (x, y)
                else:
                    pos = (x, z)

                circle = plt.Circle(pos, radius, color=point['color'],
                                    alpha=point['alpha'])
                ax.add_patch(circle)

        # 绘制编队连线
        for snapshot in self.formation_snapshots:
            points = np.array([snapshot['leader']] + snapshot['followers'])
            for conn in [[0, 1], [0, 2], [0, 3], [1, 2], [2, 3], [3, 1]]:
                if projection == 'xy':
                    ax.plot(points[conn, 0], points[conn, 1],
                            color='lime', alpha=0.6, lw=0.8)
                else:
                    ax.plot(points[conn, 0], points[conn, 2],
                            color='lime', alpha=0.6, lw=0.8)

        # 绘制路径
        ax.plot(self.leader_path[:, 0], self.leader_path[:, 1 if projection == 'xy' else 2],
                'r-', lw=2, label='Leader')
        for i, path in enumerate(self.followers_paths):
            ax.plot(path[:, 0], path[:, 1 if projection == 'xy' else 2],
                    'b--', alpha=0.5)

        ax.scatter(0, 0, c='g', s=100, marker='*')
        ax.scatter(15, 15, c='orange', s=100, marker='*')
        ax.set(xlim=[0, 20], ylim=[0, 20])
        ax.grid(True)
        ax.set_title(f'{projection.upper()} Projection')

    def _plot_sphere(self, ax, center, radius, color, alpha=0.4):
        """绘制3D球体"""
        u = np.linspace(0, 2 * np.pi, 8)
        v = np.linspace(0, np.pi, 8)
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)

    def show_static_visualization(self):
        """显示静态可视化"""
        self.leader_path = np.array(self.leader_path)
        self.followers_paths = [np.array(p) for p in self.followers_paths]

        fig = plt.figure(figsize=(18, 6))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])

        # 3D视图
        ax3d = fig.add_subplot(gs[0], projection='3d')
        self._plot_static_3d(ax3d)

        # XY投影
        ax_xy = fig.add_subplot(gs[1])
        self._plot_static_2d(ax_xy, 'xy')

        # XZ投影
        ax_xz = fig.add_subplot(gs[2])
        self._plot_static_2d(ax_xz, 'xz')

        plt.tight_layout()
        plt.show()

    def _init_dynamic_plot(self):
        """初始化动态可视化元素（终极修复障碍物显示）"""
        self.dynamic_ax.set_xlim(-2.5, 20)
        self.dynamic_ax.set_ylim(-2.5, 20)
        self.dynamic_ax.set_zlim(-2.5, 20)

        # 领航者路径线
        self.leader_line, = self.dynamic_ax.plot([], [], [], 'r-', lw=2, label='Leader')

        # 跟随者路径线
        self.follower_lines = [
            self.dynamic_ax.plot([], [], [], 'b--', lw=1, alpha=0.5)[0
            ] for _ in range(len(self.followers_paths))
        ]

        # 编队连线
        self.formation_lines = [
            self.dynamic_ax.plot([], [], [], 'lime', alpha=0.6, lw=0.8)[0
            ] for _ in range(6)
        ]

        # 静态障碍物绘制（精确修复）
        self.static_obstacles_artists = []
        for ob in self.static_obstacles:
            x, y, z, r = ob
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x_surf = x + r * np.outer(np.cos(u), np.sin(v))
            y_surf = y + r * np.outer(np.sin(u), np.sin(v))
            z_surf = z + r * np.outer(np.ones_like(u), np.cos(v))
            artist = self.dynamic_ax.plot_surface(
                x_surf, y_surf, z_surf,
                color='gray', alpha=0.4, edgecolor='none'
            )
            self.static_obstacles_artists.append(artist)

        # 动态障碍物初始化（终极修复）
        self.dynamic_obstacles_scatter = self.dynamic_ax.scatter(
            [], [], [],
            s=50,  # 初始尺寸占位
            c='blue',  # 初始颜色占位
            marker='o',
            alpha=0.6,
            depthshade=True,
            edgecolors='k'
        )

        # 目标点
        self.dynamic_ax.scatter(0, 0, 0, c='g', s=100, marker='*', zorder=100)
        self.dynamic_ax.scatter(15, 15, 15, c='orange', s=100, marker='*', zorder=100)
        self.dynamic_ax.legend()

    def dynamic_update(self, frame):
        """动态更新函数（终极修复障碍物显示）"""
        current_time = frame * self.dt

        # ==== 更新动态障碍物 ====
        obst_positions = []
        obst_sizes = []
        obst_colors = []

        # 生成动态障碍物数据（精确到每个时间步）
        for cfg in self.dynamic_config:
            pos = self._get_obstacle_position(cfg, current_time)
            obst_positions.append(pos)
            obst_sizes.append(200 * cfg['obstacle_radius'] ** 2)  # 尺寸放大系数
            obst_colors.append(self._get_obstacle_color(cfg['type']))

        # 转换数据结构（关键修复）
        if obst_positions:
            obst_pos_array = np.array(obst_positions)
            # 精确设置三维坐标
            self.dynamic_obstacles_scatter._offsets3d = (
                obst_pos_array[:, 0].ravel(),
                obst_pos_array[:, 1].ravel(),
                obst_pos_array[:, 2].ravel()
            )
            self.dynamic_obstacles_scatter.set_sizes(obst_sizes)
            self.dynamic_obstacles_scatter.set_color(obst_colors)

        # ==== 更新路径 ====
        # 领航者路径
        if frame > 0:
            current_leader = self.leader_path[:frame]
            self.leader_line.set_data(current_leader[:, 0], current_leader[:, 1])
            self.leader_line.set_3d_properties(current_leader[:, 2])

        # 跟随者路径
        for i, line in enumerate(self.follower_lines):
            if frame < len(self.followers_paths[i]):
                current_follower = self.followers_paths[i][:frame]
                line.set_data(current_follower[:, 0], current_follower[:, 1])
                line.set_3d_properties(current_follower[:, 2])

        # ==== 更新编队连线 ====
        if frame < len(self.formation_snapshots):
            snapshot = self.formation_snapshots[frame]
            leader = snapshot['leader']
            followers = snapshot['followers']

            if leader.shape == (3,) and all(f.shape == (3,) for f in followers):
                points = np.vstack([leader, followers])
                connections = [[0, 1], [0, 2], [0, 3], [1, 2], [2, 3], [3, 1]]

                for idx, conn in enumerate(connections):
                    if points.shape[0] > max(conn):
                        x = np.array([points[conn[0], 0], points[conn[1], 0]])
                        y = np.array([points[conn[0], 1], points[conn[1], 1]])
                        z = np.array([points[conn[0], 2], points[conn[1], 2]])
                        self.formation_lines[idx].set_data(x, y)
                        self.formation_lines[idx].set_3d_properties(z)

        # 碰撞检测显示
        self.collision_mgr.update(current_time, self.dynamic_ax)

        # 返回所有可迭代对象（关键包含障碍物）
        return [self.leader_line] + self.follower_lines + self.formation_lines + [self.dynamic_obstacles_scatter]


# ====================== 碰撞检测函数 ======================
def check_collision(position, obstacles, safety_radius=0.5):
    collision_info = []
    for idx, ob in enumerate(obstacles):
        obst_pos = np.array(ob[:3])
        obst_radius = ob[3]
        distance = np.linalg.norm(position - obst_pos)
        if distance < (obst_radius + safety_radius):
            collision_info.append({
                'obstacle_index': idx,
                'distance': distance,
                'obstacle_position': obst_pos,
                'obstacle_radius': obst_radius
            })
    return collision_info


# ====================== 主程序 ======================
if __name__ == "__main__":
    params = {
        'zeta_robot': 5, 'zeta_formation': 3.5, 'zeta_z': 1,
        'rho': 5, 'eta': 10, 'rho_rob': 2.0, 'eta_rob': 0.3,
        'speed_rep_weight': 0.5, 'dir_rep_weight': 0.5,
        'max_force': 4, 'dt': 0.1, 'save_interval': 2.0,
        'alpha': 0.8, 'w_min': 0.2, 'disturb_gain': 0.5
    }

    static_obstacles = []  # 不设置静态障碍物，设置大量动态障碍物
    dynamic_obstacles_config = []


        # ==== 生成30个满足条件的往返障碍物，避免障碍物出现在无人机诞生地和呆在终点的极端情况 ====
    import random
    count0 =0
    count1 = 0
    count2 = 0
    count3 = 0
    NUM = 100
    ob_NUM = 5
    collision_count_leader = 0
    collision_count_followers = [0, 0, 0]
    stable_success_count = 0  # 编队稳定成功次数
    total_distance_all = 0.0  # 所有跟随者总路程
    for i in tqdm(range(NUM)):
        # 每次实验生成独立的动态障碍物配置
        dynamic_obstacles_config = []
        random.seed(i)  # 保证可重复性

        flag1 = 0
        flag2= 0
        flag3 = 0
        flag4 = 0
        random.seed(i)  # 保证可重复性

        # 生成30个满足距离条件的随机点
        valid_points = []
        while len(valid_points) < ob_NUM:
            x = random.uniform(0, 15)
            y = random.uniform(0, 15)
            z = random.uniform(0, 15)

            # 计算到起点和终点的距离平方
            d_start_sq = x ** 2 + y ** 2 + z ** 2
            d_end_sq = (15 - x) ** 2 + (15 - y) ** 2 + (15 - z) ** 2

            # 确保两点距离都超过5米
            if d_start_sq >= 25 and d_end_sq >= 25:
                valid_points.append([x, y, z])

        # 添加到动态障碍物配置
        for pt in valid_points:
            dynamic_obstacles_config.append({
                'type': 'reciprocating',
                'center': pt,
                'axis': random.choice(['x', 'y']),
                'amplitude': 2.0,
                'speed': 0.3,
                'obstacle_radius': 1.0
            })

        # 初始化路径规划器
        start = (0, 0, 0)
        goal = (15, 15, 15)
        apt = IAPT3D(start=start, goal=goal,
                     k_att=1.0, k_rep=0.8, rr=3,
                     step_size=0.2, max_iters=500,
                     goal_threshold=0.5)

        # 初始化编队控制器
        formation_controller = FormationControl(num_followers=3, radius=2.0, params=params)
        followers_pos = [np.array(start) for _ in range(3)]

        # 初始化路径和碰撞检测
        leader_path = [start]
        current_leader_pos = start
        collision_mgr = CollisionManager()
        check_interval = 0.5
        last_check_time = -check_interval

        # 动态路径规划主循环
        for step in range(apt.max_iters):
            current_time = step * params['dt']

            # 计算动态障碍物位置
            dynamic_obstacles = []
            for obs in dynamic_obstacles_config:
                if obs['type'] == 'circular':
                    theta = obs['speed'] * current_time
                    x = obs['center'][0] + obs['radius'] * np.cos(theta)
                    y = obs['center'][1] + obs['radius'] * np.sin(theta)
                    z = obs['center'][2]
                    dynamic_obstacles.append([x, y, z, obs['obstacle_radius']])
                elif obs['type'] == 'linear':
                    start_pos = np.array(obs['start'])
                    end_pos = np.array(obs['end'])
                    direction = (end_pos - start_pos) / np.linalg.norm(end_pos - start_pos)
                    distance = min(obs['speed'] * current_time, np.linalg.norm(end_pos - start_pos))
                    pos = start_pos + direction * distance
                    dynamic_obstacles.append([pos[0], pos[1], pos[2], obs['obstacle_radius']])
                elif obs['type'] == 'reciprocating':
                    center = np.array(obs['center'])
                    axis = obs['axis']
                    amplitude = obs['amplitude']
                    speed = obs['speed']
                    total_distance = speed * current_time
                    mod_distance = total_distance % (2 * amplitude)
                    if mod_distance <= amplitude:
                        displacement = mod_distance
                    else:
                        displacement = 2 * amplitude - mod_distance
                    displacement -= amplitude

                    if axis == 'x':
                        x = center[0] + displacement
                        y = center[1]
                    elif axis == 'y':
                        x = center[0]
                        y = center[1] + displacement
                    z = center[2]
                    dynamic_obstacles.append([x, y, z, obs['obstacle_radius']])

            combined_obstacles = static_obstacles + dynamic_obstacles

            # 领航者路径规划
            next_leader_pos = apt.compute_next_step(current_leader_pos, combined_obstacles)
            leader_path.append(next_leader_pos)
            current_leader_pos = next_leader_pos

            # 更新编队控制
            formation_controller.update_followers(
                current_leader_pos,
                followers_pos,
                [np.array(ob[:3]) for ob in combined_obstacles]
            )

            # 碰撞检测
            if current_time - last_check_time >= check_interval:
                if check_collision(np.array(current_leader_pos), combined_obstacles):
                    print(f"时间 {current_time:.1f}s: 领航者发生碰撞！")
                    collision_mgr.add_collision(current_leader_pos, current_time)
                    flag1 =1


                for f_idx, f_pos in enumerate(followers_pos):
                    if check_collision(f_pos, combined_obstacles):
                        print(f"时间 {current_time:.1f}s: 跟随者{f_idx + 1}发生碰撞！")
                        collision_mgr.add_collision(f_pos, current_time)
                        if(f_idx+1==1): flag2=1
                        if (f_idx + 1 == 2): flag3=1
                        if (f_idx + 1 == 3): flag4=1

                last_check_time = current_time

            if np.linalg.norm(np.array(current_leader_pos) - np.array(goal)) < apt.goal_threshold:
                apt.is_path_plan_success = True
                break

        count0 +=flag1
        count1 += flag2
        count2 += flag3
        count3 += flag4
        # 统计编队稳定性
        if formation_controller.formation_stable:
            stable_success_count += 1

        # 统计跟随者总路程
        total_distance_all += np.sum(formation_controller.total_distance)

    print("领航者成功避障概率:",((NUM- count0)/NUM))
    print("跟随者1成功避障概率:", ((NUM - count1) / NUM))
    print("跟随者2成功避障概率:", ((NUM - count2) / NUM))
    print("跟随者3成功避障概率:", ((NUM - count3 )/ NUM))
    print(f"编队稳定成功率: {stable_success_count / NUM * 100:.2f}%")
    print(f"跟随者平均总路程: {total_distance_all / (NUM * 3):.2f} meters")



