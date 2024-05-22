import open3d as o3d
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.optimize import curve_fit,leastsq
import plotly.graph_objects as go
import math as m
from mpl_toolkits.mplot3d import Axes3D
import torch
import pytorch_kinematics as pk
import trimesh
import geometry_msgs.msg
from tf import transformations as tf
# 读取.ma文件
def read_ma_file(filename):
    vertices = []
    spheres = []
    with open(filename, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith('v'):
            parts = line.split()
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            spheres.append(float(parts[4]))
    
    return vertices, spheres

# 创建窗口并添加几何体
def create_window_and_add_geometry(window_name, geometries, positon):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=positon[0], height=positon[1], left=positon[2], top=positon[3])
    for geometry in geometries:
        vis.add_geometry(geometry)
    return vis

# 生成投影平面
def generate_projection_planes(z_start, z_end, num_planes):
    z_values = np.linspace(z_start, z_end, num_planes)
    planes = [{'z': z} for z in z_values]
    return planes

# 计算中轴点在投影平面上的投影点
def project_points_to_planes(vertices, spheres, planes):
    projected_points = []
    projected_spheres = []
    
    for vertex, sphere_radius in zip(vertices, spheres):
        nearest_plane_idx = np.argmin(np.abs([vertex[2] - plane['z'] for plane in planes]))#返回列表中最小值的索引
        nearest_plane_z = planes[nearest_plane_idx]['z']
        projected_point = [vertex[0], vertex[1], nearest_plane_z]
        
        projected_points.append(projected_point)
        projected_spheres.append(sphere_radius)
    
    return projected_points, projected_spheres

# 计算每个平面上的最小生成树,只有边不显示点
def compute_minimum_spanning_trees(points, planes, max_distance=None):
    msts = []
    for plane in planes:
        # 过滤出位于当前平面上的点
        points_on_plane = [point for point in points if np.isclose(point[2], plane['z'])]
        
        # 
        tree = nx.Graph()
        tree.add_nodes_from(range(len(points_on_plane)))
        num_points = len(points_on_plane)
        
        # 计算点之间的距离
        #广播后是（N，N，3）输出（N,N）
        distances = np.linalg.norm(np.array(points_on_plane)[:, np.newaxis, :] - np.array(points_on_plane)[np.newaxis, :, :], axis=-1)
        
        # 添加边到图中（如果某个点和剩余点的任何一个距离大于阈值则剪短，会生成多个簇）
        for i in range(num_points):
            for j in range(i+1, num_points):
                if max_distance is None or distances[i, j] <= max_distance:
                    tree.add_edge(i, j, weight=distances[i, j])
        
        # 计算最小生成树
        mst = nx.minimum_spanning_tree(tree)
        
        # 提取最小生成树的边
        edges = list(mst.edges())
        
        msts.append((points_on_plane, edges))
    return msts

# 将最小生成树和点投影转换为Open3D几何体并添加到窗口
# def add_MST_to_window(window, msts, projected_points):
#     for points_on_plane, edges in msts:
#         # 创建点云对象
#         point_cloud = o3d.geometry.PointCloud()
#         point_cloud.points = o3d.utility.Vector3dVector(points_on_plane)
        
#         # 创建线段对象（lines是点的索引）
#         lines = [[edge[0], edge[1]] for edge in edges]
#         line_set = o3d.geometry.LineSet()
#         line_set.points = o3d.utility.Vector3dVector(points_on_plane)
#         line_set.lines = o3d.utility.Vector2iVector(lines)
        
#         # 将点云和线段添加到窗口
#         window.add_geometry(point_cloud)
#         window.add_geometry(line_set)
    
#     # 添加投影点云到窗口
#     point_cloud = o3d.geometry.PointCloud()
#     point_cloud.points = o3d.utility.Vector3dVector(projected_points)
#     window.add_geometry(point_cloud)

def add_MST_to_window(window, msts):
    # 创建点云对象列表和线段对象列表
    all_points = []
    all_edges = []
    num_points_cumulative = 0
    
    for points_on_plane, edges in msts:
        all_points.extend(points_on_plane)
        all_edges.extend([[edge[0] + num_points_cumulative, edge[1] + num_points_cumulative] for edge in edges])
        num_points_cumulative += len(points_on_plane)

    # 创建点云对象
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(all_points)

    # 创建线段对象
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(all_points)
    line_set.lines = o3d.utility.Vector2iVector(all_edges)

    # 将点云和线段添加到窗口
    window.add_geometry(point_cloud)
    window.add_geometry(line_set)

    # 添加投影点云到窗口
    # projected_cloud = o3d.geometry.PointCloud()
    # projected_cloud.points = o3d.utility.Vector3dVector(projected_points)
    # window.add_geometry(projected_cloud)
# 定义切片类
class Slice:
    def __init__(self, points, spheres, plane_id, plane_value, max_distance):
        self.points = points        #平面点集合
        self.spheres = spheres      #平面点对应的中轴球半径
        self.plane_id = plane_id    #平面所对应的索引
        self.center_point = np.mean(self.points, axis=0)   #点集合的重心位置
        self.plane_value = plane_value   #平面高度
        self.mst = None                  #最小生成树
        self.branch_num = None           #分支点个数
        self.branch_points_id = None     #分支点索引
        self.branch_points = None        #分支点集合
        self.convex_hull = None          #凸包
        self.convex_hull_points3D = None       #将凸包中包含的所有点的高度设置为平面高度
        self.convex_hull_boundary_pointsnum = None #凸包顶点个数
        self.convex_hull_area = None
        self.shape = None                #平面的形状分类，返回一个字符串
        self.fit = None                                #平面形状拟合，返回字符串 三种情况
        self.endpoint_id = None                 #仅针对self.fit = ‘line’且存在明显直线的情况，返回直线上距离最远的两点索引
        self.edgepoint_id = None                 #针对self.fit = ‘line’但不存在明显直线的情况和self.fit = ‘circle with line’，返回直线方向上在边界处距离最远的两点索引
        self.edgepoint_id_v = None               #垂线方向
        self.max_distance = max_distance  #计算最小生成树时的裁剪阈值
        self.color = [[0, 0, 0] for _ in range(self.points.shape[0])]  # 默认所有点为黑色
        self.long = self.short = None   #代表层的长短边长度可用于衡量一个截面属于细长还是匀称
        self.compute_minimum_spanning_tree()
        self.compute_convex_hull()
        self.slice_fit()
        self.choose_shape()
        
    # 计算最小生成树
    def compute_minimum_spanning_tree(self):
        if len(self.points) >= 1:
            tree = nx.Graph()
            tree.add_nodes_from(range(len(self.points)))
            num_points = len(self.points)

            # 计算点之间的距离
            distances = np.linalg.norm(np.array(self.points)[:, np.newaxis, :] - np.array(self.points)[np.newaxis, :, :], axis=-1)

            # 添加边到图中（如果某个点和剩余点的任何一个距离大于阈值则剪断，会生成多个簇）
            for i in range(num_points):
                for j in range(i+1, num_points):
                    if self.max_distance is None or distances[i, j] <= self.max_distance:
                        tree.add_edge(i, j, weight=distances[i, j])

            # 计算最小生成树
            self.mst = nx.minimum_spanning_tree(tree)
            # 创建字典存储节点的度数
            node_degrees = {node: 0 for node in self.mst.nodes()}

            # 更新节点的度数
            for node in self.mst.nodes():
                for neighbor in self.mst.neighbors(node):
                    if distances[node, neighbor] >= 0:
                        node_degrees[node] += 1

            # 提取分支点
            # self.branch_points_id = [i for i in range(num_points) if self.mst.degree[i] >= 3]
            self.branch_points_id = [node for node, degree in node_degrees.items() if degree >= 3]
            self.branch_points = [self.points[i] for i in self.branch_points_id]
            self.branch_num = len(self.branch_points_id)
    
    # 计算凸包面积
    def area_of_convex_hull(self, vertices):
        # 将多边形的第一个顶点复制到末尾，以计算最后一个边界点到第一个点的距离
        extended_vertices = np.vstack([vertices, vertices[0]])
        area = 0
        for i in range(len(vertices)):
            area += extended_vertices[i, 0] * extended_vertices[i + 1, 1] - extended_vertices[i + 1, 0] * extended_vertices[i, 1]
        area = abs(area) / 2
        return area
    
    # 计算凸包
    def compute_convex_hull(self):
        if len(self.points) >= 3:
            self.convex_hull = ConvexHull(self.points[:, :2])
            self.convex_hull_points3D = np.hstack([self.convex_hull.points, np.full((self.convex_hull.points.shape[0], 1), self.plane_value)]) # 补充 z 坐标为平面高度 2维变三维
            self.convex_hull_boundary_pointsnum = len(self.convex_hull.vertices)
            self.convex_hull_area = self.area_of_convex_hull(self.points[self.convex_hull.vertices])
    
    #对点云进行拟合
    def slice_fit(self):
        if len(self.points) >= 3:
            # 定义直线模型
            def linear_model(x, a, b):
                return a * x + b
            # 计算点到直线的距离
            def distance_to_line(c):
                a = c[0]
                b = c[1]
                return np.abs(a * x_l - y_l + b) / np.sqrt(a**2 + 1)
            def calc_R(xc, yc):
                """ 计算s数据点与圆心(xc, yc)的距离 """
                return np.sqrt((x_r-xc)**2 + (y_r-yc)**2)
            def f_2(c):
                """ 计算半径残余"""
                Ri = calc_R(*c)
                return Ri - Ri.mean()
           
            x = self.points[:,0] ; x_l = self.points[:,0] ; x_r = self.points[:,0]
            y = self.points[:,1] ; y_l = self.points[:,1] ; y_r = self.points[:,1]
            for i in range(3):
                # 拟合直线模型
                ab_estimate = [1, 0]
                params_linear, _ = leastsq(distance_to_line, ab_estimate, maxfev=10000)                
                # 计算点到直线的距离
                distances = distance_to_line(params_linear)
                better_dataid_l =distances < 1.8*distances.mean()
                x_l = x_l[better_dataid_l]
                y_l = y_l[better_dataid_l]
                # 拟合圆模型
                center_estimate = (0, 0)# 初始参数猜测
                center_2, ier = leastsq(f_2, center_estimate, maxfev=10000)
                xc_2, yc_2 = center_2
                Ri_2       = calc_R(*center_2) #每个点到圆心的距离，返回向量
                R        = Ri_2.mean()
                better_dataid_r = np.abs(Ri_2 - R) < 1.8*np.mean(np.abs(Ri_2 - R))
                x_r = x_r[better_dataid_r]
                y_r = y_r[better_dataid_r]
                if len(x_l) < 3 or len(x_r) < 3:
                    break
            self.params_linear = params_linear
            self.params_circle = [center_2[0], center_2[1], R]
            # 判断拟合结果并记入属性
            # x_l = self.points[:,0] ; x_r = self.points[:,0]
            # y_l = self.points[:,1] ; y_r = self.points[:,1]
            # error_linear = np.sum(distance_to_line(params_linear)**2)#np.sum((y - linear_model(x, *params_linear))**2)
            # error_circle = np.sum(f_2(center_2)**2)
            error_linear = np.sum(distances**2)#np.sum((y - linear_model(x, *params_linear))**2)
            error_circle = np.sum((Ri_2 - R)**2)
            if  error_circle < error_linear/40  and R < 0.085 and np.sqrt((xc_2-self.center_point[0])**2+(yc_2-self.center_point[1])**2) < 0.2*R**0.8:
                #误差要足够小 半径要在区间内 圆心和重心距离不能超过一定要求 才能判断为圆环
                self.fit = 'circle'            
            elif error_circle < error_linear  and R < 0.085 and np.sqrt((xc_2-self.center_point[0])**2+(yc_2-self.center_point[1])**2) < 0.2*R**0.8:
                #误差没有小很多 其他条件满足 为都有
                self.fit = 'circle with line' 
                x_l = self.points[:,0] ; x_r = self.points[:,0]
                y_l = self.points[:,1] ; y_r = self.points[:,1]
                distances_for_edgepoint =distance_to_line(params_linear)
                distances_for_edgepoint = np.where(distances_for_edgepoint > 0.2, np.nan, x) #如果出现全nan请增大数值 default 0.2
                self.edgepoint_id = [np.nanargmin(distances_for_edgepoint),np.nanargmax(distances_for_edgepoint)]
                self.edgepoint_id_v = self.longest_vertical()         
            else :
                self.fit = 'line'
                #计算self.endpoint_id仅针对self.fit = ‘line’，返回直线上距离最远的两点索引
                x_l = self.points[:,0] ; x_r = self.points[:,0]
                y_l = self.points[:,1] ; y_r = self.points[:,1]
                distances_for_calculate = distance_to_line(params_linear) #得到原始数据点与线之间的距离                
                distances_for_calculate = np.where(distances_for_calculate > 0.7e-3, np.nan, x)
                if np.all(np.isnan(distances_for_calculate)) or (len(distances_for_calculate)-np.sum(np.isnan(distances_for_calculate))) <= 8 or error_linear >6e-6:
                    #如果拟合直线上的点没有或者少于8个则没有端点（不存在明显直线的情况）
                    self.endpoint_id = None
                    distances_for_edgepoint =distance_to_line(params_linear)
                    distances_for_edgepoint = np.where(distances_for_edgepoint > 5e-3, np.nan, x) #如果出现全nan请增大数值 default 5e-3
                    self.edgepoint_id = [np.nanargmin(distances_for_edgepoint),np.nanargmax(distances_for_edgepoint)]
                    self.edgepoint_id_v = self.longest_vertical()
                else:
                    self.endpoint_id = [np.nanargmin(distances_for_calculate),np.nanargmax(distances_for_calculate)]
                # print(self.endpoint_id)
    #返回垂直于拟合直线方向距离最大的两点的索引
    def longest_vertical(self):
        a_vertical = -1 / self.params_linear[0]
        point_on_line = self.center_point#(self.points[self.edgepoint_id[0]] + self.points[self.edgepoint_id[1]])/2
        b_vertical = point_on_line[1] - a_vertical * point_on_line[0]   
        distance = np.abs(a_vertical * self.points[:,0] - self.points[:,1] + b_vertical) / np.sqrt(a_vertical**2 + 1)
        distance_x = np.where(distance > 0.01, np.nan, self.points[:,0])
        distance_short = np.abs(self.params_linear[0] * self.points[:,0] - self.points[:,1] + self.params_linear[1]) / np.sqrt(self.params_linear[0]**2 + 1)
        self.short = distance_short[np.nanargmin(distance_x)] + distance_short[np.nanargmax(distance_x)] + self.spheres[np.nanargmin(distance_x)] + self.spheres[np.nanargmax(distance_x)]
        return [np.nanargmin(distance_x),np.nanargmax(distance_x)]
    #对点云进行拟合并绘制（直线或圆）
    def slice_fit_draw(self):
        if len(self.points) >= 3:
            # 定义直线模型
            def linear_model(x, a, b):
                return a * x + b
            # 计算点到直线的距离
            def distance_to_line(c):
                a = c[0]
                b = c[1]
                return np.abs(a * x_l - y_l + b) / np.sqrt(a**2 + 1)
            def calc_R(xc, yc):
                """ 计算s数据点与圆心(xc, yc)的距离 """
                return np.sqrt((x_r-xc)**2 + (y_r-yc)**2)
            def f_2(c):
                """ 计算半径残余"""
                Ri = calc_R(*c)
                return Ri - Ri.mean()
           
            x = self.points[:,0] ; x_l = self.points[:,0] ; x_r = self.points[:,0]
            y = self.points[:,1] ; y_l = self.points[:,1] ; y_r = self.points[:,1]
            for i in range(3):
                # 拟合直线模型
                ab_estimate = [1, 0]
                params_linear, _ = leastsq(distance_to_line, ab_estimate)                
                # 计算点到直线的距离
                distances = distance_to_line(params_linear)
                better_dataid_l =distances < 1.8*distances.mean()
                x_l = x_l[better_dataid_l]
                y_l = y_l[better_dataid_l]
                # 拟合圆模型
                center_estimate = (0, 0)# 初始参数猜测
                center_2, ier = leastsq(f_2, center_estimate)
                xc_2, yc_2 = center_2
                Ri_2       = calc_R(*center_2) #每个点到圆心的距离，返回向量
                R        = Ri_2.mean()
                better_dataid_r = np.abs(Ri_2 - R) < 1.8*np.mean(np.abs(Ri_2 - R))
                x_r = x_r[better_dataid_r]
                y_r = y_r[better_dataid_r]
                if len(x_l) < 3 or len(x_r) < 3:
                    break
            # 判断拟合结果
            # plt.scatter(x_l, y_l, label='Filter Data')
            print('lose points num_l',len(x)-len(x_l))
            print('lose points num_r',len(x)-len(x_r))
            # x_l = self.points[:,0] ; x_r = self.points[:,0]
            # y_l = self.points[:,1] ; y_r = self.points[:,1]
            # error_linear = np.sum(distance_to_line(params_linear)**2)#np.sum((y - linear_model(x, *params_linear))**2)
            # error_circle = np.sum(f_2(center_2)**2)
            error_linear = np.sum(distances**2)#np.sum((y - linear_model(x, *params_linear))**2)
            error_circle = np.sum((Ri_2 - R)**2)
            print(params_linear)
            # print('x',x)
            # print('y',y)
            print('center_2',center_2)
            # print('Ri_2',Ri_2)
            print('R',R)
            print('error_circle',error_circle)
            print('error_linear',error_linear)
            if  error_circle < error_linear/40  and R < 0.085 and np.sqrt((xc_2-self.center_point[0])**2+(yc_2-self.center_point[1])**2) < 0.2*R**0.8:
                #误差要足够小 半径要在区间内 圆心和重心距离不能超过0.2倍半径 才能判断为圆环
                self.fit = 'circle'
                print("The data fits better with a circle.")
                theta = np.linspace(0, 2*np.pi, 100)  # 生成角度值
                x_draw = xc_2 + R * np.cos(theta)
                y_draw = yc_2 + R * np.sin(theta)
                plt.plot(x_draw, y_draw, color='blue', label='Circle')                
            elif error_circle < error_linear  and R < 0.085 and np.sqrt((xc_2-self.center_point[0])**2+(yc_2-self.center_point[1])**2) < 0.2*R**0.8:
                #误差没有小很多 其他条件满足 为都有
                self.fit = 'circle with line' 
                print("The data fits better with circle with line.") 
                plt.plot(x, linear_model(x, *params_linear), color='red', label='Line')
                theta = np.linspace(0, 2*np.pi, 100)  # 生成角度值
                x_draw = xc_2 + R * np.cos(theta)
                y_draw = yc_2 + R * np.sin(theta)
                plt.plot(x_draw, y_draw, color='blue', label='Circle')
            else :
                self.fit = 'line'
                print("The data fits better with a straight line.")
                plt.plot(x, linear_model(x, *params_linear), color='red', label='Line')                                
            # 绘制数据和拟合结果
            plt.scatter(self.center_point[0], self.center_point[1], color='green')
            plt.scatter(x, y, label='Original Data')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.legend()
            plt.grid(True)
            plt.axis('equal')
            plt.show()
    
    # 选择切片的形状
    def choose_shape(self):
        if len(self.points) >= 3:
            # if self.branch_num == 1 and self.convex_hull_area > 2e-6 and self.convex_hull_boundary_pointsnum < 10:
            #     self.shape = 'Star'
            #     self.color = [[0, 0, 1] for _ in range(self.points.shape[0])] #Star的颜色是蓝色
            if self.convex_hull_boundary_pointsnum < 10 and self.convex_hull_area <= 2e-6:
                self.shape = 'Axis'
                self.color = [[1, 1, 0] for _ in range(self.points.shape[0])] #Axis的颜色为黄色
                self.long = self.short = 2 * np.mean(self.spheres)
            elif self.fit == 'circle':
                self.shape = 'Circle'
                self.color = [[0, 0, 1] for _ in range(self.points.shape[0])] #Circle的颜色是蓝色
                self.long = self.short = 2 * (self.params_circle[2] + np.mean(self.spheres))
            elif self.fit == 'line' and self.convex_hull_area > 5e-5:#self.branch_num == 2 and and self.convex_hull_boundary_pointsnum < 10 
                self.shape = 'Line'
                self.color = [[0, 1, 0] for _ in range(self.points.shape[0])] #Line的颜色为绿色
                if self.endpoint_id != None:
                    for j in self.endpoint_id:
                        self.color[j] = [1, 0, 1]  # 将端点endpoint设置为粉色
                    self.short = self.spheres[self.endpoint_id[0]] + self.spheres[self.endpoint_id[1]]
                    self.long = np.linalg.norm(self.points[self.endpoint_id[0]] - self.points[self.endpoint_id[1]]) + self.short
                elif self.edgepoint_id != None:
                    for j in self.edgepoint_id:
                        self.color[j] = [0, 1, 1]  # 将edgepoint设置为亮蓝色
                    for k in self.edgepoint_id_v:
                        self.color[k] = [0, 0, 0]  # 将edgepoint_v设置为黑色
                    #self.short在self.longest_vertical()中已计算
                    self.long = np.linalg.norm(self.points[self.edgepoint_id[0]] - self.points[self.edgepoint_id[1]]) + self.spheres[self.edgepoint_id[0]] + self.spheres[self.edgepoint_id[1]]
            elif self.fit == 'circle with line' and self.convex_hull_area > 5e-5:
                self.shape = 'Circle with Line'
                self.color = [[1, 0, 0] for _ in range(self.points.shape[0])] #Circle with Line的颜色为红色
                if self.edgepoint_id != None:
                    for j in self.edgepoint_id:
                        self.color[j] = [0, 1, 1]  # 将edgepoint设置为亮蓝色
                    for k in self.edgepoint_id_v:
                        self.color[k] = [0, 0, 0]  # 将edgepoint_v设置为黑色 
                    #self.short在self.longest_vertical()中已计算
                    self.long = np.linalg.norm(self.points[self.edgepoint_id[0]] - self.points[self.edgepoint_id[1]]) + self.spheres[self.edgepoint_id[0]] + self.spheres[self.edgepoint_id[1]]
                    
            # for i in self.convex_hull.vertices:
            #     if self.spheres[i] < 0.018 :
            #         self.color[i] = [0, 1, 1]  # 将中轴球小于0.018的点显示为亮蓝色
                    
            # for i in self.convex_hull.vertices:
            #     self.color[i] = [1, 0, 0]  # 将凸包顶点设置为红色
            # for j in self.branch_points_id:
            #     self.color[j] = [1, 0, 1]  # 将分支点设置为粉色
            
        else:
            self.shape = 'None'
    
    # 绘制该切片
    def draw_slice(self):
        if self.shape != 'None':
            # 创建 Open3D 点云对象
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(self.points)
            point_cloud.colors = o3d.utility.Vector3dVector(self.color)

            # 创建 Open3D 凸包对象
            lines = [[self.convex_hull.vertices[i], self.convex_hull.vertices[(i + 1) % len(self.convex_hull.vertices)]] for i in range(len(self.convex_hull.vertices))] #形成闭合
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(self.convex_hull_points3D)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector([[255/255, 178/255, 102/255] for _ in range(len(lines))])

            # 创建MST
            # 提取最小生成树的边
            edges = list(self.mst.edges())

            # 创建线段对象
            MST_set = o3d.geometry.LineSet()
            MST_set.points = o3d.utility.Vector3dVector(self.points)
            MST_set.lines = o3d.utility.Vector2iVector(edges)
            # # 创建重心点
            # center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.0005)
            # center_sphere.translate(self.center_point)

            # 创建 Open3D 可视化窗口并添加几何体
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            # vis.add_geometry(center_sphere)
            vis.add_geometry(point_cloud)
            vis.add_geometry(line_set)
            vis.add_geometry(MST_set)
            

            # 设置点云和凸包的显示样式
            opt = vis.get_render_option()
            opt.point_size = 8
            # 渲染并关闭窗口
            vis.run()
            vis.destroy_window()

    # 计算抓取，返回手掌坐标系相对物体坐标系的齐次变换矩阵,抓取系数（从中间）
    def how_to_grasp_middle(self, safe_distance, back_distance, row):
        if self.shape == 'Line':
            if self.endpoint_id != None: #逼近直线的情况
                #抓取1
                grasp_direction_1 = (self.points[self.endpoint_id[0]] - self.points[self.endpoint_id[1]])/\
                                    np.linalg.norm((self.points[self.endpoint_id[0]] - self.points[self.endpoint_id[1]]))  #0 <-- 1归一化
                grasp_point_1 = self.points[self.endpoint_id[1]] - (self.spheres[self.endpoint_id[1]] + safe_distance) * grasp_direction_1    
                #抓取2
                grasp_direction_2 = -grasp_direction_1
                # (self.points[self.endpoint_id[1]] - self.points[self.endpoint_id[0]])/\
                #                     np.linalg.norm((self.points[self.endpoint_id[1]] - self.points[self.endpoint_id[0]]))  #0 --> 1归一化
                grasp_point_2 = self.points[self.endpoint_id[0]] - (self.spheres[self.endpoint_id[0]] + safe_distance) * grasp_direction_2 
                #抓取3
                grasp_direction_3 =  np.array([-grasp_direction_1[1], grasp_direction_1[0], grasp_direction_1[2]]) 
                grasp_point_3 = (self.points[self.endpoint_id[0]] + self.points[self.endpoint_id[1]])/2 - (self.short/2+back_distance)*grasp_direction_3
                #抓取4
                grasp_direction_4 =  - grasp_direction_3
                grasp_point_4 = (self.points[self.endpoint_id[0]] + self.points[self.endpoint_id[1]])/2 - (self.short/2+back_distance)*grasp_direction_4
                H = []
                H.append(make_H_1 (grasp_direction_1, row, grasp_point_1))
                H.append(make_H_1 (grasp_direction_2, row, grasp_point_2))
                H.append(make_H_1 (grasp_direction_3, row, grasp_point_3))
                H.append(make_H_1 (grasp_direction_4, row, grasp_point_4))
            elif self.edgepoint_id != None:
                a = self.params_linear[0]
                grasp_direction_1 = -np.array([1/np.sqrt(a**2 + 1), a/np.sqrt(a**2 + 1), 0]) #0 <-- 1归一化 
                grasp_point_1 = self.center_point - (np.linalg.norm(self.center_point - self.points[self.edgepoint_id[1]]) + self.spheres[self.edgepoint_id[1]] + safe_distance) * grasp_direction_1
                # self.points[self.edgepoint_id[1]] - (self.spheres[self.edgepoint_id[1]] + safe_distance) * grasp_direction_1    #抓取点1
                grasp_direction_2 = -grasp_direction_1                                       #0 --> 1归一化
                # (self.points[self.edgepoint_id[1]] - self.points[self.edgepoint_id[0]])/\
                #                     np.linalg.norm((self.points[self.edgepoint_id[1]] - self.points[self.edgepoint_id[0]]))  
                grasp_point_2 = self.center_point - (np.linalg.norm(self.center_point - self.points[self.edgepoint_id[0]]) + self.spheres[self.edgepoint_id[0]] + safe_distance) * grasp_direction_2
                # self.points[self.edgepoint_id[0]] - (self.spheres[self.edgepoint_id[0]] + safe_distance) * grasp_direction_2    #抓取点2
                #抓取3
                grasp_direction_3 =  np.array([-grasp_direction_1[1], grasp_direction_1[0], grasp_direction_1[2]]) 
                grasp_point_3 = self.center_point - (self.short/2+back_distance)*grasp_direction_3
                #抓取4
                grasp_direction_4 =  - grasp_direction_3
                grasp_point_4 = self.center_point - (self.short/2+back_distance)*grasp_direction_4
                H = []
                H.append(make_H_1 (grasp_direction_1, row, grasp_point_1))
                H.append(make_H_1 (grasp_direction_2, row, grasp_point_2))
                H.append(make_H_1 (grasp_direction_3, row, grasp_point_3))
                H.append(make_H_1 (grasp_direction_4, row, grasp_point_4))
        elif self.shape == 'Circle with Line':
            a = self.params_linear[0]
            circle_center = np.array([self.params_circle[0], self.params_circle[1], self.center_point[2]])
            grasp_direction_1 = -np.array([1/np.sqrt(a**2 + 1), a/np.sqrt(a**2 + 1), 0]) #0 <-- 1归一化 
            grasp_point_1 = circle_center - (np.linalg.norm(circle_center - self.points[self.edgepoint_id[1]]) + self.spheres[self.edgepoint_id[1]] + safe_distance) * grasp_direction_1
            # self.points[self.edgepoint_id[1]] - (self.spheres[self.edgepoint_id[1]] + safe_distance) * grasp_direction_1    #抓取点1
            grasp_direction_2 = -grasp_direction_1                                       #0 --> 1归一化
            # (self.points[self.edgepoint_id[1]] - self.points[self.edgepoint_id[0]])/\
            #                     np.linalg.norm((self.points[self.edgepoint_id[1]] - self.points[self.edgepoint_id[0]]))  
            grasp_point_2 = circle_center - (np.linalg.norm(circle_center - self.points[self.edgepoint_id[0]]) + self.spheres[self.edgepoint_id[0]] + safe_distance) * grasp_direction_2
            # self.points[self.edgepoint_id[0]] - (self.spheres[self.edgepoint_id[0]] + safe_distance) * grasp_direction_2    #抓取点2
            #抓取3
            grasp_direction_3 =  np.array([-grasp_direction_1[1], grasp_direction_1[0], grasp_direction_1[2]]) 
            grasp_point_3 = circle_center - (self.short/2+back_distance)*grasp_direction_3
            #抓取4
            grasp_direction_4 =  - grasp_direction_3
            grasp_point_4 = circle_center - (self.short/2+back_distance)*grasp_direction_4
            H = []
            H.append(make_H_1 (grasp_direction_1, row, grasp_point_1))
            H.append(make_H_1 (grasp_direction_2, row, grasp_point_2))
            H.append(make_H_1 (grasp_direction_3, row, grasp_point_3))
            H.append(make_H_1 (grasp_direction_4, row, grasp_point_4))
        elif self.shape == 'Circle':
            if self.long < 0.12 :  #如果圆太大了会抓不住
                a = self.params_linear[0]
                R = self.params_circle[2]
                circle_center = np.array([self.params_circle[0], self.params_circle[1], self.center_point[2]])
                grasp_direction_1 = -np.array([1/np.sqrt(a**2 + 1), a/np.sqrt(a**2 + 1), 0]) #0 <-- 1归一化 
                grasp_point_1 = circle_center - (R + np.mean(self.spheres) + safe_distance) * grasp_direction_1
                # self.points[self.edgepoint_id[1]] - (self.spheres[self.edgepoint_id[1]] + safe_distance) * grasp_direction_1    #抓取点1
                grasp_direction_2 = -grasp_direction_1                                       #0 --> 1归一化
                # (self.points[self.edgepoint_id[1]] - self.points[self.edgepoint_id[0]])/\
                #                     np.linalg.norm((self.points[self.edgepoint_id[1]] - self.points[self.edgepoint_id[0]]))  
                grasp_point_2 = circle_center - (R + np.mean(self.spheres) + safe_distance) * grasp_direction_2
                # self.points[self.edgepoint_id[0]] - (self.spheres[self.edgepoint_id[0]] + safe_distance) * grasp_direction_2    #抓取点2
                #抓取3
                grasp_direction_3 =  np.array([-grasp_direction_1[1], grasp_direction_1[0], grasp_direction_1[2]]) 
                grasp_point_3 = circle_center - (R + np.mean(self.spheres) + safe_distance) * grasp_direction_3
                #抓取4
                grasp_direction_4 =  - grasp_direction_3
                grasp_point_4 = circle_center - (R + np.mean(self.spheres) + safe_distance) * grasp_direction_4
                H = []
                H.append(make_H_1 (grasp_direction_1, row, grasp_point_1))
                H.append(make_H_1 (grasp_direction_2, row, grasp_point_2))
                H.append(make_H_1 (grasp_direction_3, row, grasp_point_3))
                H.append(make_H_1 (grasp_direction_4, row, grasp_point_4))
            # print(H)
        elif self.shape == 'Axis':
            if self.long < 0.12 :  #如果球太大了会抓不住
                a = self.params_linear[0]
                #抓取1
                grasp_direction_1 = -np.array([1/np.sqrt(a**2 + 1), a/np.sqrt(a**2 + 1), 0]) #0 <-- 1归一化
                grasp_point_1 = self.center_point - (np.mean(self.spheres) + safe_distance) * grasp_direction_1    
                #抓取2
                grasp_direction_2 = -grasp_direction_1
                grasp_point_2 = self.center_point - (np.mean(self.spheres) + safe_distance) * grasp_direction_2 
                #抓取3
                grasp_direction_3 =  np.array([-grasp_direction_1[1], grasp_direction_1[0], grasp_direction_1[2]]) 
                grasp_point_3 = self.center_point - (np.mean(self.spheres) + safe_distance) * grasp_direction_3
                #抓取4
                grasp_direction_4 =  - grasp_direction_3
                grasp_point_4 = self.center_point - (np.mean(self.spheres) + safe_distance) * grasp_direction_4
                H = []
                H.append(make_H_1 (grasp_direction_1, row, grasp_point_1))
                H.append(make_H_1 (grasp_direction_2, row, grasp_point_2))
                H.append(make_H_1 (grasp_direction_3, row, grasp_point_3))
                H.append(make_H_1 (grasp_direction_4, row, grasp_point_4))
        return H
    def how_to_grasp_up_and_down(self, safe_distance, str, row):
        if str == 'up':
            up_or_down = -1
        elif str == 'down':
            up_or_down = 1
        if self.shape == 'Line':
            if self.endpoint_id != None: #逼近直线的情况
                a = self.params_linear[0]
                #抓取1
                grasp_direction_1 = np.array([0, 0, 1]) * up_or_down  #0 <-- 1归一化
                grasp_point_1 = (self.points[self.endpoint_id[0]] + self.points[self.endpoint_id[1]])/2 - safe_distance * grasp_direction_1   
                #抓取2
                grasp_direction_2 = grasp_direction_1
                grasp_point_2 = 2*self.points[self.endpoint_id[0]]/3 + self.points[self.endpoint_id[1]]/3 - safe_distance * grasp_direction_2 
                #抓取3
                grasp_direction_3 =  grasp_direction_1 
                grasp_point_3 = self.points[self.endpoint_id[0]]/3 + 2*self.points[self.endpoint_id[1]]/3 - safe_distance * grasp_direction_3
                H = []
                H.append(make_H_2 (a, up_or_down, row, grasp_point_1))
                H.append(make_H_2 (a, up_or_down, row, grasp_point_2))
                H.append(make_H_2 (a, up_or_down, row, grasp_point_3))
            elif self.edgepoint_id != None:
                a = self.params_linear[0]
                #抓取1
                grasp_direction_1 = np.array([0, 0, 1]) * up_or_down  #0 <-- 1归一化
                grasp_point_1 = self.center_point - safe_distance * grasp_direction_1   
                #抓取2
                vector = np.array([1/np.sqrt(a**2 + 1), a/np.sqrt(a**2 + 1), 0])
                grasp_direction_2 = grasp_direction_1
                grasp_point_2 = self.center_point - np.linalg.norm(self.points[self.edgepoint_id[0]] - self.points[self.edgepoint_id[1]]) / 6 * vector- safe_distance * grasp_direction_2 
                #抓取3
                grasp_direction_3 =  grasp_direction_1 
                grasp_point_3 = self.center_point + np.linalg.norm(self.points[self.edgepoint_id[0]] - self.points[self.edgepoint_id[1]]) / 6 * vector- safe_distance * grasp_direction_2
                H = []
                H.append(make_H_2 (a, up_or_down, row, grasp_point_1))
                H.append(make_H_2 (a, up_or_down, row, grasp_point_2))
                H.append(make_H_2 (a, up_or_down, row, grasp_point_3))
        elif self.shape == 'Circle with Line':
            a = self.params_linear[0]
            circle_center = np.array([self.params_circle[0], self.params_circle[1], self.center_point[2]])
            #抓取1
            grasp_direction_1 = np.array([0, 0, 1]) * up_or_down  #0 <-- 1归一化
            grasp_point_1 = circle_center - safe_distance * grasp_direction_1   
            #抓取2
            vector = np.array([1/np.sqrt(a**2 + 1), a/np.sqrt(a**2 + 1), 0])
            grasp_direction_2 = grasp_direction_1
            grasp_point_2 = circle_center - np.linalg.norm(self.points[self.edgepoint_id[0]] - self.points[self.edgepoint_id[1]]) / 6 * vector- safe_distance * grasp_direction_2 
            #抓取3
            grasp_direction_3 =  grasp_direction_1 
            grasp_point_3 = circle_center + np.linalg.norm(self.points[self.edgepoint_id[0]] - self.points[self.edgepoint_id[1]]) / 6 * vector- safe_distance * grasp_direction_2
            H = []
            H.append(make_H_2 (a, up_or_down, row, grasp_point_1))
            H.append(make_H_2 (a, up_or_down, row, grasp_point_2))
            H.append(make_H_2 (a, up_or_down, row, grasp_point_3))
        elif self.shape == 'Circle':
            a = self.params_linear[0]
            R = self.params_circle[2]
            circle_center = np.array([self.params_circle[0], self.params_circle[1], self.center_point[2]])
            #抓取1
            grasp_direction_1 = np.array([0, 0, 1]) * up_or_down  #0 <-- 1归一化
            grasp_point_1 = circle_center - safe_distance * grasp_direction_1 
            H = []
            H.append(make_H_2 (a, up_or_down, row, grasp_point_1))
            H.append(make_H_2 (a, up_or_down, row + 2*np.pi/3, grasp_point_1))
            H.append(make_H_2 (a, up_or_down, row + 4*np.pi/3, grasp_point_1))
        elif self.shape == 'Axis':
            a = self.params_linear[0]
            R = self.params_circle[2]
            #抓取1
            grasp_direction_1 = np.array([0, 0, 1]) * up_or_down  #0 <-- 1归一化
            grasp_point_1 = self.center_point - safe_distance * grasp_direction_1 
            H = []
            H.append(make_H_2 (a, up_or_down, row, grasp_point_1))
            H.append(make_H_2 (a, up_or_down, row + 2*np.pi/3, grasp_point_1))
            H.append(make_H_2 (a, up_or_down, row + 4*np.pi/3, grasp_point_1))
        return H
# 对所有切片的总体绘制
def collect_geometries(all_slices):
    all_points = []
    colors = []
    all_spheres = []
    convex_hulls = []
    msts = []

    for slice_name, slice_obj in all_slices.items():
        if slice_obj.shape != 'None':# and slice_obj.shape != 'have not been choosen':
            # 创建点云对象
            all_points.extend(slice_obj.points)
            colors.extend(slice_obj.color)

            # 创建中轴球对象
            for i in range(len(slice_obj.points)):
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=slice_obj.spheres[i])
                sphere.translate(slice_obj.points[i])
                all_spheres.append(sphere)
            # 创建凸包对象
            lines = [[slice_obj.convex_hull.vertices[i], slice_obj.convex_hull.vertices[(i + 1) % len(slice_obj.convex_hull.vertices)]] for i in range(len(slice_obj.convex_hull.vertices))]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(slice_obj.convex_hull_points3D)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector([[255/255, 178/255, 102/255] for _ in range(len(lines))])
            convex_hulls.append(line_set)

            # 创建最小生成树对象
            edges = list(slice_obj.mst.edges())
            mst_set = o3d.geometry.LineSet()
            mst_set.points = o3d.utility.Vector3dVector(slice_obj.points)
            mst_set.lines = o3d.utility.Vector2iVector(edges)
            msts.append(mst_set)
    point_clouds = o3d.geometry.PointCloud()
    point_clouds.points = o3d.utility.Vector3dVector(all_points)
    point_clouds.colors = o3d.utility.Vector3dVector(colors)
    return point_clouds, convex_hulls, msts, all_spheres     
#定义抓取类 direction = 'middle' 'up' 'down'
class Grasp:
    def __init__(self, slice_obj, direction, grasp_kind, slicename, H_list):
        self.slice = slice_obj
        self.direction = direction                     #direction = 'middle' 'middle_vertical' 'up' 'down' 'up_v', 'down_v'
        self.slicename = slicename
        self.grasp_kind = grasp_kind                  #grasp_kind = 'straight' 'straight_power' 'flexed' 
        self.H_list = H_list
        self.H_list_hand_ready = None               #机械臂基座坐标系下，手掌抓取坐标系在准备时的位姿，用于安全性判断
        self.H_list_hand = None                     #机械臂基座坐标系下，手掌抓取坐标系的位姿，用于安全性判断
        self.H_list_final_ready = None              #机械臂基座坐标系下，末端坐标系在准备时的位姿，是直接发给机械臂的指令
        self.H_list_final = None                    #机械臂基座坐标系下，末端坐标系的位姿，是直接发给机械臂的指令
        self.K = None                               #调节系数 调节抓取的松紧，默认为0，还没想好怎么弄
        self.qa = None
        self.prepare()
    def prepare(self):
        if self.direction in ['middle', 'up', 'down']:             
            if self.grasp_kind in ['straight', 'straight_power']:   #直掌用.short
                self.k = k = (self.slice.short - 0.05470885)*4.727272727
                fi11 = -0.06; fi12 = 1.30; fi13 = 0.35;        
                fi21 = -0.03; fi22 = 1.35; fi23 = 0.35; 
                fi31 = 0; fi32 = 0.75;
                fi41 = 0; fi42 = 0.64;
                fi51 = 1.33; fi52 = 0.25; fi53 = 0.30;
                a,b = trans2_2_3_4(fi32-k*0.7)
                c,d = trans3_2_3_4(fi42-k*0.7)
                qa = np.array([fi11,fi12-k,fi13,trans1_3_4(fi13),
                                fi21,fi22-k*0.7,fi23,trans1_3_4(fi23),
                                fi31,fi32-k*0.7,a,b,
                                fi41,fi42-k*0.7,c,d,
                                fi51,fi52-k,fi53,trans4_3_4(fi53)])
            if self.grasp_kind in ['flexed']:
                fi11 = -0.06; fi12 = 0.9; fi13 = 1.1;         
                fi21 = -0.03; fi22 = 1.1; fi23 = 1.1; 
                fi31 = 0; fi32 = 0.85;
                fi41 = 0; fi42 = 0.62;
                fi51 = 1.33; fi52 = 0.0; fi53 = 0.6;
                self.k = k = (self.slice.short - 0.09551625244133868)*12.990663617           # flexed用
                a,b = trans2_2_3_4(fi32-k*0.5)
                c,d = trans3_2_3_4(fi42-k*0.5)
                qa = np.array([fi11,fi12-k,fi13,trans1_3_4(fi13),
                                fi21,fi22-k*0.7,fi23,trans1_3_4(fi23),
                                fi31,fi32-k*0.5,a,b,
                                fi41,fi42-k*0.5,c,d,
                                fi51,fi52-k*0.4,fi53,trans4_3_4(fi53)])
        if self.direction in ['middle_vertical', 'up_v', 'down_v']:     
            if self.grasp_kind in ['straight', 'straight_power']:   #直掌用.long
                self.k = k = (self.slice.long - 0.05470885)*4.727272727
                fi11 = -0.06; fi12 = 1.30; fi13 = 0.35;        
                fi21 = -0.03; fi22 = 1.35; fi23 = 0.35; 
                fi31 = 0; fi32 = 0.75;
                fi41 = 0; fi42 = 0.64;
                fi51 = 1.33; fi52 = 0.25; fi53 = 0.30;
                a,b = trans2_2_3_4(fi32-k*0.7)
                c,d = trans3_2_3_4(fi42-k*0.7)
                qa = np.array([fi11,fi12-k,fi13,trans1_3_4(fi13),
                                fi21,fi22-k*0.7,fi23,trans1_3_4(fi23),
                                fi31,fi32-k*0.7,a,b,
                                fi41,fi42-k*0.7,c,d,
                                fi51,fi52-k,fi53,trans4_3_4(fi53)])
            if self.grasp_kind in ['flexed']:
                fi11 = -0.06; fi12 = 0.9; fi13 = 1.1;         
                fi21 = -0.03; fi22 = 1.1; fi23 = 1.1; 
                fi31 = 0; fi32 = 0.85;
                fi41 = 0; fi42 = 0.62;
                fi51 = 1.33; fi52 = 0.0; fi53 = 0.6;
                self.k = k = (self.slice.long - 0.09551625244133868)*12.990663617           # flexed用
                a,b = trans2_2_3_4(fi32-k*0.5)
                c,d = trans3_2_3_4(fi42-k*0.5)
                qa = np.array([fi11,fi12-k,fi13,trans1_3_4(fi13),
                                fi21,fi22-k*0.7,fi23,trans1_3_4(fi23),
                                fi31,fi32-k*0.5,a,b,
                                fi41,fi42-k*0.5,c,d,
                                fi51,fi52-k*0.4,fi53,trans4_3_4(fi53)])
        ranges = { 'fi11': (-0.04, 0.04), 'fi12': (0, 2.09), 'fi13': (0, 1.74),'fi14': (-10, 10),
                    'fi21': (-0.04, 0.04), 'fi22': (0, 2.09), 'fi23': (0, 1.74),'fi24': (-10, 10),
                    'fi31': (-0.04, 0.04), 'fi32': (0, 1.74), 'fi33': (-10, 10),'fi34': (-10, 10),
                    'fi41': (-0.04, 0.04), 'fi42': (0, 1.60), 'fi43': (-10, 10),'fi44': (-10, 10),
                    'fi51': ( 0.00, 1.50), 'fi52': (0, 1.33), 'fi53': (0, 1.74),'fi54': (-10, 10),
           }
        for key, (lower, upper) in ranges.items():
            # 获取元素的索引
            index = list(ranges.keys()).index(key)
            # 对元素进行范围限制
            qa[index] = np.clip(qa[index], lower, upper)
        e,f = trans2_2_3_4(qa[9])
        g,h = trans3_2_3_4(qa[13])
        self.qa = np.array([qa[0],qa[1],qa[2],trans1_3_4(qa[2]),
                            qa[4],qa[5],qa[6],trans1_3_4(qa[6]),
                            qa[8],qa[9],e,f,
                            qa[12],qa[13],g,h,
                            qa[16],qa[17],qa[18],trans4_3_4(qa[18])])
        # print(self.qa - qa)
    def show_grasp(self):
        for H in self.H_list :
            theta = np.pi * 35 / 180
            H_base_hand = np.array([[np.cos(theta), 0, np.sin(theta), -2e-3*np.sin(theta)],              #  手base-->手
                                    [np.sin(theta), 0, -np.cos(theta), 2e-3*np.cos(theta)],
                                    [0,             1,        0,            -14e-3         ],
                                    [0,             0,        0,            1         ]
                                    ])
            H_translation = np.array([[1, 0, 0, -0.14],              # 平移一个距离
                                      [0, 1, 0, 0],
                                      [0, 0, 1, 0],
                                      [0, 0, 0, 1]
                                    ])
            H_show = H
            H_base_obj = np.dot(H_show, H_base_hand)
            H_ready = np.dot(np.dot(H_show,H_translation), H_base_hand)
            vertices_obj, faces_obj = load_obj('/home/hh/ros1_ws/src/Q-mat for grasp python/learn plotly/textured.obj')
            # 从关节角 计算手的vertices_hand
            hand_file = '/home/hh/ros1_ws/src/Q-mat for grasp python/hand_meshes/hand_right.xacro.urdf'
            hand_vision_tool = HandModelVision(hand_file,static_image_mode=False)
            hand_pose = torch.tensor(self.qa, dtype=torch.float, device="cpu").unsqueeze(0)
            hand_vision_tool.set_parameters(hand_pose)
            hand_mesh = hand_vision_tool.get_trimesh_data(0)
            vertices_hand= hand_mesh.vertices
            faces_hand = hand_mesh.faces
            vertices_hand = homogeneous_transform(vertices_hand, H_base_obj)

            #绘制初始位置
            e,f = trans2_2_3_4(0.15)
            g,h = trans3_2_3_4(0.15)
            qa_ready = np.array([-0.06,0.45,0.18,trans1_3_4(0.18),
                        -0.03,0.45,0.18,trans1_3_4(0.18),
                        0,0.15,e,f,
                        0,0.15,g,h,
                        1.33,0,0,trans4_3_4(0)])
            hand_pose_ready = torch.tensor(qa_ready, dtype=torch.float, device="cpu").unsqueeze(0)
            hand_vision_tool.set_parameters(hand_pose_ready)
            vertices_ready = homogeneous_transform(hand_vision_tool.get_trimesh_data(0).vertices, H_ready)
            #绘制物体和手
            plot_objs_plotly([vertices_ready,vertices_hand,vertices_obj], [ faces_hand, faces_hand, faces_obj])
    def show_which_grasp(self, which_grasp):
        H = self.H_list[which_grasp] 
        theta = np.pi * 35 / 180
        H_base_hand = np.array([[np.cos(theta), 0, np.sin(theta), -2e-3*np.sin(theta)],              #  手base-->手
                                [np.sin(theta), 0, -np.cos(theta), 2e-3*np.cos(theta)],
                                [0,             1,        0,            -14e-3         ],
                                [0,             0,        0,            1         ]
                                ])
        H_translation = np.array([[1, 0, 0, -0.14],              # 平移一个距离
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]
                                ])
        H_show = H
        H_base_obj = np.dot(H_show, H_base_hand)
        H_ready = np.dot(np.dot(H_show,H_translation), H_base_hand)
        vertices_obj, faces_obj = load_obj('/home/hh/ros1_ws/src/Q-mat for grasp python/learn plotly/textured.obj')
        # 从关节角 计算手的vertices_hand
        hand_file = '/home/hh/ros1_ws/src/Q-mat for grasp python/hand_meshes/hand_right.xacro.urdf'
        hand_vision_tool = HandModelVision(hand_file,static_image_mode=False)
        hand_pose = torch.tensor(self.qa, dtype=torch.float, device="cpu").unsqueeze(0)
        hand_vision_tool.set_parameters(hand_pose)
        hand_mesh = hand_vision_tool.get_trimesh_data(0)
        vertices_hand= hand_mesh.vertices
        faces_hand = hand_mesh.faces
        vertices_hand = homogeneous_transform(vertices_hand, H_base_obj)

        #绘制初始位置
        e,f = trans2_2_3_4(0.15)
        g,h = trans3_2_3_4(0.15)
        qa_ready = np.array([-0.06,0.45,0.18,trans1_3_4(0.18),
                    -0.03,0.45,0.18,trans1_3_4(0.18),
                    0,0.15,e,f,
                    0,0.15,g,h,
                    1.33,0,0,trans4_3_4(0)])
        hand_pose_ready = torch.tensor(qa_ready, dtype=torch.float, device="cpu").unsqueeze(0)
        hand_vision_tool.set_parameters(hand_pose_ready)
        vertices_ready = homogeneous_transform(hand_vision_tool.get_trimesh_data(0).vertices, H_ready)
        #绘制物体和手
        plot_objs_plotly([vertices_ready,vertices_hand,vertices_obj], [ faces_hand, faces_hand, faces_obj])
    def get_final_H(self, H_cam_base, H_obj_cam, H_tool_hand): 
        self.H_list_hand = []
        self.H_list_hand_ready = []
        self.H_list_final_ready = []
        self.H_list_final = []
        H_translation = np.array([[1, 0, 0, -0.14],              # 平移一个距离
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]
                        ])
        for H_hand_obj in self.H_list :
            self.H_list_hand.append(H_cam_base @ H_obj_cam @ H_hand_obj)
            self.H_list_hand_ready.append(H_cam_base @ H_obj_cam @ (H_hand_obj @ H_translation))
            self.H_list_final_ready.append(H_cam_base @ H_obj_cam @ (H_hand_obj @ H_translation) @ H_tool_hand)
            self.H_list_final.append(H_cam_base @ H_obj_cam @ H_hand_obj @ H_tool_hand)
#绘制直方图
def plot_histogram(data, bins=10, color='blue', edgecolor='black', title='Histogram', xlabel='Value', ylabel='Frequency'):
    """
    绘制直方图。

    参数：
    - data: 包含数据的列表或数组。
    - bins: 箱子的数量,默认为10。
    - color: 直方图的颜色，默认为蓝色。
    - edgecolor: 箱子边缘的颜色，默认为黑色。
    - title: 图表标题，默认为'Histogram'。
    - xlabel: x 轴标签，默认为'Value'。
    - ylabel: y 轴标签，默认为'Frequency'。
    """
    plt.hist(data, bins=bins, color=color, edgecolor=edgecolor)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()   

# 计算手掌坐标系和物体坐标系之间的齐次变换矩阵
def make_H_1(x_axis, row, distance):
    x = x_axis[0]; y = x_axis[1]; 
    H = np.array([
        [x, -y * np.cos(row), y * np.sin(row), distance[0]],
        [y, x * np.cos(row), -x * np.sin(row), distance[1]],
        [0, np.sin(row),     np.cos(row),      distance[2]],
        [0, 0,               0,                          1]
        ])
    return H

def make_H_2(a,up_or_down, row, distance):   #ax+b的a
    x = -1/np.sqrt(a**2 + 1) * up_or_down
    y = -a/np.sqrt(a**2 + 1) * up_or_down
    H_1 = np.array([
        [0, y,  x* up_or_down,      distance[0]],
        [0, -x, y* up_or_down,      distance[1]],
        [up_or_down, 0,  0,         distance[2]],
        [0, 0,  0,           1]
        ])
    H_2 = np.array([
        [1, 0, 0, 0],
        [0, np.cos(row), -np.sin(row), 0],
        [0, np.sin(row),  np.cos(row), 0],
        [0, 0,               0,        1]
        ])
    H = np.dot(H_1, H_2)
    return H
#齐次变换矩阵的逆
def homogeneous_inverse(T):
    """
    计算齐次变换矩阵的逆
    
    参数：
        T: 4x4 的齐次变换矩阵
        
    返回值：
        T_inv: 4x4 的逆齐次变换矩阵
    """
    # 提取旋转矩阵和平移向量
    R = T[:3, :3]  # 旋转矩阵
    t = T[:3, 3]   # 平移向量
    
    # 计算旋转矩阵的转置
    R_transpose = R.T
    
    # 计算平移部分的逆
    t_inv = -np.dot(R_transpose, t)
    
    # 构建逆齐次变换矩阵
    T_inv = np.eye(4)
    T_inv[:3, :3] = R_transpose
    T_inv[:3, 3] = t_inv
    
    return T_inv
########################################################################新加
#基元旋转矩阵
def rot(cta_rad, x):
    if x == 1:
        A = np.array([[1, 0, 0],
                      [0, np.cos(cta_rad), -np.sin(cta_rad)],
                      [0, np.sin(cta_rad), np.cos(cta_rad)]])
    elif x == 2:
        A = np.array([[np.cos(cta_rad), 0, np.sin(cta_rad)],
                      [0, 1, 0],
                      [-np.sin(cta_rad), 0, np.cos(cta_rad)]])
    elif x == 3:
        A = np.array([[np.cos(cta_rad), -np.sin(cta_rad), 0],
                      [np.sin(cta_rad), np.cos(cta_rad), 0],
                      [0, 0, 1]])
    else:
        raise ValueError("x must be 1, 2, or 3")
    return A
#给出rpy和d(顺序xyz)返回H
def rpy_d_to_H(rpy, d):
    rot_matrix = rot(rpy[2], 3)@rot(rpy[1], 2)@rot(rpy[0], 1)
    H = np.eye(4)
    H[:3, :3] = rot_matrix
    H[0,3] = d[0]
    H[1,3] = d[1]
    H[2,3] = d[2]
    return H
#输入齐次变换矩阵输出用于moveit的消息格式
def homogeneous_matrix_to_pose(matrix):
    # Extract translation vector from homogeneous matrix
    translation = tf.translation_from_matrix(matrix)
    # Extract quaternion from rotation part of homogeneous matrix
    quaternion = tf.quaternion_from_matrix(matrix)
    
    pose_msg = geometry_msgs.msg.Pose()
    pose_msg.position = geometry_msgs.msg.Point(*translation)
    pose_msg.orientation = geometry_msgs.msg.Quaternion(*quaternion)
    
    return pose_msg
#####################################################################################plotly 绘图用
#region
def load_obj(filename):
    mesh = o3d.io.read_triangle_mesh(filename)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    return vertices, faces


def plot_objs_plotly(vertices_list, faces_list):
    fig = go.Figure()

    colors = ['yellow', 'pink', 'lightgray' ]  # 设置颜色列表

    for idx, (vertices, faces) in enumerate(zip(vertices_list, faces_list)):
        x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]

        # 计算模型的中心点
        center_x = np.mean(x)
        center_y = np.mean(y)
        center_z = np.mean(z)

        # 计算每个轴的范围
        max_range = max(np.max(x) - np.min(x), np.max(y) - np.min(y), np.max(z) - np.min(z))
        x_range = [center_x - max_range / 2, center_x + max_range / 2]
        y_range = [center_y - max_range / 2, center_y + max_range / 2]
        z_range = [center_z - max_range / 2, center_z + max_range / 2]

        # 扩大每个轴的范围，确保整个模型都能够显示在图中
        x_padding = max_range * 1
        y_padding = max_range * 1
        z_padding = max_range * 1
        x_range = [x_range[0] - x_padding, x_range[1] + x_padding]
        y_range = [y_range[0] - y_padding, y_range[1] + y_padding]
        z_range = [z_range[0] - z_padding, z_range[1] + z_padding]

        # 为每个三角形定义顶点索引
        i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]

        fig.add_trace(go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=i,
            j=j,
            k=k,
            color=colors[idx],  # 设置颜色
            opacity=1.0  # 设置透明度为不透明
        ))

    fig.update_layout(scene=dict(
        xaxis=dict(nticks=4, range=x_range),
        yaxis=dict(nticks=4, range=y_range),
        zaxis=dict(nticks=4, range=z_range),
        aspectmode='cube'
    ))

    fig.show()

#返回经过齐次变换矩阵变换后的向量
def homogeneous_transform(vectors, transformation_matrix):
    # 将每个向量转换为齐次坐标
    vectors_homogeneous = np.hstack((vectors, np.ones((vectors.shape[0], 1))))

    # 进行齐次变换
    transformed_vectors_homogeneous = np.dot(transformation_matrix, vectors_homogeneous.T).T

    # 去除齐次坐标得到变换后的向量
    transformed_vectors = transformed_vectors_homogeneous[:, :3]

    return transformed_vectors
def trans1_3_4(degree_3):#模组1 关节3转4
    A=9.5;
    B=32.35;
    C=29.867;
    D=7.3;
    initial_degree1=m.radians(61.4845);
    initial_degree2=m.radians(118.1266);
    L1=m.sqrt(A**2+C**2-2*A*C*m.cos(degree_3+initial_degree1));
    parameter1=(C-A*m.cos(degree_3+initial_degree1))/L1;
    parameter2=(L1**2+D**2-B**2)/(2*L1*D);
    degree_4=initial_degree2-(m.acos(parameter2)-m.acos(parameter1));
    return degree_4

def trans2_2_3_4(theta):#模组2 关节2转3转4
    A=7;
    B=46;
    C=45;
    D=6;
    initial_degree1=m.radians(66.3557);
    initial_degree2=m.radians(111.6954);
    L1=m.sqrt(A**2+C**2-2*A*C*m.cos(theta+initial_degree1));
    parameter1=(C-A*m.cos(theta+initial_degree1))/L1;
    parameter2=(L1**2+D**2-B**2)/(2*L1*D);
    degree_3=initial_degree2-(m.acos(parameter2)-m.acos(parameter1));
    
    A1=4.5;
    B1=26.5;
    C1=25;
    D1=6;
    initial_degree3=m.radians(45.0619);
    initial_degree4=m.radians(124.5317);
    L1_=m.sqrt(A1**2+C1**2-2*A1*C1*m.cos(degree_3+initial_degree3));
    parameter1=(C1-A1*m.cos(degree_3+initial_degree3))/L1_;
    parameter2=(L1_**2+D1**2-B1**2)/(2*L1_*D1);
    degree_4=initial_degree4-(m.acos(parameter2)-m.acos(parameter1));
    return degree_3,degree_4

def trans3_2_3_4(theta):#模组3 关节2转3转4
    A=7;
    B=36.85;
    C=36;
    D=6;
    initial_degree1=m.radians(67.63534);
    initial_degree2=m.radians(103.46481);
    L1=m.sqrt(A**2+C**2-2*A*C*m.cos(theta+initial_degree1));
    parameter1=(C-A*m.cos(theta+initial_degree1))/L1;
    parameter2=(L1**2+D**2-B**2)/(2*L1*D);
    degree_3=initial_degree2-(m.acos(parameter2)-m.acos(parameter1));
    
    A1=4.5;
    B1=26.5;
    C1=25;
    D1=6;
    initial_degree3=m.radians(45.0619);
    initial_degree4=m.radians(124.5317);
    L1_=m.sqrt(A1**2+C1**2-2*A1*C1*m.cos(degree_3+initial_degree3));
    parameter1=(C1-A1*m.cos(degree_3+initial_degree3))/L1_;
    parameter2=(L1_**2+D1**2-B1**2)/(2*L1_*D1);
    degree_4=initial_degree4-(m.acos(parameter2)-m.acos(parameter1));
    return degree_3,degree_4

def trans4_3_4(theta):#模组4 关节3转4
    A=7;
    B=37.610;
    C=36;
    D=5.5;
    initial_degree1=m.radians(70.96005765);#71.98912
    initial_degree2=m.radians(111.50535364);#109.82193
    L1=m.sqrt(A**2+C**2-2*A*C*m.cos(theta+initial_degree1));
    parameter1=(C-A*m.cos(theta+initial_degree1))/L1;
    parameter2=(L1**2+D**2-B**2)/(2*L1*D);
    degree_4=initial_degree2-(m.acos(parameter2)-m.acos(parameter1));
    return degree_4      
    
class HandModelVision:
    def __init__(self,
                 hand_file,
                 static_image_mode=True,
                 mesh_path=None, 
                 device='cpu'):
        # self.hand_file = hand_file
        # self.is_static = static_image_mode
        
        # self.hand_mp_sol = mp.solutions.hands
        # self.hand_mp = self.hand_mp_sol.Hands(
        #     static_image_mode=static_image_mode,
        #     max_num_hands=2,
        #     model_complexity=1, # 更准确
        #     min_detection_confidence=0.8,
        #     min_tracking_confidence=0.7);     
        #     # 创建了 Hands 类的实例，用于处理手部关键点检测。 
        #     # 'hands'对象将用于处理每一帧图像，检测手的关键点
        # self.draw_mp = mp.solutions.drawing_utils
        # # self.finger_index = [[0, 1, 2, 3, 4],
        # #        [0, 5, 6, 7, 8],
        # #        [0, 9, 10, 11, 12],
        # #        [0, 13, 14, 15, 16],
        # #        [0, 17, 18, 19, 20]]
        # # self.finger_end_index = [4, 8, 12, 16, 20]
        
        # self.finger_index = [
        #        [0, 5, 6, 7, 8],
        #        [0, 9, 10, 11, 12],
        #        [0, 13, 14, 15, 16],
        #        [0, 17, 18, 19, 20],
        #        [0, 1, 2, 3, 4],]
        # self.finger_end_index = [8, 12, 16, 20, 4, ]
        
        # self.fig = plt.figure(figsize=(10, 5))
        # self.ax1 = self.fig.add_subplot(121, projection='3d')
        # self.ax2 = self.fig.add_subplot(122, projection='3d')
        ##############################
        self.device = device
        # not build_serial_chain_from_urdf
        self.chain = pk.build_chain_from_urdf(
            open(hand_file).read()).to(dtype=torch.float, device=device)
        self.n_dofs = len(self.chain.get_joint_parameter_names())
        # print(self.n_dofs)
        # print(self.chain.get_joint_parameter_names())

        self.mesh = {}
        
        def build_mesh_recurse(body):
            if (len(body.link.visuals) > 0):
                # print(body.link)
                # print(body.link.visuals)
                link_vertices = []
                link_faces = []
                n_link_vertices = 0
                for visual in body.link.visuals:
                    if visual.geom_type is None:
                        continue
                    else:
                        scale = torch.tensor(
                            [1, 1, 1], dtype=torch.float, device=device)
                        if visual.geom_type == "box":
                            link_mesh = trimesh.primitives.Box(
                                extents=2*visual.geom_param)
                        elif visual.geom_type == "capsule":
                            link_mesh = trimesh.primitives.Capsule(
                                radius=visual.geom_param[0], height=visual.geom_param[1]*2).apply_translation((0, 0, -visual.geom_param[1]))
                        else:
                            # print(visual.geom_param[0])
                            # print(mesh_path)
                            # link_mesh = trimesh.load_mesh(
                            #     os.path.join(mesh_path, visual.geom_param[0].split(":")[1]+".obj"), process=False)
                            # print(os.path.join(mesh_path, visual.geom_param[0]+".STL"))
                            # link_mesh = trimesh.load_mesh(
                            #     os.path.join(mesh_path, visual.geom_param[0]+".STL"), process=False)
                            path = visual.geom_param[0]
                            if path.startswith("file://"):
                                path = path[len("file://"):]
                            link_mesh = trimesh.load_mesh(path, process=False)
                            if visual.geom_param[1] is not None:
                                scale = (visual.geom_param[1]).to(dtype=torch.float, device=device)
                        vertices = torch.tensor(
                            link_mesh.vertices, dtype=torch.float, device=device)
                        faces = torch.tensor(
                            link_mesh.faces, dtype=torch.float, device=device)
                        pos = visual.offset.to(dtype=torch.float, device=device)
                        vertices = vertices * scale
                        vertices = pos.transform_points(vertices)
                        link_vertices.append(vertices)
                        link_faces.append(faces + n_link_vertices)
                        n_link_vertices += len(vertices)
                    link_vertices = torch.cat(link_vertices, dim=0)
                    link_faces = torch.cat(link_faces, dim=0)
                    self.mesh[body.link.name] = {'vertices': link_vertices,
                                                'faces': link_faces,
                                                }
            for children in body.children:
                build_mesh_recurse(children)
        # print(self.chain._root)
        build_mesh_recurse(self.chain._root)

        self.joints_names = []
        self.joints_lower = []
        self.joints_upper = []

        # def set_joint_range_recurse(body):
        #     if body.joint.joint_type != "fixed":
        #         print(body.joint)
        #         self.joints_names.append(body.joint.name)
        #         self.joints_lower.append(body.joint.range[0])
        #         self.joints_upper.append(body.joint.range[1])
        #     for children in body.children:
        #         set_joint_range_recurse(children)
        # set_joint_range_recurse(self.chain._root)
        # self.joints_lower = torch.stack(
        #     self.joints_lower).float().to(device)
        # self.joints_upper = torch.stack(
        #     self.joints_upper).float().to(device)

        self.hand_pose = None
        self.global_translation = None
        self.global_rotation = None
        self.current_status = None
        
    def pred_hand_pose_mediapipe(self, img: np.ndarray, draw=True):
        # 默认输入RGB
        results = self.hand_mp.process(img) 
        img_draw= img.copy()
        hand_positions = {'right': [], 'left': []}
        if results.multi_hand_world_landmarks:
            for ih, handedness in enumerate(results.multi_handedness):
                # print(handedness)
                joint_positions = np.empty((21, 3)) 
                for i in range(21):
                    joint_positions[i] = np.array([results.multi_hand_world_landmarks[ih].landmark[i].x,
                                                results.multi_hand_world_landmarks[ih].landmark[i].y,
                                                results.multi_hand_world_landmarks[ih].landmark[i].z])
                # https://github.com/google/mediapipe/issues/4785
                # hand_type = 'right' if handedness.classification[0].label == 'Right' else 'left'
                hand_type = 'right' if handedness.classification[0].label == 'Left' else 'left'                          
                hand_positions[hand_type].append(joint_positions)
                if draw:
                    # 在img上添加手的图像 handLms 节点坐标  mpHands.HAND_CONNECTIONS 连接线
                    self.draw_mp.draw_landmarks(img_draw, 
                                                results.multi_hand_landmarks[ih], 
                                                self.hand_mp_sol.HAND_CONNECTIONS)  
        # 低通滤波 写在主函数
        # joint_positions_data_filter = self.filter_k * joint_positions + (1-self.filter_k)*joint_positions_data_filter  
        return hand_positions, img_draw
    
    def transform_hand_pose(self, hand_positions: np.ndarray, rot: np.ndarray) -> np.ndarray:
        # 将每个手部节点的位置向量与旋转矩阵相乘
        transformed_positions = np.dot(rot, hand_positions.T).T
        return transformed_positions    
    
    def extract_endpoints(self, hand_positions):
        finger_endpoints = []
        for finger in self.finger_index:
            finger_endpoint = hand_positions[finger[-1]]
            finger_endpoints.append(finger_endpoint)
        return finger_endpoints


    def plot_hands(self, hand_positions):
                
        if hand_positions['left']:
            self.ax1.clear()
            self.plot_hand_pose_3d(self.ax1, hand_positions['left'][0])
            # rot = np.array([[0,0,-1],[0,-1,0],[-1,0,0]]) # TODO
            # self.plot_hand_pose_3d(self.ax1, self.transform_hand_pose(hand_positions['left'][0],rot))
            
        if hand_positions['right']:
            self.ax2.clear()
            # self.plot_hand_pose_3d(self.ax2, hand_positions['right'][0])
            rot = np.array([[0,0,-1],[1,0,0],[0,-1,0]])
            self.plot_hand_pose_3d(self.ax2, self.transform_hand_pose(hand_positions['right'][0],rot))
        
        self.ax1.set_title('Left Hand') 
        self.ax2.set_title('Right Hand')
        
        self.fig.suptitle('Hand Positions')
        plt.show(block=False)
    
    def plot_hand_pose_3d(self, ax: Axes3D, hand_positions):
        # 绘制手指关节点
        for finger in self.finger_index:
            ax.plot(hand_positions[finger, 0], 
                    hand_positions[finger, 1], 
                    hand_positions[finger, 2], 
                    marker='o', linestyle='-')
        # 加粗显示手指末端节点
        ax.scatter(hand_positions[self.finger_end_index, 0], 
                   hand_positions[self.finger_end_index, 1], 
                   hand_positions[self.finger_end_index, 2], 
                   c='r', s=100)
        # ax.set_box_aspect([1,1,1])
        ax.set_aspect('equal')
        ax.legend(['Index', 'Middle', 'Ring', 'Little', 'Thumb', 'End Points'], loc='best')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
    
    #######################################################################33
        
    def get_trimesh_data(self, i):
        """
        Get full mesh
        
        Returns
        -------
        data: trimesh.Trimesh
        """
        data = trimesh.Trimesh()
        for link_name in self.mesh:
            v = self.current_status[link_name].transform_points(
                self.mesh[link_name]['vertices'])
            if len(v.shape) == 3:
                v = v[i]
            # v = v @ self.global_rotation[i].T + self.global_translation[i]
            v = v.detach().cpu()
            f = self.mesh[link_name]['faces'].detach().cpu()
            data += trimesh.Trimesh(vertices=v, faces=f)
        return data
    
    def set_parameters(self, hand_pose):
        """
        Set translation, rotation, and joint angles of grasps
        
        Parameters
        ----------
        hand_pose: (B, 3+6+`n_dofs`) torch.FloatTensor
            translation, rotation in rot6d, and joint angles
        """
        self.hand_pose = hand_pose
        if self.hand_pose.requires_grad:
            self.hand_pose.retain_grad()
        # self.global_translation = self.hand_pose[:, 0:3]
        # self.global_rotation = robust_compute_rotation_matrix_from_ortho6d(
        #     self.hand_pose[:, 3:9])
        # self.current_status = self.chain.forward_kinematics(
        #     self.hand_pose[:, 9:])
        self.current_status = self.chain.forward_kinematics(
            self.hand_pose[:, :])
#endregion
# def slice_fit(self):
#         if len(self.points) >= 3:
#             # 定义直线模型
#             def linear_model(x, a, b):
#                 return a * x + b
#             # 计算点到直线的距离
#             def distance_to_line(a, b):
#                 return np.abs(a * x - y + b) / np.sqrt(a**2 + 1)
#             def calc_R(xc, yc):
#                 """ 计算s数据点与圆心(xc, yc)的距离 """
#                 return np.sqrt((x-xc)**2 + (y-yc)**2)
#             def f_2(c):
#                 """ 计算半径残余"""
#                 Ri = calc_R(*c)
#                 return Ri - Ri.mean()
            
#             x = self.points[:,0]
#             y = self.points[:,1]             
#             # 拟合直线模型
#             ab_estimate = (1,0)
#             params_linear, _ = curve_fit(linear_model, x, y)
#             # 计算点到直线的距离
#             distances = distance_to_line(*params_linear)
#             # 拟合圆模型
#             center_estimate = (0, 0)# 初始参数猜测
#             center_2, ier = leastsq(f_2, center_estimate)
#             xc_2, yc_2 = center_2
#             Ri_2       = calc_R(*center_2) #每个点到圆心的距离，返回向量
#             R        = Ri_2.mean()
#             # 判断拟合结果
#             error_linear = np.sum(distances**2)#np.sum((y - linear_model(x, *params_linear))**2)
#             error_circle = np.sum((Ri_2 - R)**2)
#             # print('x',x)
#             # print('y',y)
#             print('center_2',center_2)
#             # print('Ri_2',Ri_2)
#             print('R',R)
#             print('error_circle',error_circle)
#             print('error_linear',error_linear)
#             if  error_circle < error_linear  and R < 0.08:
#                 print("The data fits better with a circle.")
#                 theta = np.linspace(0, 2*np.pi, 100)  # 生成角度值
#                 x_draw = xc_2 + R * np.cos(theta)
#                 y_draw = yc_2 + R * np.sin(theta)
#                 plt.plot(x_draw, y_draw, color='blue', label='Circle')                
#             else:
#                 print("The data fits better with a straight line.")                              
#             # 绘制数据和拟合结果
#             plt.scatter(x, y, label='Original Data')
#             plt.plot(x, linear_model(x, *params_linear), color='red', label='Line')
#             plt.xlabel('X')
#             plt.ylabel('Y')
#             plt.legend()
#             plt.grid(True)
#             plt.axis('equal')
#             plt.show()
    



# def slice_fit_draw(self):
#         if len(self.points) >= 3:
#             # 定义直线模型
#             def linear_model(x, a, b):
#                 return a * x + b
#             # 计算点到直线的距离
#             def distance_to_line(c):
#                 a = c[0]
#                 b = c[1]
#                 return np.abs(a * x_l - y_l + b) / np.sqrt(a**2 + 1)
#             def calc_R(xc, yc):
#                 """ 计算s数据点与圆心(xc, yc)的距离 """
#                 return np.sqrt((x_r-xc)**2 + (y_r-yc)**2)
#             def f_2(c):
#                 """ 计算半径残余"""
#                 Ri = calc_R(*c)
#                 return Ri - Ri.mean()
           
#             x = self.points[:,0] ; x_l = self.points[:,0] ; x_r = self.points[:,0]
#             y = self.points[:,1] ; y_l = self.points[:,1] ; y_r = self.points[:,1]
#             for i in range(3):
#                 # 拟合直线模型
#                 ab_estimate = [1, 0]
#                 params_linear, _ = leastsq(distance_to_line, ab_estimate)                
#                 # 计算点到直线的距离
#                 distances = distance_to_line(params_linear)
#                 better_dataid_l =distances < 1.8*distances.mean()
#                 x_l = x_l[better_dataid_l]
#                 y_l = y_l[better_dataid_l]
#                 # 拟合圆模型
#                 center_estimate = (0, 0)# 初始参数猜测
#                 center_2, ier = leastsq(f_2, center_estimate)
#                 xc_2, yc_2 = center_2
#                 Ri_2       = calc_R(*center_2) #每个点到圆心的距离，返回向量
#                 R        = Ri_2.mean()
#                 better_dataid_r = np.abs(Ri_2 - R) < 1.8*np.mean(np.abs(Ri_2 - R))
#                 x_r = x_r[better_dataid_r]
#                 y_r = y_r[better_dataid_r]
#                 if len(x_l) < 3 or len(x_r) < 3:
#                     break
#             # 判断拟合结果
#             error_linear = np.sum(distances**2)#np.sum((y - linear_model(x, *params_linear))**2)
#             error_circle = np.sum((Ri_2 - R)**2)
#             print(params_linear)
#             # print('x',x)
#             # print('y',y)
#             print('center_2',center_2)
#             # print('Ri_2',Ri_2)
#             print('R',R)
#             print('error_circle',error_circle)
#             print('error_linear',error_linear)
#             print('lose points num_l',len(x)-len(x_l))
#             print('lose points num_r',len(x)-len(x_r))
#             if  error_circle < error_linear/20  and R < 0.085 and np.sqrt((xc_2-self.center_point[0])**2+(yc_2-self.center_point[1])**2) < 0.2*R:
#                 #误差要足够小 半径要在区间内 圆心和重心距离不能超过0.2倍半径 才能判断为圆环
#                 self.fit = 'circle'
#                 print("The data fits better with a circle.")
#                 theta = np.linspace(0, 2*np.pi, 100)  # 生成角度值
#                 x_draw = xc_2 + R * np.cos(theta)
#                 y_draw = yc_2 + R * np.sin(theta)
#                 plt.plot(x_draw, y_draw, color='blue', label='Circle')                
#             elif error_circle < error_linear  and R < 0.085 and np.sqrt((xc_2-self.center_point[0])**2+(yc_2-self.center_point[1])**2) < 0.2*R:
#                 #误差没有小很多 其他条件满足 为都有
#                 self.fit = 'circle with line' 
#                 print("The data fits better with circle with line.") 
#                 plt.plot(x, linear_model(x, *params_linear), color='red', label='Line')
#                 theta = np.linspace(0, 2*np.pi, 100)  # 生成角度值
#                 x_draw = xc_2 + R * np.cos(theta)
#                 y_draw = yc_2 + R * np.sin(theta)
#                 plt.plot(x_draw, y_draw, color='blue', label='Circle')
#             else :
#                 self.fit = 'line'
#                 print("The data fits better with a straight line.")
#                 plt.plot(x, linear_model(x, *params_linear), color='red', label='Line')                                
#             # 绘制数据和拟合结果
#             # plt.scatter(x_l, y_l, label='Filter Data')
#             plt.scatter(x, y, label='Original Data')
#             plt.xlabel('X')
#             plt.ylabel('Y')
#             plt.legend()
#             plt.grid(True)
#             plt.axis('equal')
#             plt.show()