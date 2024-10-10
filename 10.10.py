# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 12:11:24 2024

@author: yizhe
"""
import time
import numpy as np
import mujoco
import mujoco.viewer as viewer
import math
import os
import matplotlib.pyplot as plt
import scipy
import mediapy as media
import pandas as pd
import threading
import sympy
from copy import deepcopy
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from scipy.spatial import Delaunay
def file1(filepath,a,b):        #读取excel
    df = pd.read_excel(filepath)
    kneedata = df.iloc[a:, b].values  # iloc[1:] 表示从第二行开始（即去掉第一个数据），8 表示第9列（因为索引从0开始）
    return kneedata #返回第11列的第二个数据以后的一列数据

rad=3.1415926/180
m = mujoco.MjModel.from_xml_path(r"C:/Users\yizhe\Desktop\mujoco3.1.3\model\pidtest\10.10.xml")
d = mujoco.MjData(m)
kneedata = file1(r"C:\Users\yizhe\Desktop\mujoco3.1.3\model\tibiafumur\Jointangle.xls",1,10)  # iloc[1:] 表示从第二行开始（即去掉第一个数据），8 表示第9列（因为索引从0开始）
# kneedata=np.concatenate((kneedata[::-1], kneedata,kneedata[::-1], kneedata,kneedata[::-1]))
kneedata=np.concatenate((kneedata, kneedata, kneedata, kneedata))
hipdata= file1(r"C:\Users\yizhe\Desktop\mujoco3.1.3\model\tibiafumur\Jointangle.xls",1,9)  # iloc[1:] 表示从第二行开始（即去掉第一个数据），8 表示第9列（因为索引从0开始）
# hipdata=np.concatenate((hipdata[::-1], hipdata,hipdata[::-1], hipdata,hipdata[::-1]))
hipdata=np.concatenate((hipdata,hipdata, hipdata,hipdata))
x = np.arange(len(kneedata))

# 绘制曲线图
plt.plot(x, kneedata, label='kneedata curve', color='blue', linewidth=2)
plt.plot(x, hipdata, label='hipdata curve', color='red', linewidth=2)
# 添加标题和标签
plt.title('qfrc_data Curve')
plt.xlabel('Index')
plt.ylabel('Value')
# 显示网格
plt.grid(True)
# 示图例
plt.legend()
# 显示图像
plt.show()
mujoco.mj_printData(m,d,r"C:/Users\yizhe\Desktop\mujoco3.1.3\model\text1.txt")




with mujoco.viewer.launch_passive(m, d) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  time.sleep(3)
  p=0
  e=0
  max1=0
  min1=100
  m.site_rgba[:, 3] = .3
  m.site_size[:, 0] = .008
  qacc_kneedata = []
  qacc_hipdata = []
  qfrc_kneedata = []
  qfrc_hipdata = []
  joint_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "femur_angle_r")
  joint_dof_id = m.jnt_dofadr[joint_id]
  joint_id1 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "knee_angle_r")
  joint_dof_id1 = m.jnt_dofadr[joint_id1]

  while viewer.is_running() and d.time < 50:
    print("e2:",e)
    e=e+1

    if e>len(kneedata):
        break

    d.joint("knee_angle_r").qpos=kneedata[e]*rad
    d.joint("femur_angle_r").qpos=-hipdata[e]*rad
    print (f"femur_angle_r:{d.joint("femur_angle_r").qpos}")
    mujoco.mj_forward(m,d)
    qacc_kneedata.append(d.joint("knee_angle_r").qacc.copy())  
    qacc_hipdata.append(d.joint("femur_angle_r").qacc.copy())      # 获取当前时刻的 qacc
    
    
    mujoco.mj_inverse(m,d)
    qfrc_kneedata.append(d.joint("knee_angle_r").qfrc_inverse.copy())
    qfrc_hipdata.append(d.joint("femur_angle_r").qfrc_inverse.copy())
    
    points = []
    colors = []
    ww=[]
    step_start = time.time()
    print (f"time:{d.time}")
    print (f"d.ncon:{d.ncon}")
    for i in range(d.ncon):
      contact = d.contact[i]
      contact_force = np.zeros(6)
      mujoco.mj_contactForce(m, d, i, contact_force)
      normal_force_magnitude = np.linalg.norm(contact_force[:3])
      ww.append(normal_force_magnitude)          #ww 读取所有接触力绝对值大小并存储          
      
      if max1<normal_force_magnitude:
          max1=normal_force_magnitude
      if min1>normal_force_magnitude:
          min1=normal_force_magnitude            #取接触力的极值
    ww=np.array(ww)                       

        # print(f"Contact {i}:")
        # print(f"pos:{d.contact.pos[i]}")
        # print(f"  Normal force magnitude: {normal_force_magnitude}")
    # print(d.time)
    # print(d.ncon)
    # print(len(d.xpos))
    # m.site_pos[0]=[1+d.time,1,1] #移动

    m.site_pos=0
    mask0 = ww>.1               #mask0取大于0.1的接触力显示 
    if d.ncon!=0 and True in mask0:
        norm = plt.Normalize(vmin=ww[mask0].min(), vmax=ww.max())  #norm标量化接触力最值
        jet_colors = plt.cm.jet 
        # print(np.max(ww))
        # print("ww:",ww)
        # print(d.ncon)
        for w in range(d.ncon):          #w 遍历接触点个数
            if not mask0[w]:
                continue
            if w>m.nsite-1:
                break
            for z in range(w + 1, m.nsite):
        # 将 m.site_rgba[n, :3] 设置为白色
                # m.site_pos[z]=[0., 0., 0.]    #对于没有接触力的部分,site隐形
                m.site_rgba[z] = [0, 1.0, 1.0, 1.0]
            color = jet_colors(norm(ww[w]))   #用norm标量表示接触力，然后color取接触力的颜色值
            m.site_pos[w]=d.contact.pos[w]    #移动site位置至contact位置，如果site位置多余contact位置？
            m.site_rgba[w, :3] = color[:3]    #site的颜色变为color颜色
            points.append(d.contact.pos[w])   
            colors.append(color[:3])
            
               
        points = np.array(points)
        colors = np.array(colors) 
        x=d.contact.pos[:, 0]
        y=d.contact.pos[:, 1]
        
        z=ww
        xi = np.linspace(min(x), max(x), 100)
        yi = np.linspace(min(y), max(y), 100)
        xi, yi = np.meshgrid(xi, yi)
        if ww[mask0].size>20:           
            # mujoco.mj_step(m, d)
            viewer.sync()
     # 绘制三维点云图
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=50)
        
        # 设置标签
        ax.set_xlim([min(points[:, 0]), max(points[:, 0])])
        ax.set_ylim([min(points[:, 1]), max(points[:, 1])])
        ax.set_zlim([min(points[:, 2]), max(points[:, 2])])
        ax.set_xticks(np.linspace(min(points[:, 0]), max(points[:, 0]), 4))
        ax.set_yticks(np.linspace(min(points[:, 1]), max(points[:, 1]),  4))
        ax.set_zticks(np.linspace(min(points[:, 2]), max(points[:, 2]), 4))
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Contact Scatter Points')
        
        # cbar = plt.colorbar(sc, ax=ax)
        # cbar.set_label('WW Value')  # 设置 colorbar 的标签
        # cbar.set_ticks([norm.vmin, norm.vmax])  # 设置 colorbar 的刻度值
        # cbar.ax.set_yticklabels([f'{norm.vmin:.2f}', f'{norm.vmax:.2f}'])  # 设置 colorbar 的刻度标签
        # cbar.ax.set_position([0.85, 0.1, 0.03, 0.8])  # [x, y, width, height]
        # # 显示图形
        plt.show()

        

    
      # mujoco.mj_step(m, d)

    viewer.sync()
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step+0.02)
# time.sleep(time_until_next_step+0.03)



df = pd.DataFrame(qfrc_kneedata, columns=['qfrc_kneedata'])
file_path = r"C:\Users\yizhe\Desktop\mujoco3.1.3\model\tibiafumur\qfrc_kneedata.xlsx"

df = pd.DataFrame(qfrc_hipdata, columns=['qfrc_hipdata'])
file_path = r"C:\Users\yizhe\Desktop\mujoco3.1.3\model\tibiafumur\qfrc_hipdata.xlsx"

df.to_excel(file_path, index=False)

print(f"qfrc_data 已成功写入到 {file_path}")  
  





