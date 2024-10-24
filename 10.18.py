# test
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
from scipy.signal import butter, filtfilt
import warnings
warnings.filterwarnings("ignore")
import ipdb

# method
def Read_data(filepath,a,b):       
    df = pd.read_excel(filepath)
    kneedata = df.iloc[a:, b].values  
    return kneedata 
def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a
def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y
def jointgetid(m,name):
    joint_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    jointid1 = m.jnt_dofadr[joint_id]
    return jointid1
def getacc(pos):
    return np.concatenate([[0], np.diff(pos, 2), [0]]).flatten()
def getvel(pos):
    return np.concatenate([[0], np.diff(pos, 1)]).flatten()
def printjointid(m):
    for geom_id in range(m.njnt):
        geom_name =mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, geom_id)
        print(f"ID: {geom_id}, Name: {geom_name}")      
def plot_p(x,y1,y2,label1,label2,title='',xlim='Index',ylim="Value"):
    plt.plot(x, y1, label=label1, color='blue', linewidth=2)
    plt.plot(x, y2, label=label2, color='red', linewidth=2)
    # 添加标题和标签
    plt.title(title)
    plt.xlabel(xlim)
    plt.ylabel(ylim)
    # 显示网格
    plt.grid(True)
    # 示图例
    plt.legend()
    # 显示图像
    plt.show()

# main method
def main():
    # constants
    rad=np.pi/180
    m = mujoco.MjModel.from_xml_path("10.18tendonpid.xml")
    d = mujoco.MjData(m)
    kneedata = Read_data("Jointangle.xls",1,10)  
    kneedata=np.concatenate((kneedata, kneedata, kneedata, kneedata))*rad
    hipdata= Read_data("Jointangle.xls",1,7)  
    hipdata=np.concatenate((hipdata,hipdata, hipdata,hipdata))*rad
    cutoff = 6  # 截止频率
    order = 4
    fs = 100  # 采样频率
    kneedata = lowpass_filter(kneedata, cutoff, fs, order)
    hipdata = lowpass_filter(hipdata, cutoff, fs, order)

    plot_p(x=np.arange(len(kneedata)),y1=kneedata,y2=hipdata,label1='kneedata curve',
        label2='hipdata curve',title='qfrc_data Curve')
    
    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        e=0
        max1=0
        min1=100
        qacc_kneedata = []
        qacc_hipdata = []
        qfrc_kneedata1 = []
        qfrc_femurdata1 = []
        qfrc_knee_angle_r_translation2 = []
        qfrc_knee_angle_r_translation1 = []
        qfrc_knee_angle_r_rotation2 = []
        qfrc_knee_angle_r_rotation3 = []

        jointfemurid = jointgetid(m,  "femur_angle_r")
        jointkneeid = jointgetid(m,  "knee_angle_r")
        kneerotation2id = jointgetid(m,  "knee_angle_r_rotation2")
        kneerotation3id = jointgetid(m,  "knee_angle_r_rotation3")
        kneetranslation1id = jointgetid(m,  "knee_angle_r_translation1")
        kneetranslation2id = jointgetid(m,  "knee_angle_r_translation2")

        kneeacc=getacc(kneedata)
        hipacc=getacc(hipdata)
        hipvel=getvel(hipdata)
        kneevel=getvel(kneedata)

        for f in range(0, 399):
            f=f+1
            d.joint("knee_angle_r").qpos=kneedata[f]
            d.joint("femur_angle_r").qpos=-hipdata[f]
            d.joint("knee_angle_r").qvel=kneevel[f]
            d.joint("femur_angle_r").qvel=-hipvel[f]
            d.joint("knee_angle_r").qacc=kneeacc[f]
            d.joint("femur_angle_r").qacc=-hipacc[f]
            mujoco.mj_inverse(m,d)
            qfrc_knee_angle_r_translation2.append(d.joint(kneetranslation2id).qfrc_inverse.copy())
            qfrc_knee_angle_r_translation1.append(d.joint(kneetranslation1id).qfrc_inverse.copy())
            qfrc_knee_angle_r_rotation2.append(d.joint(kneerotation2id).qfrc_inverse.copy())
            qfrc_knee_angle_r_rotation3.append(d.joint(kneerotation3id).qfrc_inverse.copy())
            qfrc_kneedata1.append(d.joint(jointkneeid).qfrc_inverse.copy())
            qfrc_femurdata1.append(d.joint(jointfemurid).qfrc_inverse.copy())
        plot_p(np.arange(len(kneedata)-1),qfrc_kneedata1,qfrc_femurdata1,
        'qfrc_kneedata1','qfrc_femurdata1','qfrc_data Curve111')
        time.sleep(3)
        while viewer.is_running() and d.time < 50:
            print("iteration count:",e)
            m.opt.timestep=0.01
            e=e+1
            if e>len(kneedata)-1:
                break
            if e==0:
                d.joint("knee_angle_r").qpos=0
                d.joint("femur_angle_r").qpos=0
            d.qfrc_applied[jointfemurid]=qfrc_femurdata1[e]
            d.qfrc_applied[jointkneeid]=qfrc_kneedata1[e]
            print (f"femur_angle_r:{d.joint("femur_angle_r").qpos}")
            print (f"qfrc_kneedata:",qfrc_kneedata1[e])
            mujoco.mj_step(m,d)
            qacc_kneedata.append(d.joint("knee_angle_r").qacc.copy())  
            qacc_hipdata.append(d.joint("femur_angle_r").qacc.copy()) 
            points = []
            colors = []
            ww=[]
            step_start = time.time()
            print (f"d.time:{d.time}")
            print (f"time.time:{time.time}")
            print (f"d.ncon:{d.ncon}")
            for i in range(d.ncon):
                contact = d.contact[i]
                contact_force = np.zeros(6)
                mujoco.mj_contactForce(m, d, i, contact_force)  #读取接触力至contact_force公式
                normal_force_magnitude = np.linalg.norm(contact_force[:3])   #取极值
                ww.append(normal_force_magnitude)   
                max1=max(max1,normal_force_magnitude)     
                min1=min(min1,normal_force_magnitude)         
            ww=np.array(ww)                       
            mask0 = ww>.1               #mask0取大于0.1的接触力显示 
            if d.ncon!=0 and True in mask0:
                norm = plt.Normalize(vmin=ww[mask0].min(), vmax=ww.max())  #norm标量化接触力最值
                jet_colors = plt.cm.jet 
                for w in range(d.ncon):   
                    if not mask0[w]:
                        continue
                    if w>199:
                        break
                    for z in range(w + 1, 199): #对于没有接触力的部分,site隐形
                        m.site_rgba[z] = [0, 1.0, 1.0, 1.0]
                        m.site_pos[z] = [0, 0,0]
                    color = jet_colors(norm(ww[w]))   #用norm标量表示接触力，然后color取接触力的颜色值
                    m.site_pos[w]=d.contact.pos[w]    #移动site位置至contact位置
                    m.site_rgba[w, :3] = color[:3]    #site的颜色变为color颜色
                    points.append(d.contact.pos[w])   #记录总的接触点
                    colors.append(color[:3])          #记录总的颜色
                points = np.array(points)
                colors = np.array(colors) 
                x=d.contact.pos[:, 0]                 
                y=d.contact.pos[:, 1]
                z=ww
                xi = np.linspace(min(x), max(x), 100)
                yi = np.linspace(min(y), max(y), 100)
                xi, yi = np.meshgrid(xi, yi)
                if ww[mask0].size>20:
                    viewer.sync()
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=50)
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
                cbar = plt.colorbar(sc, ax=ax)
                cbar.set_label('force mean')  # 设置 colorbar 的标签
                cbar.ax.set_yticklabels([f'{norm.vmin:.2f}', f'{(norm.vmax/5):.2f}',f'{(norm.vmax/2.5):.2f}',f'{(norm.vmax/1.67):.2f}',f'{(norm.vmax/1.25):.2f}',f'{norm.vmax:.2f}'])  # 设置 colorbar 的刻度标签
                cbar.ax.set_position([0.85, 0.1, 0.03, .8])  # [x, y, width, height]
                plt.show()
            viewer.sync()
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step+0.02)


if __name__ == '__main__': 
    main()


  





