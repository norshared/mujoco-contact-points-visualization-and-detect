# mujoco-contact-points-visualization-and-detect
A python algrithom that can show knee joint contact scatter points in both 3d point cloud map and mujoco software.


**Variable**
ww:   All the mean value of contact points

w:    Walk through the length of contact points(d.ncon)

normal_force_magnitude:  the mean value of a contact


**Function**
 mujoco.mj_contactForce(m, d, i, contact_force)  #读取接触力至contact_force公式
 
normal_force_magnitude = np.linalg.norm(contact_force[:3])   #取极值
