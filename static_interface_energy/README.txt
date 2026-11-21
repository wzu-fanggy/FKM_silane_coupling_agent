Static Interface Interaction Energy Workspace
=============================================

目标
----
对平衡后的 FKM/SiO2 复合体系做多帧 NVT 采样，逐帧计算 FKM-SiO2
非键相互作用能，并拆分为 vdW 与静电。

当前文件
--------
data/nve_verify.data            平衡复合构象 (NVE verify final)
scripts/in.nvt_200ps.lammps     200 ps NVT 多帧输出脚本
scripts/in.nvt_200ps_gpu.lammps 200 ps NVT GPU 加速版

运行 200 ps NVT
---------------
cd F:\Lammps-MS\FKM-SIO2-TEST\static_interface_energy\scripts
lmp -in in.nvt_200ps.lammps

GPU 版
------
cd F:\Lammps-MS\FKM-SIO2-TEST\static_interface_energy\scripts
lmp -in in.nvt_200ps_gpu.lammps

输出
----
outputs/nvt_200ps_traj.dump     每 1 ps 一帧，共 200 帧
outputs/nvt_200ps_monitor.txt   温度/能量/体积/密度监控
outputs/nvt_200ps.data          结束构象
outputs/nvt_200ps.restart       结束 restart
outputs/nvt_200ps_gpu.*         GPU 版对应输出

下一步
------
1. 用 nvt_200ps_traj.dump rerun 计算 E_total（原始电荷）；
2. 生成 charge_zero.data，rerun 计算 E_vdW；
3. E_ele = E_total - E_vdW；
4. 输出逐帧 CSV 和 mean ± std。
