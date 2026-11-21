静态界面相互作用能计算说明
========================

输入
----
1. nvt_200ps_gpu_traj.dump : 200 ps NVT 轨迹（201 帧，含 t=0）
2. 只取最后 50 帧（t=150-200 ps），存为 last50.dump
3. 原始电荷 data : data/nve_verify.data
4. 零电荷 data   : outputs/data_zero.data（用 make_charge_zero.py 生成）

计算方法
--------
1. rerun 1（E_total）：
   in.rerun_interface_total.lammps
   用原始电荷 data + last50.dump 回放，
   compute group/group FKM group/group SIO2 pair yes kspace yes
   -> 每帧输出 FKM-SiO2 总非键相互作用能（pair + kspace）。

2. rerun 2（E_vdw）：
   in.rerun_interface_vdw.lammps
   用零电荷 data + last50.dump 回放，
   compute group/group FKM group/group SIO2 pair yes
   -> 每帧输出纯 vdW（LJ）相互作用能。

3. E_ele = E_total - E_vdw
   因为零电荷体系没有静电，kspace 项为 0，
   所以相减得到纯静电贡献。

4. 统计：对 50 帧取 mean ± std。

结果（kcal/mol）
----------------
E_total = -1302.29 ± 15.40
E_vdw   = -969.46 ± 9.30
E_ele   = -332.83 ± 14.15

脚本
----
make_charge_zero.py        生成零电荷 data
extract_last_frames.py     抽取轨迹最后 N 帧
analyze_interface_energy.py 解析 rerun 日志并出 CSV/图
