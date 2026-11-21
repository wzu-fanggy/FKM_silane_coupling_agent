KH570 静态界面相互作用能
========================

目标：计算 KH570 接枝体系中 FKM 与“接枝 SiO2 复合体”的非键相互作用能，
并拆分为 vdW 与静电。

输入
----
data/nve_verify_relabeled_centered.data   重映射 + 质心居中数据（17 种类型）
outputs/last50.dump                       最后 50 帧轨迹（来自 200 ps NVT）

类型
----
FKM  : 1 2 3 4 8 10
SIO2 : 5 6 7 9 11 12 13 14 15 16 17（接枝复合体）

计算
----
1. python outputs/make_charge_zero.py data/nve_verify_relabeled_centered.data outputs/data_zero.data
2. lmp -in scripts/in.rerun_interface_total.lammps -log outputs/log.total50.txt
3. lmp -in scripts/in.rerun_interface_vdw.lammps   -log outputs/log.vdw50.txt
4. python outputs/analyze_interface_energy.py

输出
----
interface_energy_last50.csv
interface_energy_last50.png
