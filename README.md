# ccntfps

## 11.25
- demo_env3 提供了其他交互设备的接口 样例见demo_env3_example.py 数据处理样例见server.py
- 由于top级指令存在move后无法攻击的bug 实在无法找到简易的ai指令完成围攻动作 先采用的替代方案添加线程检测unit是否到达目的地

### TODO
- 几个阵型的距离参数尚未调整 阵型较难看
- 包围目前只是简单的走直线 是否需要调整？？？


## 0110
- demo_env用于5*5大地图移动 demo_env2为肖祥写的5*5对战环境 demo_env3为最小粒度的step环境
- vr_version_fps_show_script 为演示流程 network为数据中转，sendeye模拟眼动脑机手势输入 