### 这是为无人机表型平台开发的项目。以下为项目结构

该项目包含5个功能模块，通过模块间的工作流，实现无人机图像对育种小区性状的识别。

uav_pheno_web/
├── app.py                    # Flask 应用主文件
├── auth/                     # 认证模块
│   ├── __init__.py           # 认证模块的初始化文件
│   └── login.py              # 登录功能实现
├── config/                   # 配置文件目录
│   ├── __init__.py           # 配置模块的初始化文件
│   └── login.py              # 存储登录信息（用户名和密码）
├── controller/               # 控制器模块
│   ├── __init__.py           # 控制器模块的初始化文件
│   ├── metashape.py          # 处理图像拼图的逻辑
│   ├── plot_grid_segment.py  # 处理网格绘制的逻辑（调用 R 脚本）
│   ├── plot_grid_segment.R   # 处理网格绘制的 R 脚本
│   ├── plot_image_mask.py    # 图像掩膜处理逻辑
│   ├── image_predict.py      # 图像推理逻辑
│   ├── result_show.py        # 结果可视化逻辑（调用 result_show.R）
│   └── result_show.R         # 结果可视化的 R 语言脚本
├── resources/                # 资源模块
│   ├── __init__.py           # 资源模块的初始化文件
│   ├── metashape_resource.py # 定义 API 路由和处理请求
│   ├── plot_seg_resource.py  # 定义 plot_seg API 路由和处理请求
│   ├── plot_image_mask_resource.py # 图像掩膜 API 路由
│   ├── image_predict_resource.py   # 图像推理 API 路由
│   └── result_show_resource.py     # 结果可视化 API 路由
├── models/                   # 模型模块
│   ├── __init__.py           # 模型模块的初始化文件
│   ├── IntegrateNet.py       # IntegrateNet 模型
│   ├── V3liteNet.py          # V3liteNet 模型
│   ├── V3segnet.py           # V3segnet 模型
│   └── V3segplus.py          # V3segplus 模型
├── weights/                  # 权重文件夹
│   ├── maize/                # 玉米权重文件夹
│   │   ├── seedling_count/   # 玉米幼苗计数模型权重
│   │   │   ├── V3liteNet_model_best.pth.tar  # V3liteNet 幼苗计数模型权重
│   │   │   ├── V3segnet_model_best.pth.tar   # V3segnet 幼苗计数模型权重
│   │   │   └── IntegrateNet_model_best.pth   # IntegrateNet 幼苗计数模型权重
│   │   ├── tassel_count/     # 玉米雄穗计数模型权重
│   │   │   ├── V3liteNet_model_best.pth.tar  # V3liteNet 雄穗计数模型权重
│   │   │   ├── V3segnet_model_best.pth.tar   # V3segnet 雄穗计数模型权重
│   │   │   └── IntegrateNet_model_best.pth   # IntegrateNet 雄穗计数模型权重
│   │   ├── height/           # 玉米高度模型权重（替换为线性回归模型参数）
│   │   │   └── height_model_params.npy  # 株高线性回归模型参数
│   │   └── yield/            # 玉米产量模型权重（替换为线性回归模型参数）
│   │       └── yield_model_params.npy   # 产量线性回归模型参数
│   └── rice/                 # 水稻权重文件夹
│       ├── seedling_count/
│       │   ├── V3liteNet_model_best.pth.tar
│       │   ├── V3segnet_model_best.pth.tar
│       │   └── IntegrateNet_model_best.pth
│       ├── height/           # 水稻高度模型权重（替换为线性回归模型参数）
│       │   └── height_model_params.npy  # 株高线性回归模型参数
│       └── yield/            # 水稻产量模型权重（替换为线性回归模型参数）
│           └── yield_model_params.npy   # 产量线性回归模型参数
├── data/                     # 数据文件夹
│   ├── raw_uav_img/          # 输入文件夹，存放无人机图像
│   ├── out/                  # 输出文件夹，存放处理后的结果
│   └── pt4.shp               # 输入点文件
├── log/                      # 日志文件夹
│   ├── access.log            # 访问日志
│   └── error.log             # 错误日志
├── Metashape-2.1.2-cp39-cp39-linux_x86_64.whl  # Metashape wheel 文件
├── celery_worker.py          # Celery Worker 文件
├── requirements.txt          # 项目依赖文件
└── Dockerfile                # Dockerfile