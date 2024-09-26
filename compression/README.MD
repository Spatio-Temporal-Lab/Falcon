```bash
nvcc FOR2.cu -o FOR2
./FOR2
nsys profile ./FOR2 #运行可执行生成性能报告
nsys-ui #打开图形界面
nsys stats report1.nsys-rep #统计数据
```
