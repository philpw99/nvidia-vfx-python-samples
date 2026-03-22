REM 先要安装好所有 python 的东西。
REM 然后这里自己修改路径，一次可以拖一个文件到这个批处理文件
I:
cd \AI\nvidia-vfx-python-samples\
call .venv\Scripts\activate.bat
rem %~dpn1 file name with full path, %~x1 extension
python .\video_super_resolution.py -i %1 -o "%~dpn1-out%~x1"
pause