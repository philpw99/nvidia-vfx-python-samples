I:
cd \AI\nvidia-vfx-python-samples\
call .venv\Scripts\activate.bat
rem %~dpn1 file name with full path, %~x1 extension
python .\video_super_resolution.py -i %1 -o "%~dpn1-out%~x1"
pause