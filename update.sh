cd /home/ai2/work/OPENFGL
find . -type d -name "__pycache__" -exec rm -r {} +
git add .
git commit -m "update"
git push -u origin master