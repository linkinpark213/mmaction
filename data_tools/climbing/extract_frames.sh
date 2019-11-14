#! /usr/bin/bash env

cd ../
python build_rawframes.py ../data/climbing/videos/ ../data/climbing/rawframes/ --level 2 --ext avi
echo "Raw frames (RGB and tv-l1) Generated"
cd climbing/
