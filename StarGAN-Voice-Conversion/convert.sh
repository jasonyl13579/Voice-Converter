#!/bin/bash

name=1
mkdir toWAV
for i in {1..164}
do
	FILE=$(printf "Home %d.m4a" $i)
	if test -f "$FILE"; then
		echo "$FILE exist"
		echo "$name.wav"
		A=$(printf 1%05d.wav $name)
		ffmpeg -i "$FILE" -ab 128k -ac 1 -ar 16000 ./toWAV/$A
		((name++))
	else
		echo "$FILE does not exist"
	fi

done
