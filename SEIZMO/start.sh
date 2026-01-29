#!/bin/sh

proc=`ps -ef |grep python |grep "selfy_webserver.py" |awk '{print$2}'`

if [ X"$proc" != X"" ]; then
	echo "[$proc] aleady executed.."
else
	sudo env "PATH=$PATH" nohup python selfy_webserver.py   output.log 2&1
fi

