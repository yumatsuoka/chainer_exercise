#!/bin/sh

WORKING_DIRECTORY=`dirname $0`
PYTHON=`which python2.7`
PACKAGE_LIST="$WORKING_DIRECTORY/requirements.txt"

cd "$WORKING_DIRECTORY"

virtualenv -p "$PYTHON" .
if [ $? -ne 0 ]
then
	echo "$PYTHON がインストールされていません。" 1>&2
	exit 1
fi
	
if [ -e "$PACKAGE_LIST" ]
then
	source bin/activate
	pip install -r "$PACKAGE_LIST"
	deactivate
fi
