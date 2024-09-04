python -m venv ~/goinfre/pythons
source ~/goinfre/pythons/bin/activate
pip install -r ./requirements.txt

cp ~/goinfre/directory.zip
sha1sum ./directory.zip

unzip ./directory.zip