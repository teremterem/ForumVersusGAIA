pip-compile --upgrade --strip-extras
pip-compile --upgrade --strip-extras dev-requirements.in
./sync_dependencies.sh
