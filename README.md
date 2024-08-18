run this :
```bash
export PYTHONPATH="/$(pwd):$PYTHONPATH"
```
```bash
http -f POST \
    http://localhost:8001 \
    context@data/Academic-CV-V1.pdf\
    query="What are the skills"
```
