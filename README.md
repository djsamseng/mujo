# Mujo

`$ conda create --name mujo python=3.8`
`$ conda activate mujo`
`$ pip install -r requirements.txt`


To call `env.render(w, h)` then:
```bash
$ LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so python examples/disco_fetch.py
``` 

To call `env.sim.render(w, h)` then:
```bash
$ unset LD_PRELOAD
```