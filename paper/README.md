# QMCPy JOSS Paper

To render `paper.md` as a PDF, run the following from the `QMCSoftware/` directory. 

```bash
docker run --rm     --volume $PWD/paper:/data     --user $(id -u):$(id -g)     --env JOURNAL=joss     openjournals/inara
```

<!-- 

* Zenodo page: https://zenodo.org/records/17889172
* PR#460: https://github.com/QMCSoftware/QMCSoftware/pulls

-->