# QMCPy JOSS Paper

To render `paper.md` as a PDF, 

* Install DockerHub and login. See also [exec: "docker-credential-desktop.exe": executable file not found in $PATH](https://stackoverflow.com/questions/65896681/exec-docker-credential-desktop-exe-executable-file-not-found-in-path). 

* Run the following from the `QMCSoftware/` directory. 

```bash
docker run --rm     --volume $PWD/paper:/data     --user $(id -u):$(id -g)     --env JOURNAL=joss     openjournals/inara
```


<!-- 

* Reviews: https://github.com/openjournals/joss-reviews/issues/9705 
* Slack:   https://speedyreliabl-dvj5497.slack.com/archives/C07PNS0DMDX
* GitHub
	- JOSS branch: https://github.com/QMCSoftware/QMCSoftware/tree/joss
	- PR 460 to merge JOSS to develop: https://github.com/QMCSoftware/QMCSoftware/pull/460

-->