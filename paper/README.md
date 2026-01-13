# QMCPy JOSS Paper

To render `paper.md` as a PDF, 

* Install DockerHub and login. See also [exec: "docker-credential-desktop.exe": executable file not found in $PATH](https://stackoverflow.com/questions/65896681/exec-docker-credential-desktop-exe-executable-file-not-found-in-path). 

* Run the following from the `QMCSoftware/` directory. 

```bash
docker run --rm     --volume $PWD/paper:/data     --user $(id -u):$(id -g)     --env JOURNAL=joss     openjournals/inara
```

* To count number of words without header and Acknowledgements, run the following from the repository root (QMCSoftware/) in Terminal:

```bash
start=$(grep -n '^# Summary' paper/paper.md | cut -d: -f1) && \
end=$(grep -n '^# Acknowledgements' paper/paper.md | cut -d: -f1) && \
echo Words between lines $start and $end && \
awk -v s=$start -v e=$end 'NR>s && NR<e{print}' paper/paper.md | wc -w
```

<!-- 

* Reviews: https://github.com/openjournals/joss-reviews/issues/9705 
* Slack:   https://speedyreliabl-dvj5497.slack.com/archives/C07PNS0DMDX
* GitHub
	- JOSS branch: https://github.com/QMCSoftware/QMCSoftware/tree/joss
	- PR 460 to merge JOSS to develop: https://github.com/QMCSoftware/QMCSoftware/pull/460

-->